from argparse import Namespace
import datetime
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.configs import parse_args
from src.logger import get_logger
from src.models.full import HypernetRLMIL
from src.results.metrics import load_rl_model
from src.trainers.base import Trainer
from src.trainers.hypernet import HypernetRLMILTrainer
from src.trainers.util import create_net_container, get_dataloaders, get_model, load_mil_model_from_config, prepare_data
from src.util.timing import TimingAnalyzer
from src.utils import (
    get_model_save_directory,
    get_balanced_weights,
    EarlyStopping,
    load_json, save_json,
    set_seed
)

import copy


# TODO: Make model saving and loading work properly with different trainer classes
""" def load_model_from_config(mil_config_file, rl_config_file, rl_model_file):
    # TODO: make create_mil_model compatible with dictionary input
    mil_config = load_json(mil_config_file)
    rl_config = load_json(rl_config_file)
    task_model = create_mil_model(mil_config)
    policy_network = PolicyNetwork(task_model=task_model,
                                   state_dim=rl_config['state_dim'],
                                   hdim=rl_config['hdim'],
                                   learning_rate=0,
                                   device="cuda:0" if torch.cuda.is_available() else "cpu")
    policy_network.load_state_dict(torch.load(rl_model_file))
    
    return policy_network """


def train(
        trainer: Trainer,
        optimizer,
        scheduler,
        early_stopping,
        train_dataloader,
        eval_dataloader,
        test_dataloader,
        device,
        bag_size,
        epochs,
        no_wandb,
        train_pool_size,
        eval_pool_size,
        test_pool_size,
        rl_model,
        prefix,
        epsilon,
        reg_coef,
        sample_algorithm,
        warmup_epochs=0,
        run_name=None,
        task_type='classification',
        only_ensemble=False, 
):  
    metric = 'f1' if task_type == 'classification' else 'r2'

    torch.multiprocessing.set_sharing_strategy('file_system')
    
    # wandb.watch(policy_network.actor, log="all", log_freq=100, log_graph=True)
    if not no_wandb and not only_ensemble:
        log_dict = trainer.get_first_batch_info(eval_dataloader, device, bag_size, sample_algorithm)
        wandb.log(log_dict)
    
    # logger.info(f"Training model started ....")

    net_ct: HypernetRLMIL = trainer.net_container
    initial_hyper_weights = copy.deepcopy(net_ct.hyper.state_dict())
    initial_stored_weights = net_ct.policy_weights.clone().detach()
    initial_debias_weights = copy.deepcopy(net_ct.debiasing_model.state_dict())

    timer = TimingAnalyzer()

    for epoch in range(epochs):
        log_dict = {}
        warmup = epoch < warmup_epochs
        with timer.context_category("Training"):
            total_loss, policy_loss, value_loss, mil_loss, reg_loss, bias_loss, preference = trainer.episode(
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                bag_size=bag_size,
                train_pool_size=train_pool_size,
                only_ensemble=only_ensemble,
                epsilon=epsilon,
                reg_coef=reg_coef,
                sample_algorithm=sample_algorithm,
                timer=timer,
            )
        timer.next_category("Evaluation")
        # logger.info(f"Finished epoch {epoch}")
        # if not no_wandb and not only_ensemble:
        #     for indx, layer in enumerate(policy_network.actor.actor):
        #         if isinstance(layer, torch.nn.Linear):
        #             wandb.log({f"parameters/actor_{indx}_weight": wandb.Histogram(layer.weight.cpu().detach().numpy().tolist()),
        #                     f"parameters/actor_{indx}_bias": wandb.Histogram(layer.bias.cpu().detach().numpy().tolist())})
        #             wandb.log({f"gradients/actor_{indx}_weight": wandb.Histogram(layer.weight.grad.cpu().detach().numpy().tolist()),
        #                     f"gradients/actor_{indx}_bias": wandb.Histogram(layer.bias.grad.cpu().detach().numpy().tolist())})
        timer.sub_category("Eval Loader")
        trainer.net_container.policy.eval()
        # eval_data = policy_network.select_from_dataloader(eval_dataloader, bag_size)
        eval_pool = trainer.create_pool_data(eval_dataloader, bag_size, eval_pool_size, random=only_ensemble)
        reward, eval_loss, ensemble_reward = trainer.expected_reward_loss(eval_pool)
        
        early_stopping(ensemble_reward, trainer.net_container)

        if not no_wandb:
            timer.next_category("Train Loader")
            with timer.context_category("Pool Creation"):
                train_pool = trainer.create_pool_data(train_dataloader, bag_size, eval_pool_size, random=only_ensemble)
            with timer.context_category("Expected Reward Loss"):
                train_reward, _, train_ensemble_reward = trainer.expected_reward_loss(train_pool)
            timer.sub_category("Until best model test")
            log_dict.update({"train/total_loss": total_loss,
                        "train/policy_loss": policy_loss,
                        "train/value_loss": value_loss,
                        "train/reg_loss": reg_loss,
                        "train/bias_loss": bias_loss,
                        "train/preference": preference,
                        "train/mil_loss": mil_loss,
                        "eval/avg_mil_loss": eval_loss,
                        f"train/avg_{metric}": train_reward,
                        f"train/ensemble_{metric}": train_ensemble_reward,
                        f"eval/avg_{metric}": reward,
                        f"eval/ensemble_{metric}": ensemble_reward})

            # log action probabilities
            if not only_ensemble:
                batch_log_dict = trainer.get_first_batch_info(eval_dataloader, device, bag_size, sample_algorithm)
                log_dict.update(batch_log_dict)
         
            # log best model based on early stopping
            if early_stopping.counter == 0:
                log_dict.update({"best/eval_avg_mil_loss": eval_loss,
                            f"best/eval_avg_{metric}": reward,
                            f"best/eval_ensemble_{metric}": ensemble_reward})
            wandb.log(log_dict)

            timer.next_category("Best model test")

        if run_name:  # sweep
            global BEST_REWARD
            # print(f"ensemble rewards: {ensemble_reward:.6f}, Best rewaed: {BEST_REWARD:.6f}, Reward: {reward:.6f}")
            if ensemble_reward > BEST_REWARD:
                logger.info(
                    f"Found the best model in all of sweep runs in sweep run {run_name} at epoch {epoch}. ensemble F1 "
                    f"increased ({BEST_REWARD:.6f} --> {ensemble_reward:.6f})."
                )
                best_sweep_config = {
                    "critic_learning_rate": args.critic_learning_rate,
                    "actor_learning_rate": args.actor_learning_rate,
                    "learning_rate": args.learning_rate,
                    "epoch": args.epochs,
                    "hdim": args.hdim,
                }
                logger.info(
                    f"Saving the model in run {run_name}, with parameters config={best_sweep_config}"
                )
                BEST_REWARD = ensemble_reward
                torch.save(
                    trainer.net_container.state_dict(),
                    os.path.join(
                        early_stopping.models_dir,
                        "sweep_best_model.pt",
                    ),
                )
                # TODO: move this part to utils
                best_model_config = {}
                args_dict = vars(args)
                config_dict = dict(best_sweep_config)
                for key in set(args_dict.keys()).union(config_dict.keys()):
                    if key == "device": continue
                    if key in args_dict and key in config_dict:
                        best_model_config[key] = config_dict[key]
                    elif key in args_dict:
                        best_model_config[key] = args_dict[key]
                    else:
                        best_model_config[key] = config_dict[key]
                save_json(
                    path=os.path.join(early_stopping.models_dir, "sweep_best_model_config.json"),
                    data=best_model_config
                )
                trainer.net_container.policy.eval()
                # from IPython import embed; embed();
                test_pool = trainer.create_pool_data(test_dataloader, bag_size, test_pool_size, random=only_ensemble)
                test_avg_reward, test_loss, test_ensemble_reward = trainer.expected_reward_loss(test_pool)

                train_pool = trainer.create_pool_data(train_dataloader, bag_size, eval_pool_size, random=only_ensemble)
                train_reward, _, train_ensemble_reward = trainer.expected_reward_loss(train_pool)
            
                dictionary = {
                    "model": "rl-" + args.baseline,
                    "embedding_model": args.embedding_model,
                    "bag_size": args.bag_size,
                    "label": args.label,
                    "seed": args.random_seed,
                    "test/loss": test_loss,
                    f"test/{metric}": None,
                    f"test/avg-{metric}": test_avg_reward,
                    f"test/ensemble-{metric}": test_ensemble_reward,
                    f"train/avg-{metric}": train_reward,
                    f"train/ensemble-{metric}": train_ensemble_reward,
                    f"eval/avg-{metric}": reward,
                    f"eval/ensemble-{metric}": ensemble_reward
                }
                if task_type == 'classification':
                    dictionary.update({"test/accuracy": None,
                                       "test/precision": None,
                                       "test/recall": None,
                                       })

                save_json(os.path.join(early_stopping.models_dir, "results.json"), dictionary)
        # when warmup is done, reset early stopping
        if warmup_epochs + 1 == epoch:
            early_stopping.counter = 0
            early_stopping.early_stop = False
        # early stopping after warmup
        if early_stopping.early_stop and not warmup:
            logger.info(f"Early stopping at epoch {epoch} out of {epochs}")
            break

        timer.up()
        timer.up_next_category("Diagnostics")

        def get_ratios(current: dict[str, torch.Tensor], initial: dict[str, torch.Tensor]):
            np_list = np.ndarray((0))
            for key, tensor in current.items():
                rshape = (tensor / initial[key]).reshape((np.prod(tensor.shape)))
                new_items = rshape.cpu().numpy()
                np_list = np.concat((np_list, new_items))
            return np_list
        
        def get_clean_median_ratio(ratios: np.ndarray):
            ratios = ratios[(ratios != 0) & ~(np.isnan(ratios))]
            if len(ratios) == 0: return 1
            ratios[ratios < 1] = 1 / ratios[ratios < 1]            
            return np.median(ratios)
        
        ratios = get_ratios(trainer.net_container.hyper.state_dict(), initial_hyper_weights)
        logger.info(f"Hypernet parameters changed on average by a factor of {get_clean_median_ratio(ratios)}")

        ratios = get_ratios({"a": net_ct.policy_weights.clone().detach()}, {"a": initial_stored_weights})
        logger.info(f"Stored Policy parameters changed on average by a factor of {get_clean_median_ratio(ratios)}")
        
        ratios = get_ratios(trainer.net_container.debiasing_model.state_dict(), initial_debias_weights)
        logger.info(f"Debias parameters changed on average by a factor of {get_clean_median_ratio(ratios)}")

        timer.finish_timing()
        timer.print_updating()

    # load the best model
    trainer.net_container.load_state_dict(torch.load(early_stopping.model_address))
    trainer.net_container.policy.eval()
    
    test_pool = trainer.create_pool_data(test_dataloader, bag_size, test_pool_size, random=only_ensemble)
    test_avg_reward, test_loss, test_ensemble_reward = trainer.expected_reward_loss(test_pool)
    dictionary = {"test/avg_mil_loss": test_loss,
                  f"test/avg_{metric}": test_avg_reward,
                  f"test/ensemble_{metric}": test_ensemble_reward}
    if not no_wandb:
        wandb.log(dictionary)
    logger.info(dictionary)

    timer.save_dataframe(f"results/TIMING_{datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}.csv")
    
    return trainer.net_container


def main_sweep():
    run = wandb.init(
        tags=[
            f"BAG_SIZE_{args.bag_size}",
            f"BASELINE_{args.baseline}",
            f"LABEL_{args.label}",
            f"EMBEDDING_MODEL_{args.embedding_model}",
        ],
    )
    config = wandb.config

    args.learning_rate = config.learning_rate
    args.epochs = config.epochs
    args.hdim = config.hdim
    args.embedding_dim = config.embedding_dim
    args.fourier_scale = config.fourier_scale
    args.hyper_ratio = config.hyper_ratio
    args.early_stopping_patience = config.early_stopping_patience
    args.warmup_epochs = config.get("warmup_epochs", 0)
    args.epsilon = config.get("epsilon", 0)
    args.no_wandb = False
    
    args.batch_size = config.batch_size
    args.device = DEVICE

    global train_dataset, eval_dataset, test_dataset 
    current_train_dataloader, current_eval_dataloader, current_test_dataloader = \
        get_dataloaders(args, train_dataset, eval_dataset, test_dataset, logger)
    logger.info(f"SWEEP DEBUG: Recreated dataloaders with batch_size = {args.batch_size}")


    # Model Optimizer Scheduler EarlyStopping
    net_container: HypernetRLMIL = create_net_container(args, run_dir, HypernetRLMILTrainer.get_model_constructor(), logger) # TODO: Make parameter for trainer
    # from IPython import embed; embed(); exit()
    net_container.to(DEVICE)

    optimizer = optim.AdamW(
        [{"params": net_container.hyper.parameters(),
          "lr": 0.1,},
         {"params": [net_container.policy_weights],
          "lr": args.learning_rate,},
         {"params": net_container.debiasing_model.parameters(),
          "lr": args.learning_rate,}],
        lr=args.learning_rate,
    )
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dataloader))
    # scheduler = optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2])
    early_stopping = EarlyStopping(models_dir=run_dir, save_model_name=f"sweep_checkpoint.pt",
                                   trace_func=logger.info,
                                   patience=args.early_stopping_patience, verbose=True, descending=False)
    
    trainer = HypernetRLMILTrainer(
        net_container = net_container,
        learning_rate = args.learning_rate,
        device = DEVICE,
        task_type = args.task_type,
        min_clip = args.min_clip,
        max_clip = args.max_clip,
        sample_algorithm = args.sample_algorithm
    ) # TODO: Make parameter for trainer

    net_container = train(
        trainer=trainer,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopping=early_stopping,
        train_dataloader=current_train_dataloader,
        eval_dataloader=current_eval_dataloader,
        test_dataloader=current_test_dataloader,
        device=DEVICE,
        bag_size=args.bag_size,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        no_wandb=args.no_wandb,
        train_pool_size=args.train_pool_size,
        eval_pool_size=args.eval_pool_size,
        test_pool_size=args.test_pool_size,
        run_name=run.name,
        task_type=args.task_type,
        only_ensemble=args.only_ensemble,
        rl_model=args.rl_model,
        prefix=args.prefix,
        epsilon=args.epsilon,
        reg_coef=args.reg_coef,
        sample_algorithm=args.sample_algorithm
    )

    run.finish()


def main():
    if not args.no_wandb:
        run = wandb.init(
            tags=[
                f"BAG_SIZE_{args.bag_size}",
                f"BASELINE_{args.baseline}",
                f"LABEL_{args.label}",
                f"ACTOR_LR_{args.actor_learning_rate}",
                f"CRITIC_LR_{args.critic_learning_rate}",
                f"MIL_LR_{args.learning_rate}",
                f"EMBEDDING_MODEL_{args.embedding_model}",
            ],
            config=args,
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=f"RL_{args.model_name}_{args.label}_{args.bag_size}_2sided_ExponentialLR",
        )

    config = Namespace(**load_json(os.path.join(run_dir, "sweep_best_model_config.json")))

    args.critic_learning_rate = config.critic_learning_rate
    args.actor_learning_rate = config.actor_learning_rate
    args.learning_rate = config.learning_rate
    args.epochs = config.epochs
    args.hdim = config.hdim
    args.embedding_dim = config.embedding_dim
    args.fourier_scale = config.fourier_scale
    args.hyper_ratio = config.hyper_ratio
    args.early_stopping_patience = config.early_stopping_patience
    args.warmup_epochs = config.warmup_epochs if config.warmup_epochs is not None else 0
    args.epsilon = config.epsilon if config.epsilon is not None else 0
    args.no_wandb = False
    
    args.batch_size = config.batch_size
    args.device = DEVICE

    global train_dataset, eval_dataset, test_dataset 
    current_train_dataloader, current_eval_dataloader, current_test_dataloader = \
        get_dataloaders(args, train_dataset, eval_dataset, test_dataset, logger)
    logger.info(f"SWEEP DEBUG: Recreated dataloaders with batch_size = {args.batch_size}")

    # # Model Optimizer Scheduler EarlyStopping
    net_container = load_rl_model(run_dir, load_best=False)
    net_container.to(DEVICE)

    optimizer = optim.AdamW(
        [{"params": net_container.hyper.parameters(),
          "lr": 0.1,},
         {"params": [net_container.policy_weights],
          "lr": args.learning_rate,},
         {"params": net_container.debiasing_model.parameters(),
          "lr": args.learning_rate,}],
        lr=args.learning_rate,
    )
    
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dataloader))
    # scheduler = optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2])
    early_stopping = EarlyStopping(models_dir=run_dir,
                                   save_model_name=f"checkpoint.pt",
                                   trace_func=logger.info, patience=args.early_stopping_patience, verbose=True,
                                   descending=False)
    
    trainer = HypernetRLMILTrainer(
        net_container = net_container,
        learning_rate = args.learning_rate,
        device = DEVICE,
        task_type = args.task_type,
        min_clip = args.min_clip,
        max_clip = args.max_clip,
        sample_algorithm = args.sample_algorithm
    ) 

    net_container = train(
        trainer=trainer,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopping=early_stopping,
        train_dataloader=current_train_dataloader,
        eval_dataloader=current_eval_dataloader,
        test_dataloader=current_test_dataloader,
        device=DEVICE,
        bag_size=args.bag_size,
        epochs=args.epochs,
        no_wandb=args.no_wandb,
        train_pool_size=args.train_pool_size,
        eval_pool_size=args.eval_pool_size,
        test_pool_size=args.test_pool_size,
        task_type=args.task_type,
        only_ensemble=args.only_ensemble,
        rl_model=args.rl_model,
        epsilon=args.epsilon,
        run_name="no_sweep", # uncomment it to force to write the json result
        warmup_epochs=args.warmup_epochs,
        prefix=args.prefix,
        reg_coef=args.reg_coef,
        sample_algorithm=args.sample_algorithm
    )
    torch.save(net_container.state_dict(),
                os.path.join(early_stopping.models_dir, f"model.pt",))

    if not args.no_wandb:
        run.finish()


if __name__ == "__main__":
    BEST_REWARD = float("-inf")
    args = parse_args()
    # Model name and directory
    run_dir = get_model_save_directory(data_embedded_column_name=args.data_embedded_column_name,
                                       embedding_model_name=args.embedding_model,
                                       target_column_name=args.label, 
                                       bag_size=args.bag_size,
                                       baseline=args.baseline,
                                       random_seed=args.random_seed,
                                       dev=args.dev, 
                                       task_type=args.task_type, 
                                       prefix=args.prefix,
                                       multiple_runs=args.multiple_runs)
    logger = get_logger(run_dir)
    logger.info(f"{args=}")

    DEVICE = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"DEVICE={DEVICE}")

    set_seed(args.random_seed)

    model_name = args.baseline
    args.model_name = model_name

    # read data
    train_dataset, eval_dataset, test_dataset, number_of_classes = prepare_data(args, logger)
    
    if args.task_type == 'regression':
        args.min_clip, args.max_clip = float(train_dataset.Y.min()), float(train_dataset.Y.max())
    else:
        args.min_clip, args.max_clip = None, None
        
    if (args.balance_dataset) & (args.task_type == "classification"):
        logger.info(f"Using weighted random sampler to balance the dataset")
        sample_weights = get_balanced_weights(train_dataset.Y.tolist())
        w_sampler = WeightedRandomSampler(sample_weights, len(train_dataset.Y.tolist()), replacement=True)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, sampler=w_sampler)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    args.number_of_classes = number_of_classes
    args.input_dim = train_dataset.__getitem__(0)[0].shape[1]
    args.state_dim = 22

    logger.info(f"{number_of_classes=}")
    # log train_dataset shape
    logger.info(f"{train_dataset.__len__()=}")
    logger.info(f"{train_dataset.__getitem__(0)[0].shape=}")
    logger.info(f"{train_dataset.__getitem__(0)[1].shape=}")
    logger.info(f"{train_dataset.__getitem__(0)[1]=}")

    if args.run_sweep:
        args.sweep_config["name"] = f"{args.prefix}_{args.label}_rl_{args.baseline}".replace("_", "-")
        # sweep_config = args.pop("sweep_config")
        # sweep_config.update(vars(args))
        sweep_id = wandb.sweep(args.sweep_config, entity=args.wandb_entity, project=args.wandb_project)
        wandb.agent(sweep_id, main_sweep)
    else:
        args.run_name = f"{args.prefix}_{args.label}_rl_{args.baseline}_no_sweep"
        main()
