from argparse import Namespace
import os

import numpy as np
import torch
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.configs import parse_args
from src.logger import get_logger
from src.models.full import DebiasWarmup
from src.trainers.util import create_task_model, get_dataloaders, get_model, prepare_data
from src.utils import (
    get_model_save_directory,
    get_balanced_weights,
    EarlyStopping,
    load_json, save_json,
    set_seed
)

def compute_loss(warmup: DebiasWarmup, dataloader, device):
    warmup.debiasing_model.eval()
    with torch.no_grad():
        bias_losses = []
        for batch_x, batch_y, _, _ in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            true_indices = (torch.max(batch_x[:, (2, 4, 5, 7), :], dim=-1).values - 1).to(dtype=torch.int64)
            _, bias_loss = warmup.predict(batch_x, batch_y, true_indices)
            bias_losses.append(bias_loss)
    return np.mean(bias_losses)

def train(
        warmup: DebiasWarmup,
        optimizer,
        scheduler,
        early_stopping,
        train_dataloader,
        eval_dataloader,
        test_dataloader,
        device,
        epochs,
        run_name=None
):
    for epoch in range(epochs):
        warmup.debiasing_model.train()

        bias_losses = []
        for batch in train_dataloader:
            x, y = batch[0], batch[1]
            x = x.to(device)
            y = y.to(device)

            true_indices = (torch.max(x[:, (2, 4, 5, 7), :], dim=-1).values - 1).to(dtype=torch.int64)
            bias_loss = warmup.predict_train(optimizer, x, y, true_indices)
            bias_losses.append(bias_loss)

        train_loss = np.mean(bias_losses)

        eval_loss = compute_loss(warmup, eval_dataloader, device)
        scheduler.step()

        if not args.no_wandb or args.run_sweep:
            wandb.log(
                {
                    "train/bias_loss": train_loss,
                    "eval/bias_loss": eval_loss,
                }
            )
        early_stopping(eval_loss, warmup)
        if early_stopping.counter == 0:
            logger.info(f"Epoch: {epoch+1}/{epochs} train loss: {train_loss:.6f}")
            logger.info(f"Epoch: {epoch+1}/{epochs} eval loss: {eval_loss:.6f}")

        global BEST_VAL_LOSS
        if eval_loss < BEST_VAL_LOSS:
            logger.info(
                f"Found the best model in all of sweep runs in sweep run {run_name} at epoch {epoch}. Validation loss "
                f"decreased ({BEST_VAL_LOSS:.6f} --> {eval_loss:.6f})."
            )
            logger.info(
                f"Saving the model in run {run_name}"
            )
            BEST_VAL_LOSS = eval_loss
            torch.save(
                warmup.state_dict(),
                os.path.join(
                    run_dir,
                    "best_debiaser.pt",
                ),
            )

            test_loss = compute_loss(warmup, test_dataloader, device)
            
            dictionary = {
                "model": args.baseline,
                "embedding_model": args.embedding_model,
                "bag_size": args.bag_size,
                "label": args.label,
                "seed": args.random_seed,
                "test/bias_loss": test_loss
            }
            save_json(os.path.join(run_dir, "debias_results.json"), dictionary)

        if early_stopping.early_stop:
            logger.info(
                f"Early stopping at epoch {epoch} out of {epochs}"
            )
            break

    logger.info(f"Loading the best model from early stopping checkpoint")
    warmup.load_state_dict(torch.load(early_stopping.model_address))

    test_loss = compute_loss(warmup, test_dataloader, device)
    if not args.no_wandb or args.run_sweep:
        wandb.log(
            {
                "test/loss": test_loss
            }
        )
    logger.info(f"Test loss: {test_loss:.6f}")
    return warmup

def main():
    if not args.no_wandb:
        run = wandb.init(
            tags=[
                f"BAG_SIZE_{args.bag_size}",
                f"BASELINE_{args.baseline}",
                f"LABEL_{args.label}",
                f"MIL_LR_{args.learning_rate}",
                f"EMBEDDING_MODEL_{args.embedding_model}",
            ],
            config=args,
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=f"debias_warmup_{args.model_name}_2sided_ExponentialLR",
        )

    config = Namespace(**load_json(os.path.join(run_dir, "best_model_config.json")))

    args.learning_rate = config.learning_rate
    args.epochs = config.epochs
    args.hdim = config.hdim
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
    warmup = DebiasWarmup(
        task_model=create_task_model(args, run_dir, logger, is_rlmil_dir=False),
        hidden_dim=config.hidden_dim,
        device=args.device
    )

    optimizer = optim.AdamW(
        warmup.debiasing_model.parameters(),
        lr=args.learning_rate,
    )
    
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dataloader))
    # scheduler = optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2])
    early_stopping = EarlyStopping(models_dir=run_dir,
                                   save_model_name=f"debiasing_checkpoint.pt",
                                   trace_func=logger.info, patience=args.early_stopping_patience, verbose=True,
                                   descending=False)

    warmup = train(
        warmup=warmup,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopping=early_stopping,
        train_dataloader=current_train_dataloader,
        eval_dataloader=current_eval_dataloader,
        test_dataloader=current_test_dataloader,
        device=args.device,
        epochs=args.epochs,
        run_name=args.run_name
    )
    torch.save(warmup.state_dict(),
                os.path.join(early_stopping.models_dir, f"debiaser.pt",))

    if not args.no_wandb:
        run.finish()


if __name__ == "__main__":
    BEST_VAL_LOSS = float("inf")
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
                                       prefix=None,
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

    args.run_name = f"debias_warmup_{args.baseline}"
    main()
