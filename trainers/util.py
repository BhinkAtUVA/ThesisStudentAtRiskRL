import os

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from RLMIL_Datasets import RLMILDataset
from models.full import NetworkContainer
from models.mil import create_mil_model_with_dict
from trainers.base import Trainer
from utils import (
    get_data_directory,
    read_data_split,
    preprocess_dataframe,
    get_df_mean_median_std,
    get_balanced_weights,
    load_json
)

def prepare_data(args, logger):
    logger.info(f"Prepare datasets: Column={args.data_embedded_column_name}")
    data_dir = get_data_directory(args.data_embedded_column_name, args.random_seed)
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory \"{data_dir}\" does not exist.")
    train_dataframe = read_data_split(data_dir, args.embedding_model, "train")
    val_dataframe = read_data_split(data_dir, args.embedding_model, "val")
    test_dataframe = read_data_split(data_dir, args.embedding_model, "test")

    train_dataframe_mean, train_dataframe_median, train_dataframe_std = get_df_mean_median_std(
        train_dataframe, args.label
    )
    if args.instance_labels_column is not None:
        extra_columns = [args.instance_labels_column]
    else:
        extra_columns = []
    train_dataframe, label2id, id2label = preprocess_dataframe(df=train_dataframe, dataframe_set="train", label=args.label,
                                           train_dataframe_mean=train_dataframe_mean,
                                           train_dataframe_median=train_dataframe_median,
                                           train_dataframe_std=train_dataframe_std, task_type=args.task_type,
                                           extra_columns=extra_columns)
    val_dataframe, _, _ = preprocess_dataframe(df=val_dataframe, dataframe_set="val", label=args.label,
                                         train_dataframe_mean=train_dataframe_mean,
                                         train_dataframe_median=train_dataframe_median,
                                         train_dataframe_std=train_dataframe_std, task_type=args.task_type,
                                         extra_columns=extra_columns)
    test_dataframe, _, _ = preprocess_dataframe(df=test_dataframe, dataframe_set="test", label=args.label,
                                          train_dataframe_mean=train_dataframe_mean,
                                          train_dataframe_median=train_dataframe_median,
                                          train_dataframe_std=train_dataframe_std, task_type=args.task_type,
                                          extra_columns=extra_columns)

    # If label2id and id2label were valid dictionaries, add them to args
    if label2id is not None and id2label is not None:
        args.label2id = label2id
        args.id2label = id2label

    train_dataset = RLMILDataset(
        df=train_dataframe,
        bag_masks=None,
        subset=False,
        task_type=args.task_type,
        instance_labels_column=args.instance_labels_column,
    )
    val_dataset = RLMILDataset(
        df=val_dataframe,
        bag_masks=None,
        subset=False,
        task_type=args.task_type,
        instance_labels_column=args.instance_labels_column,
    )
    test_dataset = RLMILDataset(
        df=test_dataframe,
        bag_masks=None,
        subset=False,
        task_type=args.task_type,
        instance_labels_column=args.instance_labels_column,
    )

    number_of_classes = len(train_dataframe["labels"].unique())

    return train_dataset, val_dataset, test_dataset, number_of_classes

def create_task_model(args, mil_best_model_dir, logger):
    if args.rl_task_model == "ensemble":
        for ensemble_dir in os.listdir(os.path.join(mil_best_model_dir, "..")):
            if "only_"+args.rl_task_model in ensemble_dir:
                mil_best_model_dir = os.path.join(mil_best_model_dir, "..", ensemble_dir)
                break
        logger.info(f"Loading ensemble model from {mil_best_model_dir}")
        ensemble_state_dict = torch.load(os.path.join(mil_best_model_dir, "sweep_best_model.pt"), map_location=torch.device("cpu"))
        state_dict = {}
        for k in ensemble_state_dict.keys():
            if k.startswith("task_model."):
                state_dict[k.split("task_model.")[1]] = ensemble_state_dict[k]
    else:
        state_dict = torch.load(os.path.join(mil_best_model_dir, "..", "best_model.pt"))
    task_model = load_mil_model_from_config(os.path.join(mil_best_model_dir, "..", "best_model_config.json"),
                                            state_dict)
    return task_model

def create_net_container(args, mil_best_model_dir, constructor: type[NetworkContainer], logger):
    if args.rl_task_model == "ensemble":
        for ensemble_dir in os.listdir(os.path.join(mil_best_model_dir, "..")):
            if "only_"+args.rl_task_model in ensemble_dir:
                mil_best_model_dir = os.path.join(mil_best_model_dir, "..", ensemble_dir)
                break
        logger.info(f"Loading ensemble model from {mil_best_model_dir}")
        ensemble_state_dict = torch.load(os.path.join(mil_best_model_dir, "sweep_best_model.pt"), map_location=torch.device("cpu"))
        state_dict = {}
        for k in ensemble_state_dict.keys():
            if k.startswith("task_model."):
                state_dict[k.split("task_model.")[1]] = ensemble_state_dict[k]
    else:
        state_dict = torch.load(os.path.join(mil_best_model_dir, "..", "best_model.pt"))
    task_model = load_mil_model_from_config(os.path.join(mil_best_model_dir, "..", "best_model_config.json"),
                                            state_dict)
    net_container: NetworkContainer = constructor(
        task_model=task_model,
        state_dim=args.state_dim,
        hdim=args.hdim,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        device=args.device,
        task_type=args.task_type,
        min_clip=args.min_clip,
        max_clip=args.max_clip,
        sample_algorithm=args.sample_algorithm,
        no_autoencoder=args.no_autoencoder_for_rl
    )
    return net_container

def load_mil_model_from_config(mil_config_file, state_dict):
    mil_config = load_json(mil_config_file)
    task_model = create_mil_model_with_dict(mil_config)
    task_model.load_state_dict(state_dict)
    
    return task_model

def get_dataloaders(args, train_dataset, eval_dataset, test_dataset, logger):
    if (args.balance_dataset) & (args.task_type == "classification"):
        logger.info(f"Using weighted random sampler to balance the dataset for training")
        sample_weights = get_balanced_weights(train_dataset.Y.tolist()) # Ensure get_balanced_weights is imported/defined
        w_sampler = WeightedRandomSampler(sample_weights, len(train_dataset.Y.tolist()), replacement=True)
        current_train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, sampler=w_sampler)
    else:
        current_train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    current_eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    current_test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    return current_train_dataloader, current_eval_dataloader, current_test_dataloader

def predict(net_container: Trainer, dataloader, bag_size=20, pool_size=10):
    pool_data = net_container.create_pool_data(dataloader, bag_size, pool_size)
    preds = net_container.predict_pool(pool_data)
    
    return preds