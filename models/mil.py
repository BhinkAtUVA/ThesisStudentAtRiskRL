from abc import ABC, abstractmethod
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, f1_score, r2_score, roc_auc_score

from atoms import BaseNetwork

# Base network defining the classification flow
class BaseMLP(nn.Module, ABC):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            dropout_p: float = 0.5,
            autoencoder_layer_sizes=None,
    ):
        super(BaseMLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_p = dropout_p  # register the droupout probability as a buffer

        self.autoencoder_layer_sizes = autoencoder_layer_sizes
        self.base_network = BaseNetwork(self.autoencoder_layer_sizes)

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.base_network(x)
        x = self.aggregate(x)  # Aggregate the data
        x = self.mlp(x)
        return x  # Apply the MLP

    def get_aggregated_data(self, x: torch.Tensor) -> torch.Tensor:
        x = self.base_network(x)
        x = self.aggregate(x)
        return x
    
    @abstractmethod
    def aggregate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Abstract method for data aggregation. This method should be implemented by any class that inherits from BaseMLP.

        :param x: input data
        :return: aggregated data
        """
        pass


class MeanMLP(BaseMLP):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            output_dim,
            dropout_p: float = 0.5,
            autoencoder_layer_sizes=None,
    ):
        super(MeanMLP, self).__init__(
            input_dim,
            hidden_dim,
            output_dim,
            dropout_p,
            autoencoder_layer_sizes=autoencoder_layer_sizes,
        )

    def aggregate(self, x):
        return torch.mean(x, dim=1)  # Compute the mean along the bag_size dimension


class MaxMLP(BaseMLP):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            output_dim,
            dropout_p: float = 0.5,
            autoencoder_layer_sizes=None,
    ):
        super(MaxMLP, self).__init__(
            input_dim,
            hidden_dim,
            output_dim,
            dropout_p,
            autoencoder_layer_sizes=autoencoder_layer_sizes,
        )

    def aggregate(self, x):
        return torch.max(
            x, dim=1
        ).values  # Compute the max along the bag_size dimension


class AttentionMLP(BaseMLP):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            output_dim,
            dropout_p: float = 0.5,
            is_linear_attention: bool = True,
            attention_size: int = 64,
            attention_dropout_p: float = 0.5,
            autoencoder_layer_sizes=None,
    ):
        super(AttentionMLP, self).__init__(
            input_dim,
            hidden_dim,
            output_dim,
            dropout_p,
            autoencoder_layer_sizes=autoencoder_layer_sizes,
        )
        self.attention = None
        self.is_linear_attention = is_linear_attention
        self.attention_size = attention_size
        self.attention_dropout_p = attention_dropout_p

        self.init_attention()
        self.initialize_weights()

    def init_attention(self):
        if self.is_linear_attention:
            self.attention = nn.Linear(self.input_dim, 1)
        else:
            self.attention = torch.nn.Sequential(
                torch.nn.Linear(self.input_dim, self.attention_size),
                torch.nn.Dropout(p=self.attention_dropout_p),
                torch.nn.Tanh(),
                torch.nn.Linear(self.attention_size, 1),
            )

    def aggregate(self, x):
        attention = self.attention(x)
        attention = F.softmax(attention, dim=1)
        return torch.sum(x * attention, dim=1)


class ApproxRepSet(nn.Module):
    def __init__(
            self,
            input_dim,
            n_hidden_sets,
            n_elements,
            n_classes,
            autoencoder_layer_sizes=None,
    ):
        super(ApproxRepSet, self).__init__()
        self.n_hidden_sets = n_hidden_sets
        self.n_elements = n_elements

        self.autoencoder_layer_sizes = autoencoder_layer_sizes
        self.base_network = BaseNetwork(self.autoencoder_layer_sizes)

        self.Wc = nn.Parameter(torch.FloatTensor(input_dim, n_hidden_sets * n_elements))

        self.fc1 = nn.Linear(n_hidden_sets, 32)
        self.fc2 = nn.Linear(32, n_classes)
        self.relu = nn.ReLU()

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.Wc.data)
        nn.init.xavier_uniform_(self.fc1.weight.data)
        nn.init.xavier_uniform_(self.fc2.weight.data)

    def forward(self, x):  # x: (batch_size, bag_size, d)
        t = self.base_network(x)  # t: (batch_size, bag_size, d)
        t = self.relu(
            torch.matmul(t, self.Wc)
        )  # t: (batch_size, bag_size, n_hidden_sets * n_elements)
        t = t.view(
            t.size()[0], t.size()[1], self.n_elements, self.n_hidden_sets
        )  # t: (batch_size, bag_size, n_elements, n_hidden_sets)
        t, _ = torch.max(t, dim=2)  # t: (batch_size, bag_size, n_hidden_sets)
        t = torch.sum(t, dim=1)  # t: (batch_size, n_hidden_sets)
        t = self.relu(self.fc1(t))  # t: (batch_size, 32)
        out = self.fc2(t)  # t: (batch_size, n_classes)
        return out


class StratifiedRandomBaseline:
    """
    class_counts: dict of class counts having the labels as keys and the counts as values.
    It is compatible with the value_counts() method of pandas.
    """

    def __init__(self, class_counts):
        self.class_labels = list(class_counts.keys())
        counts = list(class_counts.values())
        total_count = sum(counts)
        self.probs = [count / total_count for count in counts]

    def __call__(self, size):
        choices = np.random.choice(self.class_labels, size=size, p=self.probs)
        return choices


class MajorityBaseline:
    """
    class_counts: dict of class counts having the labels as keys and the counts as values.
    It is compatible with the value_counts() method of pandas.
    """

    def __init__(self, class_counts):
        self.class_labels = list(class_counts.keys())
        counts = list(class_counts.values())
        self.majority_class = self.class_labels[np.argmax(counts)]

    def __call__(self, size):
        choices = np.full(size, self.majority_class)
        return choices

def random_model(train_dataframe, test_dataframe, args, logger):
    # Get the class counts
    class_counts = train_dataframe["labels"].value_counts().to_dict()
    # Initialize the random baseline
    random_baseline = StratifiedRandomBaseline(class_counts)
    # Get the predictions
    predictions = random_baseline(size=len(test_dataframe["labels"].tolist()))
    # Get the ground truth
    ground_truth = test_dataframe["labels"].values
    # Get the precision, recall, f1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true=ground_truth, y_pred=predictions, average="macro"
    )
    # Get the accuracy
    accuracy = (predictions == ground_truth).sum() / len(ground_truth)
    if not args.no_wandb:
        # Log the metrics
        wandb.init(
            tags=[
                f"BAG_SIZE_{args.bag_size}",
                f"BASELINE_{args.baseline}",
                f"LABEL_{args.label}",
                f"EMBEDDING_MODEL_{args.embedding_model}",
                f"DATA_EMBEDDED_COLUMN_NAME_{args.data_embedded_column_name}",
                f"RANDOM_SEED_{args.random_seed}"
                f"EMBEDDING_MODEL_{args.embedding_model}",
            ],
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=f"{args.baseline}_{args.label}",
        )
        wandb.log(
            {
                "test/accuracy": accuracy,
                "test/precision": precision,
                "test/recall": recall,
                "test/f1": f1,
            }
        )
    logger.info(f"test/accuracy: {accuracy}")
    logger.info(f"test/precision: {precision}")
    logger.info(f"test/recall: {recall}")
    logger.info(f"test/f1: {f1}")

    # Confusion matrix
    cm = confusion_matrix(y_true=ground_truth, y_pred=predictions)
    # Log the confusion matrix
    logger.info(f"Confusion matrix:\n{cm}")

def majority_model(train_dataframe, test_dataframe, args, logger):
    # Get the class counts
    class_counts = train_dataframe["labels"].value_counts().to_dict()
    # Initialize the majority baseline
    majority_baseline = MajorityBaseline(class_counts)
    # Get the predictions
    predictions = majority_baseline(size=len(test_dataframe["labels"]))
    # Get the ground truth
    ground_truth = test_dataframe["labels"].values
    # Get the precision, recall, f1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true=ground_truth, y_pred=predictions, average="macro"
    )
    # Get the accuracy
    accuracy = (predictions == ground_truth).sum() / len(ground_truth)
    if not args.no_wandb:
        # Log the metrics
        wandb.init(
            tags=[
                f"BAG_SIZE_{args.bag_size}",
                f"BASELINE_{args.baseline}",
                f"LABEL_{args.label}",
                f"EMBEDDING_MODEL_{args.embedding_model}",
                f"DATA_EMBEDDED_COLUMN_NAME_{args.data_embedded_column_name}",
                f"RANDOM_SEED_{args.random_seed}"
                f"EMBEDDING_MODEL_{args.embedding_model}",
            ],
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=f"{args.baseline}_{args.label}",
        )
        wandb.log(
            {
                "test/accuracy": accuracy,
                "test/precision": precision,
                "test/recall": recall,
                "test/f1": f1,
            }
        )
    logger.info(f"test/accuracy: {accuracy}")
    logger.info(f"test/precision: {precision}")
    logger.info(f"test/recall: {recall}")
    logger.info(f"test/f1: {f1}")

    # Confusion matrix
    cm = confusion_matrix(y_true=ground_truth, y_pred=predictions)
    # Log the confusion matrix
    logger.info(f"Confusion matrix:\n{cm}")

def create_mil_model(args):
    if args.baseline == "MaxMLP":
        model = MaxMLP(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.number_of_classes,
            dropout_p=args.dropout_p,
            autoencoder_layer_sizes=args.autoencoder_layer_sizes,
        )
    elif args.baseline == "MeanMLP":
        model = MeanMLP(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.number_of_classes,
            dropout_p=args.dropout_p,
            autoencoder_layer_sizes=args.autoencoder_layer_sizes,
        )
    elif args.baseline == "AttentionMLP":
        model = AttentionMLP(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.number_of_classes,
            dropout_p=args.dropout_p,
            autoencoder_layer_sizes=args.autoencoder_layer_sizes,
        )
    elif args.baseline == "repset":
        model = ApproxRepSet(
            input_dim=args.input_dim,
            n_hidden_sets=args.n_hidden_sets,
            n_elements=args.n_elements,
            n_classes=args.number_of_classes,
            autoencoder_layer_sizes=args.autoencoder_layer_sizes,
        )
    else:
        model = None
    return model


def create_mil_model_with_dict(args):
    if args['baseline'] == "MaxMLP":
        model = MaxMLP(
            input_dim=args["input_dim"],
            hidden_dim=args["hidden_dim"],
            output_dim=args["number_of_classes"],
            dropout_p=args["dropout_p"],
            autoencoder_layer_sizes=args["autoencoder_layer_sizes"],
        )
    elif args['baseline'] == "MeanMLP":
        model = MeanMLP(
            input_dim=args["input_dim"],
            hidden_dim=args["hidden_dim"],
            output_dim=args["number_of_classes"],
            dropout_p=args["dropout_p"],
            autoencoder_layer_sizes=args["autoencoder_layer_sizes"],
        )
    elif args['baseline'] == "AttentionMLP":
        model = AttentionMLP(
            input_dim=args["input_dim"],
            hidden_dim=args["hidden_dim"],
            output_dim=args["number_of_classes"],
            dropout_p=args["dropout_p"],
            autoencoder_layer_sizes=args["autoencoder_layer_sizes"],
        )
    elif args['baseline'] == "repset":
        model = ApproxRepSet(
            input_dim=args["input_dim"],
            n_hidden_sets=args["n_hidden_sets"],
            n_elements=args["n_elements"],
            n_classes=args["number_of_classes"],
            autoencoder_layer_sizes=args["autoencoder_layer_sizes"],
        )
    elif args['baseline'] == "SimpleMLP":
        model = MaxMLP(
            input_dim=args["input_dim"],
            hidden_dim=args["hidden_dim"],
            output_dim=args["number_of_classes"],
            dropout_p=args["dropout_p"],
        )
    else:
        model = None
    return model