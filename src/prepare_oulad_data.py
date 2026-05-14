import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import save_pickle, set_seed
from logger import get_logger

INPUT_PATH = "data/oulad/oulad_aggregated.pkl"
SEED = 0
DATASET_NAME = "oulad_aggregated"
EMBEDDING_MODEL = "tabular"
DATA_COLUMN = "instances"  
SAVE_DIR = f"data/seed_{SEED}/{DATASET_NAME}/{DATA_COLUMN}/{EMBEDDING_MODEL}"
os.makedirs(SAVE_DIR, exist_ok=True)

# Max instances per bag (1302 for full dataset, 39 for aggregated dataset)
FIXED_BAG_SIZE = 39
# Feature dimension per instance (20 for full dataset, 22 for aggregated dataset)
FEATURE_DIM = 22

logger = get_logger(SAVE_DIR)
logger.info(f"Preparing dataset: {DATASET_NAME}, seed: {SEED}, embedding: {EMBEDDING_MODEL}")
set_seed(SEED)

# Load bags
logger.info(f"Loading bags from {INPUT_PATH}")
with open(INPUT_PATH, "rb") as f:
    data = pickle.load(f)

bags = data["bags"]
labels = data["labels"]
bag_ids = data["bag_ids"]
logger.info(f"Loaded {len(bags)} bags")

# Extra info
bag_sizes = [len(bag) for bag in bags]
logger.info(f"Bag size stats - min: {np.min(bag_sizes)}, max: {np.max(bag_sizes)}, avg: {np.mean(bag_sizes):.2f}")

# Shuffle and split
all_indices = np.arange(len(bags))
train_idx, test_idx = train_test_split(all_indices, test_size=0.2, random_state=SEED)
test_idx, val_idx = train_test_split(test_idx, test_size=0.5, random_state=SEED)
logger.info(f"Train size: {len(train_idx)}, Val size: {len(val_idx)}, Test size: {len(test_idx)}")

def index_to_df(indices):
    return pd.DataFrame({
        "bag": [bags[i] for i in indices],
        "label": [labels[i] for i in indices],
        "bag_id": [bag_ids[i] for i in indices],
    })

train_df = index_to_df(train_idx)
val_df = index_to_df(val_idx)
test_df = index_to_df(test_idx)

# Adding padding and masks
def pad_bag_and_create_mask(bag, bag_size, instance_dim):
    padded_bag = np.zeros((bag_size, instance_dim), dtype=np.float32)
    mask = np.zeros((bag_size,), dtype=np.float32)
    n_instances = min(len(bag), bag_size)
    padded_bag[:n_instances] = bag[:n_instances]
    mask[:n_instances] = 1.0
    return padded_bag, mask

for df in [train_df, val_df, test_df]:
    embeddings, masks = [], []
    for bag in df["bag"]:
        emb, mask = pad_bag_and_create_mask(np.array(bag), FIXED_BAG_SIZE, FEATURE_DIM)
        embeddings.append(emb)
        masks.append(mask)
    df["bag_embeddings"] = embeddings
    df["bag_mask"] = masks

# Save files
logger.info("Saving split dataframes to disk")
save_pickle(os.path.join(SAVE_DIR, "train.pickle"), train_df)
save_pickle(os.path.join(SAVE_DIR, "val.pickle"), val_df)
save_pickle(os.path.join(SAVE_DIR, "test.pickle"), test_df)

# Save encoders/scalers for future use
pickle.dump(data["cat_encoders"], open(os.path.join(SAVE_DIR, "cat_encoders.pkl"), "wb"))
pickle.dump(data["num_scalers"], open(os.path.join(SAVE_DIR, "num_scalers.pkl"), "wb"))

logger.info(f"Finished saving splits to {SAVE_DIR}")
