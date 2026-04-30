import os
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

CLEAN_DATA_PATH     = "data/clean/"
RAW_OUTPUT_PATH     = "data/oulad/oulad_aggregated_raw.pkl"
ENCODED_OUTPUT_PATH = "data/oulad/oulad_aggregated.pkl"
os.makedirs(os.path.dirname(RAW_OUTPUT_PATH), exist_ok=True)

# Load data
courses_df            = pd.read_csv(os.path.join(CLEAN_DATA_PATH, "courses.csv"))
assessments_df        = pd.read_csv(os.path.join(CLEAN_DATA_PATH, "assessments.csv"))
studentInfo_df        = pd.read_csv(os.path.join(CLEAN_DATA_PATH, "studentInfo.csv"))
studentRegistration_df= pd.read_csv(os.path.join(CLEAN_DATA_PATH, "studentRegistration.csv"))
studentAssessment_df  = pd.read_csv(os.path.join(CLEAN_DATA_PATH, "studentAssessment.csv"))
studentVle_agg_df     = pd.read_csv(os.path.join(CLEAN_DATA_PATH, "studentVle_aggregated.csv"))

# Merge data
studentInfo_merged = (
    studentInfo_df
      .merge(studentRegistration_df, on=["id_student","code_module","code_presentation"], how="left")
      .merge(courses_df, on=["code_module","code_presentation"], how="left")
)

stuAssess = studentAssessment_df.merge(assessments_df, on="id_assessment", how="left")
stuAssess.set_index(["code_module","code_presentation","id_student"], inplace=True)
assess_groups = {k:g for k,g in stuAssess.groupby(level=[0,1,2])}

studentVle_agg_df.set_index(["code_module","code_presentation","id_student"], inplace=True)
vle_groups = {k:g for k,g in studentVle_agg_df.groupby(level=[0,1,2])}

main_groups = studentInfo_merged.groupby(["code_module","code_presentation","id_student"])
label_map = {"Pass":1, "At-Risk":0}

# Step 1: Build raw bags
bags_raw, labels, bag_ids = [], [], []
for (mod,pres,sid), group in tqdm(main_groups, desc="Building raw aggregated bags"):
    label = group.iloc[0]["final_result"]
    if label not in label_map:
        continue
    bag_id = (mod, pres, int(sid))
    instances = []

    # Course & demographics
    for feat in ["code_module","code_presentation","gender","region","imd_band","age_band","disability","highest_education"]:
        val = group.iloc[0].get(feat)
        if pd.notna(val):
            instances.append([(feat, val)])

    # Numeric course-level
    for feat in ["module_presentation_length","num_of_prev_attempts","studied_credits","date_registration"]:
        val = group.iloc[0].get(feat)
        if pd.notna(val):
            instances.append([(feat, val)])

    # Assessments
    for _, row in assess_groups.get((mod, pres, sid), pd.DataFrame()).iterrows():
        feats = []
        for feat in ["assessment_type","score","weight","date_submitted","is_banked"]:
            v = row.get(feat)
            if pd.notna(v):
                feats.append((feat, v))
        if feats:
            instances.append(feats)

    # Aggregated VLE (one instance per activity_type)
    for _, row in vle_groups.get((mod, pres, sid), pd.DataFrame()).iterrows():
        feats = []
        for feat in ["activity_type","sum_click","first_click_day","last_click_day","click_days"]:
            v = row.get(feat)
            if pd.notna(v):
                feats.append((feat, v))
        if feats:
            instances.append(feats)

    if instances:
        bags_raw.append(instances)
        labels.append(label_map[label])
        bag_ids.append(bag_id)

# Save raw aggregated bags
with open(RAW_OUTPUT_PATH, "wb") as f:
    pickle.dump({"raw_bags": bags_raw, "labels": labels, "bag_ids": bag_ids}, f)

# Step 2: Fit encoders and scalers
cat_feats  = ["code_module","code_presentation","gender","region","imd_band","age_band","disability","highest_education","assessment_type","activity_type"]
num_feats  = ["module_presentation_length","num_of_prev_attempts","studied_credits","score","weight","is_banked","sum_click","click_days"]
date_feats = ["date_registration","date_submitted","first_click_day","last_click_day"]

cat_encoders = {f: LabelEncoder() for f in cat_feats}
num_scalers  = {f: MinMaxScaler() for f in num_feats}
date_scaler  = MinMaxScaler()

merged = studentInfo_merged.copy()
for f in cat_feats:
    vals = []
    if f in merged: vals += merged[f].dropna().tolist()
    if f in assessments_df: vals += assessments_df[f].dropna().tolist()
    if f in studentVle_agg_df.reset_index(): vals += studentVle_agg_df.reset_index()[f].dropna().tolist()
    cat_encoders[f].fit(vals)

for f in num_feats:
    vals = []
    for df in [merged, assessments_df, studentAssessment_df.reset_index(), studentVle_agg_df.reset_index()]:
        if f in df: vals += df[f].dropna().tolist()
    num_scalers[f].fit(np.array(vals).reshape(-1,1))

vals = []
for f in date_feats:
    for df in [merged, assessments_df, studentAssessment_df.reset_index(), studentVle_agg_df.reset_index()]:
        if f in df: vals += df[f].dropna().tolist()
    date_scaler.fit(np.array(vals).reshape(-1,1))

# Step 3: Encode aggregated bags
def encode_inst(inst):
    vec = np.zeros(len(cat_feats) + len(num_feats) + len(date_feats))
    for feat, val in inst:
        if pd.isna(val): continue
        if feat in cat_feats:
            idx = cat_feats.index(feat)
            vec[idx] = cat_encoders[feat].transform([val])[0] + 1
        elif feat in num_feats:
            idx = len(cat_feats) + num_feats.index(feat)
            vec[idx] = num_scalers[feat].transform([[val]])[0][0]
        elif feat in date_feats:
            idx = len(cat_feats) + len(num_feats) + date_feats.index(feat)
            vec[idx] = date_scaler.transform([[val]])[0][0]
    return vec

# Load raw and encode
with open(RAW_OUTPUT_PATH, "rb") as f:
    data = pickle.load(f)
raw_bags = data["raw_bags"]
labels   = data["labels"]
bag_ids  = data["bag_ids"]

bags_enc = [np.vstack([encode_inst(inst) for inst in bag]) for bag in raw_bags]

# Save encoded aggregated bags
with open(ENCODED_OUTPUT_PATH, "wb") as f:
    pickle.dump({
        "bags": bags_enc,
        "labels": labels,
        "bag_ids": bag_ids,
        "cat_encoders": cat_encoders,
        "num_scalers": num_scalers,
        "date_scaler": date_scaler
    }, f)