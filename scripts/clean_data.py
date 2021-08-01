import io
from pathlib import Path

import boto3
import numpy as np
import pandas as pd

s3 = boto3.client("s3")
obj = s3.get_object(Bucket="ds-rg271", Key="data/labelled/mebank_tweets_1_year_labelled.csv")
df = pd.read_csv(io.BytesIO(obj["Body"].read()), index_col=0)

# Drop any complaint with a label of -1 or 0.5
df = df.loc[df["complaint"].isin([0, 1])]

# Ensure all labels are the same by making them lower case and stripping trailing whitespace
df["topic"] = df["topic"].str.lower().str.strip()

# Ensure complaint is integer
df["complaint"] = df["complaint"].astype(int)

# Fix missed problem - other
df.loc[df["topic"] == "problem - other", "topic"] = "problem/others"
df.to_csv("mebank_tweets_1_year_clean.csv", index=False)

# create randomly sampled train/test/val split
random_seed = 1234
np.random.seed(random_seed)
train, validate, test = np.split(
    df.sample(frac=1, random_state=random_seed), [int(0.6 * len(df)), int(0.8 * len(df))]
)

output_dir = Path("random_sampling")
output_dir.mkdir(exist_ok=True, parents=True)

train.to_csv(output_dir.joinpath("train.csv"), index=False)
validate.to_csv(output_dir.joinpath("val.csv"), index=False)
test.to_csv(output_dir.joinpath("test.csv"), index=False)


def test_df_same(label_df, df):
    df = df.copy().drop(["set_type"], axis=1)
    _df = df.loc[df.index.isin(label_df.index.values)]
    pd.testing.assert_frame_equal(_df.sort_index(), label_df.sort_index())


# stratified sampling
for label, label_df in df.groupby("complaint"):
    train, validate, test = np.split(
        label_df.sample(frac=1, random_state=random_seed),
        [int(0.6 * len(label_df)), int(0.8 * len(label_df))],
    )

    df.loc[df.index.isin(train.index.values), "set_type"] = "train"
    df.loc[df.index.isin(validate.index.values), "set_type"] = "val"
    df.loc[df.index.isin(test.index.values), "set_type"] = "test"

    # sanity checking
    test_df_same(train, df)
    test_df_same(validate, df)
    test_df_same(test, df)


output_dir = Path("stratified_random_sampling")
output_dir.mkdir(exist_ok=True, parents=True)

for set_type, set_type_df in df.groupby("set_type"):
    set_type_df = set_type_df.drop(columns=["set_type"], axis=1)
    set_type_df.to_csv(output_dir.joinpath(f"{set_type}.csv"), index=False)
