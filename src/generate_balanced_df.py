import pandas as pd
import numpy as np
import imblearn
from collections import Counter
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

train_df = pd.read_csv("/Users/favea/Downloads/swiss-data/train_df.csv", index_col=0)

columns_to_remove = [
    "Unnamed: 0",
    "wh_fleg_leg_i",
    "id_flight",
    "carrier",
    "season_trip",
    "delay_minutes",
    "scheduled_time_departure",
    "year",
    "previous_is_delayed",
    "flights_within_hour",
]
before_flights = pd.read_csv(
    "/Users/favea/Downloads/swiss-data/previous_delayed_same_day.csv"
).drop("Unnamed: 0", axis=1)
train_df = train_df.merge(before_flights, left_on="id_flight", right_on="wh_fleg_leg_i")
train_df = train_df.drop(columns_to_remove, axis=1)
len_ = train_df.shape[0]
ratio = 1

total_len = int(ratio * len_)
train_df = train_df.iloc[:total_len]

print("INPUT SHAPE: ", train_df.shape[0])


X, y = train_df.drop("delayed", axis=1), train_df.loc[:, "delayed"]


print(X.columns)


print(Counter(y))
# define pipeline
over = SMOTENC(
    categorical_features=[0, 1, 3, 4, 12],
    random_state=12345,
    sampling_strategy=0.4,
    n_jobs=-1,
)
under = RandomUnderSampler(
    sampling_strategy=0.5,
    random_state=12345,
)
steps = [("o", over), ("u", under)]
pipeline = Pipeline(steps=steps)
# transform the dataset
print("Balancing")
X, y = pipeline.fit_resample(X, y)
print(Counter(y))


X["delayed"] = y

X.to_csv(f"/Users/favea/Downloads/swiss-data/train_df_balanced_{ratio}.csv")

import wandb

with wandb.init(
    project="swiss-delay-prediction", entity=None, job_type="processed-dataset"
) as run:
    table_merged = wandb.Table(dataframe=X)

    # Create an artifact for our dataset
    dataset_artifact = wandb.Artifact(
        "train-dataset-v2-balanced",
        type="dataset",
        description="Table containing the cleaned dataset that can be use for training",
    )
    # Add the table to the artifact & log the artifact
    dataset_artifact.add(table_merged, "data-table-delay-balanced")
    dataset_artifact.add_file(
        f"/Users/favea/Downloads/swiss-data/train_df_balanced_{ratio}.csv"
    )

    # Add the
    run.log_artifact(dataset_artifact)
