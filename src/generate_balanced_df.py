import pandas as pd
import numpy as np
import imblearn
from collections import Counter
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

train_df = pd.read_csv('/Users/favea/Downloads/swiss-data/train_df.csv', index_col=0).drop('Unnamed: 0',axis=1)

len_ = train_df.shape[0]
ratio = 0.9

total_len = int(ratio*len_)
train_df = train_df.iloc[:total_len]

print("INPUT SHAPE: ", train_df.shape[0])


X,y = train_df.drop('delayed',axis=1), train_df.loc[:,'delayed']


print(Counter(y))
# define pipeline
over = SMOTENC(categorical_features=[0, 1, 2, 4,9,], random_state=12345, sampling_strategy=0.4, n_jobs= -1)
under = RandomUnderSampler(sampling_strategy=0.5, random_state=12345,)
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)
# transform the dataset
print("Balancing")
X, y = pipeline.fit_resample(X, y)
print(Counter(y))


X['delayed'] = y

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
    dataset_artifact.add_file(f"/Users/favea/Downloads/swiss-data/train_df_balanced_{ratio}.csv")

    # Add the
    run.log_artifact(dataset_artifact)