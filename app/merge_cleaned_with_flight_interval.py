
import pandas as pd
from multiprocessing import cpu_count, pool
import wandb

cleaned_df = pd.read_csv('/Users/favea/Downloads/swiss-data/10122022_cleaned.csv')
cleaned_df.scheduled_time_departure = pd.to_datetime(cleaned_df.scheduled_time_departure)


flights = pd.read_csv('/Users/favea/Downloads/swiss-data/num_flights_within_hour.csv', index_col=0)

cleaned_df['flights_within_hour'] = flights.values
cleaned_df.to_csv('/Users/favea/Downloads/swiss-data/10122022b_cleaned.csv')
print(cleaned_df.head().T)

with wandb.init(project='swiss-delay-prediction', entity=None, job_type="processed-dataset") as run:
    table_merged = wandb.Table(dataframe=cleaned_df)


    # Create an artifact for our dataset
    dataset_artifact = wandb.Artifact(
        'dataset-cleaned', type='dataset',
        description='Table containing the cleaned dataset that can be use for training',
    )
    # Add the table to the artifact & log the artifact
    dataset_artifact.add(table_merged, 'data-table-delay-cleaned')
    dataset_artifact.add_file('/Users/favea/Downloads/swiss-data/10122022b_cleaned.csv')


    # Add the 
    run.log_artifact(dataset_artifact)
    
