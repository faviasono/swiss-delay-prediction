import pandas as pd
import wandb
from sklearn.model_selection import train_test_split
import os

DATAFRAME = '/Users/favea/Downloads/swiss-data/10122022_cleaned.csv'

cleaned_df = pd.read_csv(DATAFRAME, index_col=0)

train_full, test = train_test_split(cleaned_df.values, test_size=.10,  shuffle = True, random_state=123456,stratify= cleaned_df.values[:,-1])
train, dev = train_test_split(train_full, test_size=.20,  shuffle = True, random_state=123456,stratify= train_full[:,-1])


train_df = pd.DataFrame(train, columns=cleaned_df.columns)
dev_df = pd.DataFrame(dev, columns=cleaned_df.columns)
test_df = pd.DataFrame(test, columns=cleaned_df.columns)


out_path = '/Users/favea/Downloads/swiss-data'

train_df.to_csv(os.path.join(out_path,'train_df.csv'))
dev_df.to_csv(os.path.join(out_path,'dev_df.csv'))
test_df.to_csv(os.path.join(out_path,'test_df.csv'))

with wandb.init(project='swiss-delay-prediction', entity=None, job_type="train-dev-test-split") as run:
    
    artifact = wandb.Artifact('stratified_split', type='dataset')
    artifact.add_file('/Users/favea/Downloads/swiss-data/train_df.csv')
    artifact.add_file('/Users/favea/Downloads/swiss-data/dev_df.csv')
    artifact.add_file('/Users/favea/Downloads/swiss-data/test_df.csv')

    run.log_artifact(artifact)

