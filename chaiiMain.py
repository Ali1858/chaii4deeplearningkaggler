from chaiiUtils import *
from train import Trainer,save_model

import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold


log.info('*****loading dataset*****')
train_df = pd.read_csv(DATA_DIR+TRAIN_FN)
test_df = pd.read_csv(DATA_DIR+TEST_FN)

## getting id mapper for train data
train_eid2idx,train_idx2eid = get_id_mapper(train_df)
train_df['answer_end'] = train_df.apply(add_end_index,axis=1)

log.info(f'loading pretain model and tokenizer from path{PRETRAIN_PATH}')
tokenizer, model, data_collator = load_model(PRETRAIN_PATH,device)
cls_token_id = tokenizer.cls_token_id

log.info('Splitting train data')
## random shuffle and splitting data set
## ratio is 90-10
train_df = train_df.sample(frac=1)
context_bin_list = train_df.apply(lambda x:get_context_bin(x['context']), axis=1)
skf = StratifiedKFold(n_splits=10)
for train_index, test_index in skf.split(train_df, context_bin_list):
    train_df = train_df.iloc[train_index].reset_index(drop=True)
    valid_df = train_df.iloc[test_index].reset_index(drop=True)
    break

log.info(f'Creating Hugging face Train and Valid Dataset using pandas df')
## creating Dataset from pandas. This is hugging face dataset not pytorch
train_dataset = Dataset.from_pandas(train_df)
valid_dataset = Dataset.from_pandas(valid_df)
test_dataset = Dataset.from_pandas(test_df)

log.info(f'Creating Token of Train and Valid Dataset')
tokenized_train_ds, tokenized_valid_ds = tokenize_train_valid_dataset(train_dataset,valid_dataset,tokenizer,train_eid2idx)

log.info(f'Creating Pytorch DataLoader')
## Using pytorch DataLoader function 
train_dataloader = DataLoader(tokenized_train_ds, shuffle=True, batch_size=BATCH_SIZE, collate_fn=data_collator)
eval_dataloader = DataLoader(tokenized_valid_ds, batch_size=BATCH_SIZE, collate_fn=data_collator)

log.info('Training.....')
train = Trainer(train_idx2eid,cls_token_id,device,model,len(train_dataloader))
train.fit(train_dataloader,[eval_dataloader,valid_df])
save_model(model,tokenizer)

log.info('Testing....')
## getting id mapper for train data
test_eid2idx,test_idx2eid = get_id_mapper(test_df)
tokenized_test_ds = tokenize_test_dataset(test_dataset,tokenizer,test_eid2idx)
test_dataloader = DataLoader(tokenized_test_ds, batch_size=BATCH_SIZE, collate_fn=data_collator)

pred_df = train.predict(test_dataloader,test_df,test_idx2eid)

print(pred_df.head())