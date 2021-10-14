from chaiiConfig import *

from datasets import Dataset
from transformers import AutoTokenizer,AutoModelForQuestionAnswering,DataCollatorWithPadding
import collections

from dataprocessing import data_preproc



## function to map index to row id
def get_id_mapper(df):
    id2idx = {eid:idx for idx,eid in enumerate(df['id'].values)}
    idx2id = {idx:eid for idx,eid in enumerate(df['id'].values)}
    return id2idx,idx2id

## function to generate answer end using start index and answer
def add_end_index(row):
    strt = row['answer_start']
    end = strt+len(row['answer_text'])
    ans = row['context'][strt:end]
    assert(ans==row['answer_text'])
    return end


## load tokenizer and model given the model name or path
def load_model(pretrain_path,device):
    tokenizer = AutoTokenizer.from_pretrained(pretrain_path)
    model = AutoModelForQuestionAnswering.from_pretrained(pretrain_path)
    model = model.to(device)
    ## data collator to prepare batch data
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return tokenizer, model, data_collator


##tokenize train and valid dataset using given tokenizer
def tokenize_train_valid_dataset(train,valid,tokenizer,eid2idx):
    ## creating token using dataprep function
    tokenized_train_ds = train.map(lambda data:data_preproc(data,tokenizer,eid2idx,True),batched=True,remove_columns=train.column_names)
    tokenized_train_ds.set_format("torch")
    tokenized_valid_ds = valid.map(lambda data:data_preproc(data,tokenizer,eid2idx,True),batched=True,remove_columns=valid.column_names)
    tokenized_valid_ds.set_format("torch")
    return tokenized_train_ds, tokenized_valid_ds

def tokenize_test_dataset(test,tokenizer,eid2idx):
    tokenized_test_ds = test.map(lambda data:data_preproc(data,tokenizer,eid2idx,False),batched=True,remove_columns=test.column_names)
    tokenized_test_ds.set_format("torch")
    return tokenized_test_ds

def get_context_bin(context):
    
    context_len = len(context.split())
    if context_len <= 1000:
        return 1
    elif context_len <= 4000:
        return 2
    else:
        return 3
    