#!/usr/bin/env python
# coding: utf-8

# # Baseline Model for Kaggle chaii competition.
# ## In this competition we are suppose to predict Answer (index) given the question and context document
# ## Data Language is Hindi and Tamil

# - https://www.kaggle.com/jirkaborovec/chaii-q-a-with-pytorch-lightining
# - https://www.kaggle.com/theamitnikhade/question-answering-starter-roberta

# In[1]:


import pandas as pd
import numpy as np
import collections

from datasets import Dataset
from transformers import AutoTokenizer,AutoModelForQuestionAnswering,DataCollatorWithPadding
from transformers import get_scheduler
from transformers import AdamW

import torch
from torch.utils.data import DataLoader


# In[2]:


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)


# In[3]:


## path for kaggel notebook
data_dir = '/kaggle/input/chaii-hindi-and-tamil-question-answering/'
output_dir = '/kaggle/output/kaggle/working/'

train_fn = 'train.csv'
test_fn='test.csv'
sub_fn = 'sample_submission.csv'

BATCH_SIZE= 8
#MAX_LENGTH = 512
MAX_SEQ_LENGTH = 384
DOC_STRIDE = 128
lr = 5e-5
num_epochs = 20


# In[4]:


train_df = pd.read_csv(data_dir+train_fn)
test_df = pd.read_csv(data_dir+test_fn)


# In[5]:


example_id2idx = { eid:idx for idx,eid in enumerate(train_df['id'].values)}
example_idx2id = { idx:eid for idx,eid in enumerate(train_df['id'].values)}


# # checking if the given answer start (index) label is correct.

# In[6]:


corp_lens = []
for idx in range(0,train_df.shape[0]):
    subdf = train_df.iloc[idx]
    strt = subdf['answer_start']
    end = strt+len(subdf['answer_text'])
    ans = subdf['context'][strt:end]
    
    corp_lens.append(len(subdf['context'].split()))
    if not ans==subdf['answer_text']:
        print(idx,ans,'****',subdf['answer_text'])
        
## This is very rough estimation of corpus length. It doesnt represent actual token lenght from tokenizer        
print(f'top 10 token len of the corpus in training dataset: {sorted(corp_lens,reverse=True)[:10]}')


# In[7]:


## function to generate answer end using start index and answer
def add_end_index(row):
    strt = row['answer_start']
    end = strt+len(row['answer_text'])
    ans = row['context'][strt:end]
    assert(ans==row['answer_text'])
    return end
train_df['answer_end'] = train_df.apply(add_end_index,axis=1)
train_df.head(3)


# # Preparing dataset

# In[8]:


## random shuffle and splitting data set
## ratio is 85-15
train_df = train_df.sample(frac=1)

split_idx = int(0.85*train_df.shape[0])
valid_df = train_df.iloc[split_idx:].reset_index(drop=True)
train_df = train_df.iloc[:split_idx].reset_index(drop=True)

print(train_df.shape,valid_df.shape)


# In[9]:


## creating Dataset from pandas. This is hugging face dataset not pytorch
train_dataset = Dataset.from_pandas(train_df)
valid_dataset = Dataset.from_pandas(valid_df)
test_dataset = Dataset.from_pandas(test_df)


# # Initializing the model and tokenizer using pretrain weights. 

# In[10]:


pretrain_path = 'ai4bharat/indic-bert'
# pretrain_path = '/kaggle/input/indicbert/indic-bert-v1'

tokenizer = AutoTokenizer.from_pretrained(pretrain_path)
model = AutoModelForQuestionAnswering.from_pretrained(pretrain_path)
model = model.to(device)
## data collator to prepare batch data
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# In[11]:


all_vocab = tokenizer.get_vocab()
print(f'number of vocab our tokenizer have {len(all_vocab)}')
print(f'printing ... few **vocab** {list(all_vocab.keys())[:10]} and their **index** {list(all_vocab.values())[:10]}')


# In[12]:


## function to create token out of each question and context pair.
## context and question pair are concat together
def dataprep(data):
    
    data['context'] = [context.strip() for context in data['context']]
    data['question'] = [question.strip() for question in data['question']]
    
    tokenizer_output = tokenizer(data['question'],
                                 data['context'],
                                truncation="only_second",
                                max_length=MAX_SEQ_LENGTH,
                                stride=DOC_STRIDE,
                                return_overflowing_tokens=True,
                                return_offsets_mapping=True,
                                padding="max_length",)
    
    
    sample_mapping = tokenizer_output.pop("overflow_to_sample_mapping")
    offset_mapping = tokenizer_output.pop("offset_mapping")
    
    tokenizer_output["start_positions"] = []
    tokenizer_output["end_positions"] = []
        
    for i, offsets in enumerate(offset_mapping):
        feature = {}
        input_ids = tokenizer_output["input_ids"][i]
        attention_mask = tokenizer_output["attention_mask"][i]
        
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenizer_output.sequence_ids(i)
        
        sample_index = sample_mapping[i]
        answer_text = data['answer_text'][sample_index]
        answer_start = data['answer_start'][sample_index]
        answer_end = data['answer_end'][sample_index]

        if len(str(answer_start)) == None:
            tokenizer_output["start_positions"].append(cls_index)
            tokenizer_output["end_positions"].append(cls_index)
        else:
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

                
            # 01111111110
            # [CLS]ccccccc[SEP]
                
            if not (offsets[token_start_index][0] <= answer_start and offsets[token_end_index][1] >= answer_end):
                tokenizer_output["start_positions"].append(cls_index)
                tokenizer_output["end_positions"].append(cls_index)
            else:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= answer_start:
                    token_start_index += 1
                tokenizer_output["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= answer_end:
                    token_end_index -= 1
                tokenizer_output["end_positions"].append(token_end_index + 1)
        
    return tokenizer_output


# In[13]:


## creating token using dataprep function
tokenized_train_ds = train_dataset.map(dataprep,batched=True,remove_columns=train_dataset.column_names)
tokenized_train_ds.set_format("torch")


# In[14]:


## function to create token out of each question and context pair.
## context and question pair are concat together
def dataprep_validation(data):
    
    data['context'] = [context.strip() for context in data['context']]
    data['question'] = [question.strip() for question in data['question']]
    
    tokenizer_output = tokenizer(data['question'],
                                 data['context'],
                                truncation="only_second",
                                max_length=MAX_SEQ_LENGTH,
                                stride=DOC_STRIDE,
                                return_overflowing_tokens=True,
                                return_offsets_mapping=True,
                                padding="max_length",)
    
    
    sample_mapping = tokenizer_output.pop("overflow_to_sample_mapping")
    tokenizer_output['example_id'] = []
    
    for i in range(len(tokenizer_output["input_ids"])):
        sequence_ids = tokenizer_output.sequence_ids(i)
        sample_index = sample_mapping[i]
        
        #print(sample_index)
        tokenizer_output["example_id"].append(example_id2idx[data["id"][sample_index]])
        
        updated_offset_list = []
        old_offset_list = tokenizer_output["offset_mapping"][i]
        for k, o in enumerate(old_offset_list):
            if sequence_ids[k] == 1:
                updated_offset_list.append(o)
            else:
                updated_offset_list.append((-1,-1))
                
        tokenizer_output["offset_mapping"][i] = updated_offset_list
        
    return tokenizer_output


# In[15]:


tokenized_valid_ds = valid_dataset.map(dataprep_validation,batched=True,remove_columns=valid_dataset.column_names)


# In[16]:


tokenized_valid_ds_updated = tokenized_valid_ds.map(lambda data: data, remove_columns=['example_id', 'offset_mapping'])
tokenized_valid_ds_updated.set_format("torch")


# In[17]:


tokenized_train_ds.data.to_pandas().head(3)


# In[18]:


eval_df = tokenized_valid_ds.data.to_pandas()
eval_df.sample(3)


# In[19]:


## Using pytorch DataLoader function 

train_dataloader = DataLoader(
    tokenized_train_ds, shuffle=True, batch_size=BATCH_SIZE, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_valid_ds_updated, batch_size=BATCH_SIZE, collate_fn=data_collator
)


# In[20]:


## learning rate decay scheduler

optimizer = AdamW(model.parameters(), lr=lr)

num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)


# In[21]:


from tqdm.auto import tqdm

def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def jacard_score_(data_dict):
    score = 0
    for data in data_dict:
        p,tp = list(data.values())[0]
        if p is not '' and tp is not '':
            score += jaccard(p,tp)
    return score

def transform_logits(logits):
    
    new_logits = np.array(logits[0])
    for logit in logits[1:]:
        new_logits = np.vstack((new_logits, logit))
    
    return new_logits

## extracting answer from logits
def postprocess(val_df,eval_df,all_start_logits,all_end_logits,idx2id,n_best_size=20,max_answer_length=30):
    
    all_start_logits = transform_logits(all_start_logits)
    all_end_logits = transform_logits(all_end_logits)
    
    features_per_example = collections.defaultdict(list)
    all_features = []
    for i, feature in enumerate(eval_df):
        features_per_example[idx2id[feature['example_id']]].append(i)
        all_features.append(i)
            
    predictions = collections.OrderedDict()
    
    for idx, data  in enumerate(val_df):
        feature_indices = features_per_example[data["id"]]
        valid_ans = []
        
        context = data["context"]
        for feature_index in feature_indices:
            
            offset_mapping = eval_df[feature_index]["offset_mapping"]
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]

            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:

                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index][0] == -1
                        or offset_mapping[end_index][0] == -1
                    ):
                        continue
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_ans.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char: end_char]
                        }
                    )
                    
        if len(valid_ans) > 0:
            best_answer = sorted(valid_ans, key=lambda x: x["score"], reverse=True)[0]
        else:
            best_answer = {"text": "", "score": 0.0}
        
        predictions[data["id"]] = best_answer["text"]
        
    return predictions

## evaluation 
def model_eval(model,val_df,evaldataset_,eval_df,idx2id):
    model.eval()
    preds = []
    n = len(evaldataset_)
    
    all_start_logits = []
    all_end_logits = []
    
    for batch in evaldataset_:
        batch_ = {k: v.to(device) for k, v in batch.items() if k != 'example_id'}
        with torch.no_grad():
            outputs = model(**batch_)
        
        slogits = outputs.start_logits.cpu().detach().numpy()
        elogits = outputs.end_logits.cpu().detach().numpy()
        
        all_start_logits.append(slogits)
        all_end_logits.append(elogits)

    preds = postprocess(val_df,eval_df,all_start_logits,all_end_logits,idx2id)
    jac_score = jacard_score_(preds)
    print(f'jaccard score is {jac_score/n}')

## training
def train(model,traindata,val_df,eval_dataloader,eval_df,progress_steps,epochs):
    progress_bar = tqdm(range(progress_steps))
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in traindata:
            optimizer.zero_grad()
            
            batch_ = {k: v.to(device) for k, v in batch.items() if k != 'example_id'}
            outputs = model(**batch_)
            loss = outputs.loss
            loss.backward()
            
            optimizer.step()
            lr_scheduler.step()
            
            epoch_loss += loss.item()
            progress_bar.update(1)
            
        print(f'training loss for epoch {epoch} is {epoch_loss/len(traindata)}')
        model_eval(model,val_df,eval_dataloader,eval_df,example_idx2id)


# In[ ]:


train(model,train_dataloader,valid_dataset,eval_dataloader,tokenized_valid_ds,num_training_steps,num_epochs)


# In[ ]:


texample_id2idx = { eid:idx for idx,eid in enumerate(test_df['id'].values)}
texample_idx2id = { idx:eid for idx,eid in enumerate(test_df['id'].values)}


# In[ ]:


def dataprep_test(data):
    
    data['context'] = [context.strip() for context in data['context']]
    data['question'] = [question.strip() for question in data['question']]
    
    tokenizer_output = tokenizer(data['question'],
                                 data['context'],
                                truncation="only_second",
                                max_length=MAX_SEQ_LENGTH,
                                stride=DOC_STRIDE,
                                return_overflowing_tokens=True,
                                return_offsets_mapping=True,
                                padding="max_length",)
    
    
    sample_mapping = tokenizer_output.pop("overflow_to_sample_mapping")
    tokenizer_output['example_id'] = []
    
    for i in range(len(tokenizer_output["input_ids"])):
        sequence_ids = tokenizer_output.sequence_ids(i)
        sample_index = sample_mapping[i]
        
        tokenizer_output["example_id"].append(texample_id2idx[data["id"][sample_index]])
        
        updated_offset_list = []
        old_offset_list = tokenizer_output["offset_mapping"][i]
        for k, o in enumerate(old_offset_list):
            if sequence_ids[k] == 1:
                updated_offset_list.append(o)
            else:
                updated_offset_list.append((-1,-1))
                
        tokenizer_output["offset_mapping"][i] = updated_offset_list
    
    return tokenizer_output


# In[ ]:


tokenized_test_ds = test_dataset.map(dataprep_test,batched=True,remove_columns=test_dataset.column_names)
tokenized_test_updated = tokenized_test_ds.map(lambda data: data, remove_columns=['example_id', 'offset_mapping'])
tokenized_test_updated.set_format("torch")


# In[ ]:


test_eval_df = tokenized_test_ds.data.to_pandas()
test_eval_df.sample(3)


# In[ ]:


test_dataloader = DataLoader(
    tokenized_test_updated, batch_size=BATCH_SIZE, collate_fn=data_collator
)


# In[ ]:


def predict(predset,idx2id, test_dataset, test_eval_df):
    
    model.eval()
    preds = []
    
    all_start_logits = []
    all_end_logits = []
    
    for batch in predset:
        batch_ = {k: v.to(device) for k, v in batch.items() if k != 'example_id'}
        with torch.no_grad():
            outputs = model(**batch_)
        
        slogits = outputs.start_logits.cpu().detach().numpy()
        elogits = outputs.end_logits.cpu().detach().numpy()
        
        all_start_logits.append(slogits)
        all_end_logits.append(elogits)

    preds = postprocess(test_dataset,test_eval_df,all_start_logits,all_end_logits,idx2id)

    return preds


# In[ ]:


predictions = predict(test_dataloader,texample_idx2id, test_dataset, tokenized_test_ds)
predictions


# In[ ]:


for row in predictions.keys():
    print(row)


# In[ ]:


def prep_sub(idx):
    return predictions[idx]

test_df['PredictionString'] = test_df.id.apply(prep_sub)


# In[ ]:


test_df.PredictionString.values


# In[ ]:


test_df = test_df.drop(columns=['context','question','language'], axis=1)
test_df.to_csv('submission.csv', index=False)

