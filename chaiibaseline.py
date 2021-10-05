#!/usr/bin/env python
# coding: utf-8

# # Baseline Model for Kaggel chaii competition.
# ## In this competition we are suppose to predict Answer (index) given the question and context document
# ## Data Language is Hindi and Tamil

# In[5]:


import pandas as pd
import numpy as np

from datasets import Dataset
from transformers import (AutoTokenizer,AutoModelForQuestionAnswering,DataCollatorWithPadding)
from transformers import get_scheduler
from transformers import AdamW

import torch
from torch.utils.data import DataLoader

pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

## path for kaggel notebook
# data_dir = '/kaggle/input/chaii-hindi-and-tamil-question-answering/'
# output_dir = '/kaggle/output/kaggle/working/'


data_dir = '/home/ubuntu/repo/chaii4deeplearningkaggler/data/'
output_dir = data_dir+'outputs/'

train_fn = 'train.csv'
test_fn='test.csv'
sub_fn = 'sample_submission.csv'

BATCH_SIZE= 8
## max length of padding and trunc
MAX_LENGTH = 512
lr = 5e-5
num_epochs = 20


# In[6]:


df = pd.read_csv(data_dir+train_fn)
test_df = pd.read_csv(data_dir+test_fn)


# In[7]:


example_id2idx = { eid:idx for idx,eid in enumerate(df['id'].values)}
example_idx2id = { idx:eid for idx,eid in enumerate(df['id'].values)}


# # checking if give answer start (index) label is correct.

# In[8]:


corp_lens = []
for idx in range(0,df.shape[0]):
    subdf = df.iloc[idx]
    strt = subdf['answer_start']
    end = strt+len(subdf['answer_text'])
    ans = subdf['context'][strt:end]
    
    corp_lens.append(len(subdf['context'].split()))
    if not ans==subdf['answer_text']:
        print(idx,ans,'****',subdf['answer_text'])
        
## This is very rough estimation of corpus length. It doesnt represent actual token lenght from tokenizer        
print(f'top 10 token len of the corpus in training dataset: {sorted(corp_lens,reverse=True)[:10]}')


# In[9]:


## function to generate answer end using start index and answer
def add_end_index(row):
    strt = row['answer_start']
    end = strt+len(row['answer_text'])
    ans = row['context'][strt:end]
    assert(ans==row['answer_text'])
    return end
df['answer_end'] = df.apply(add_end_index,axis=1)
df.head()


# # Preparing dataset

# In[10]:


## random shuffle and splitting data set
## ratio is 85-15
df = df.sample(frac=1)

split_idx = int(0.85*df.shape[0])
train_df = df.iloc[:split_idx].reset_index(drop=True)
valid_df = df.iloc[split_idx:].reset_index(drop=True)

print(train_df.shape,valid_df.shape,df.shape)


# In[11]:


## creating Dataset from pandas. This is hugging face dataset not pytorch
train_dataset = Dataset.from_pandas(train_df)
valid_dataset = Dataset.from_pandas(valid_df)
test_dataset = Dataset.from_pandas(test_df)


# # Initializing the model and tokenizer using pretrain weights. 

# In[12]:


pretrain_path = 'ai4bharat/indic-bert'
# pretrain_path = '/kaggle/input/indicbert/indic-bert-v1'

tokenizer = AutoTokenizer.from_pretrained(pretrain_path)
model = AutoModelForQuestionAnswering.from_pretrained(pretrain_path)
model = model.to(device)
## data collator to prepare batch data
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# In[13]:


all_vocab = tokenizer.get_vocab()
print(f'number of vocab our tokenizer have {len(all_vocab)}')
print(f'printing ... few **vocab** {list(all_vocab.keys())[:10]} and their **index** {list(all_vocab.values())[:10]}')


# In[14]:


## function to create token out of each question and context pair.
## context and question pair are concat together
def dataprep(data):
    data['question'] = [q.strip() for q in data['question']]
    data['context'] = [c.strip() for c in data['context']]
    
    data_tokenizer = tokenizer(data['context'],
                               data['question'],
                               truncation='only_first',
                               max_length=MAX_LENGTH)
    
    
    data_tokenizer["start_positions"] = [s for s in data['answer_start']]
    data_tokenizer["end_positions"] = [e for e in data['answer_end']]
    data_tokenizer['example_id'] = [example_id2idx[i] for i in data['id']]
    return data_tokenizer
    


# In[15]:


## creating token using dataprep function
tokenized_train_ds = train_dataset.map(dataprep,batched=True,remove_columns=train_dataset.column_names)
tokenized_valid_ds = valid_dataset.map(dataprep,batched=True,remove_columns=valid_dataset.column_names)


# In[16]:


tokenized_train_ds.set_format("torch")
tokenized_valid_ds.set_format("torch")


# In[17]:


tokenized_train_ds.data.to_pandas().head()


# In[18]:


## Using pytorch DataLoader function 

train_dataloader = DataLoader(
    tokenized_train_ds, shuffle=True, batch_size=BATCH_SIZE, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_valid_ds, batch_size=BATCH_SIZE, collate_fn=data_collator
)


# In[19]:


## learning rate decay scheduler

optimizer = AdamW(model.parameters(), lr=lr)

num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)


# In[20]:


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
            
## extracting answer from logits
def postprocess(dataset,eid,slogits,elogits,spos,epos,idx2id,n_best_size=20,max_answer_length=30):
    predictions = {}
    for idx in range(len(slogits)):
        start_logits = slogits[idx]
        end_logits = elogits[idx]
        tstart = spos[idx]
        tend = epos[idx]
        ## getting from pandas dataframe using id
        data = dataset[dataset.id == idx2id[eid[idx].item()]]
        ## selecting top 20 logits
        start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
        end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
        valid_ans = []
        corp = data['context'].values[0]
        
        ## iterating over each index pair and discarding invalid ones
        for start_index in start_indexes:
            for end_index in end_indexes:
                
                if end_index < start_index or end_index-start_index+1 > max_answer_length:
                    continue
                valid_ans.append({'score':start_logits[start_index]+end_logits[end_index],
                 'text':corp[start_index:end_index]})
        # selecting best answer        
        if len(valid_ans) >0:
            best_answer = sorted(valid_ans, key=lambda x: x["score"], reverse=True)[0]
        else:
            best_answer = {"text": '', "score": 0.0}
        ## predicting answer with true answer
        
        predictions[data['id'].values[0]] = [best_answer['text'],corp[int(tstart):int(tend)]]        
    return [predictions]

## evaluation 
def model_eval(model,evaldataset_,evaldf,idx2id):
    model.eval()
    loss = 0
    preds = []
    n = len(evaldataset_)
    for batch in evaldataset_:
        batch_ = {k: v.to(device) for k, v in batch.items() if k != 'example_id'}
        with torch.no_grad():
            outputs = model(**batch_)
        loss+= outputs.loss.item()
        
        slogits = outputs.start_logits.cpu().detach().numpy()
        elogits = outputs.end_logits.cpu().detach().numpy()
        spos = batch['start_positions']
        epos = batch['end_positions']

        preds.extend(postprocess(evaldf,batch['example_id'],slogits,elogits,spos,epos,idx2id))
    jac_score = jacard_score_(preds)
    print(f'valid loss is {loss/n} and jaccard score is {jac_score/n}')

## training
def train(model,traindata,evaldata,progress_steps,epochs):
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
        model_eval(model,evaldata[0],evaldata[1],example_idx2id)


# In[21]:


train(model,train_dataloader,[eval_dataloader,valid_df],num_training_steps,num_epochs)


# In[22]:


texample_id2idx = { eid:idx for idx,eid in enumerate(test_df['id'].values)}
texample_idx2id = { idx:eid for idx,eid in enumerate(test_df['id'].values)}

def testdataprep(data):
    
    data['question'] = [q.strip() for q in data['question']]
    data['context'] = [c.strip() for c in data['context']]
    
    data_tokenizer = tokenizer(data['context'],
                               data['question'],
                               truncation='only_first',
                               padding="max_length",
                               max_length=MAX_LENGTH)
    data_tokenizer['example_id'] = [texample_id2idx[i] for i in data['id']]
    return data_tokenizer


tokenized_test_ds = test_dataset.map(testdataprep,batched=True,remove_columns=test_dataset.column_names)
tokenized_test_ds.set_format("torch")
test_dataloader = DataLoader(tokenized_test_ds,collate_fn=data_collator)


# In[23]:


def predict(predset,idx2id):
    model.eval()
    preds = []
    for batch in predset:
        batch_ = {k: v.to(device) for k, v in batch.items() if k != 'example_id'}
        with torch.no_grad():
            outputs = model(**batch_)
        
        slogits = outputs.start_logits.cpu().detach().numpy()
        elogits = outputs.end_logits.cpu().detach().numpy()
        spos=epos =np.zeros(slogits.shape[0])
        preds.extend(postprocess(test_df,batch['example_id'],slogits,elogits,spos,epos,idx2id))
    return preds

predictions =predict(test_dataloader,texample_idx2id)  


# In[24]:


def prep_sub(idx):
    for p in predictions:
        if idx in p:
            return p[idx][0]
test_df['PredictionString'] = test_df.id.apply(prep_sub)


# In[25]:


test_df = test_df.drop(columns=['context','question','language'], axis=1)
test_df.to_csv('submission.csv', index=False)


# In[ ]:




