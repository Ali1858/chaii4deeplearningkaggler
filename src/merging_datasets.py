# -*- coding: utf-8 -*-
"""merging_datasets.ipynb

# Merging different datasets for fine-tuning

1. chaii -- https://www.kaggle.com/c/chaii-hindi-and-tamil-question-answering/data
2. Squad_Translated_to_Tamil -- https://www.kaggle.com/msafi04/squad-translated-to-tamil-for-chaii
3. XQA - https://www.kaggle.com/mdhamani/preprocessed-xqa-tamil
4. xquad - https://github.com/deepmind/xquad -- xquad.hi.json 
       -  https://www.kaggle.com/rhtsingh/mlqa-hindi-processed?select=xquad.csv
5. mlqa -- https://github.com/facebookresearch/MLQA
       -  https://www.kaggle.com/rhtsingh/mlqa-hindi-processed?select=xquad.csv
6. mmqa -- https://github.com/deepaknlp/MMQA
"""

# _importing required libraries
import os
import json
import uuid

# _external libraries
import pandas as pd

"""## Target/Interested columns in all datasets

1. context
2. question
3. answer_text
4. answer_start
"""

target_cols = ['context', 'question', 'answer_text', 'answer_start']

def load_csvtodf(filepath):
    
    df = pd.read_csv(filepath)
    df = df[target_cols]
    
    return df

def load_jsontodf(filepath):
    
    data_dict = json.loads(filepath)
    df = pd.Dataframe.from_dict(data_dict)
    
    return df

"""## 1. Chaii dataset"""

chaii_df = load_csvtodf(os.getcwd() + '/../datasets/chaii/train.csv')
chaii_df.sample(1)

"""## 2. Squad_Translated_to_Tamil (stt)"""

stt_df = load_csvtodf(os.getcwd() + '/../datasets/Squad_Translated_to_Tamil/squad_translated_tamil.csv')
stt_df.sample(1)

"""## 3. XQA processed dataset"""

xqa_dataset_paths = ['/../datasets/xqa/XQA_tamil_dev_query.csv', '/../datasets/xqa/XQA_tamil_dev.csv', '/../datasets/xqa/XQA_tamil_test_query.csv', '/../datasets/xqa/XQA_tamil_test.csv']
xqa_df_list = []

for dataset_path in xqa_dataset_paths:
    xqa_df_list.append(load_csvtodf(os.getcwd()+dataset_path))
    
xqa_df = pd.concat(xqa_df_list, ignore_index=True)
xqa_df.sample(1)

"""## 4. Xquad dataset"""

xquad_df = load_csvtodf(os.getcwd() + '/../datasets/tamil_xquad/xquad.csv')
xquad_df.sample(1)

"""## 5. Mlqa dataset"""

mlqa_df = load_csvtodf(os.getcwd() + '/../datasets/mlqa/mlqa_hindi.csv')
mlqa_df.sample(1)

"""## Merging datasets"""

final_df = pd.concat([chaii_df, stt_df, xquad_df, mlqa_df], ignore_index=True)
final_df = final_df.astype({"answer_start": 'int64'})
final_df.sample(1)

final_df['id'] = final_df.apply(lambda x:uuid.uuid1(), axis=1)
final_df.sample(3)

def get_match_label(context, answer_text, answer_start):
    
    ans_len = len(answer_text)
    if answer_text == context[answer_start:answer_start+ans_len]:
        return 1
    else:
        return 0

final_df['label_match'] = final_df.apply(lambda x:get_match_label(x['context'], x['answer_text'], x['answer_start']), axis=1)
final_df.sample(2)

final_df.label_match.value_counts()

final_df = final_df[final_df['label_match'] == 1]

final_target_cols = ['id', 'context', 'question', 'answer_text', 'answer_start']
final_df = final_df[final_target_cols]
final_df.sample(2)

final_df.to_csv (os.getcwd() + '/../datasets/output/chaii_train_data.csv', index = None, header=True)