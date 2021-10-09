## Praveen--- code migth need some imports
## please identify them if and import it. If you are running it. 

ts = 20
te = 25
eids = []
idx4postproc = []

test = [idx for idx,st in enumerate(tokenized_train_ds['start_positions']) if st != 0]

for t in test[ts:te]:
    eids.append(tokenized_train_ds[t]['example_id'].item())

features_per_context = collections.defaultdict(list)
for idx, eid in enumerate(tokenized_train_ds['example_id']):
    features_per_context[train_idx2eid[eid.item()]].append(idx)
    
    
def testcase_preproc(tidxs,df,idx2eid,tokenized):
    for t in tidxs[ts:te]:
        print(f"\n*******************checking for idx{t} and example id{idx2eid[tokenized[t]['example_id'].item()]}***************************")
        e = tokenized[t]['end_positions']+1
        s = tokenized[t]['start_positions']
        ofs = tokenized[t]['offset_mapping'][s][0]
        ofe = tokenized[t]['offset_mapping'][e-1][1]
        
        a = tokenizer.decode(tokenized[t]['input_ids'][s:e])
        ta = df[df.id == idx2eid[tokenized[t]['example_id'].item()]]
        print(f"actual answer {ta['answer_text'].values[0]} preprocess answer {a}")
        print(f"actual char start index {ta['answer_start'].values[0]}, preproc  start {ofs} char end index {ta['answer_end'].values[0]} preproc end {ofe}")
    
testcase_preproc(test,train_df,train_idx2eid,tokenized_train_ds)


def testcase_postproc(df,tidxs,idx2eid,tokenized):
    all_batches = {'example_id':[],'offset_mapping':[],'input_ids':[]}
    all_outputs = {'slogits':[],'elogits':[]}
    
    for t in tidxs[ts:te]:
        features = features_per_context[idx2eid[t]]
        for f in features:
            all_batches['example_id'].append(tokenized[f]['example_id'].flatten().tolist()[0])
            all_batches['offset_mapping'].append(tokenized[f]['offset_mapping'])
            all_batches['input_ids'].append(tokenized[f]['input_ids'].tolist())
            e = tokenized[f]['end_positions']
            s = tokenized[f]['start_positions']
            sz = torch.zeros(MAX_SEQ_LENGTH)
            sz[[s]] = 1
            ez = torch.zeros(MAX_SEQ_LENGTH)
            ez[[e]] = 1
            all_outputs['slogits'].append(sz)
            all_outputs['elogits'].append(ez)
    from dataprocessing import data_postproc
    df = data_postproc(df,all_batches,all_outputs,idx2eid,cls_token_id)

        
df = testcase_postproc(train_df,test,train_idx2eid,tokenized_train_ds)

pred_keys = [train_idx2eid[tid] for tid in test[ts:te]]
df[df.id.isin(pred_keys)]