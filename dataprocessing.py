from chaiiConfig import *
import collections

## if start index or end index is invalid or not
## if its index is out of document ignore it
def is_invalid_index(start_index,end_index,offset_map,max_l):
    criteria_1 = (start_index >= len(offset_map)
                  or end_index >= len(offset_map)
                  or offset_map[start_index][0] == -1111
                  or offset_map[end_index][0] == -1111
                 )
    criteria_2 = end_index < start_index or end_index - start_index+1 > max_l
    if criteria_1 or criteria_2:
        return True
    return False
            
    
        
def data_postproc(df,features,outputs,idx2eid,cls_token_id,top=20,max_answer_length=30):
    
    features_per_context = collections.defaultdict(list)
    for idx, eid in enumerate(features['example_id']):
        features_per_context[idx2eid[eid]].append(idx)
    
    predictions = []
    for idx,row in df.iterrows():
        feature_indices = features_per_context[idx]
        best_answer = {"text":"","score":0.0}
        valid_answers = []
        context = row['context']
        for fidx in feature_indices:
            slogits = outputs['slogits'][fidx]
            elogits = outputs['elogits'][fidx]
            offset_map = features['offset_mapping'][fidx]
            cls_index = features["input_ids"][fidx].index(cls_token_id)
            
            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(slogits)[-1 : -top - 1 : -1].tolist()
            end_indexes = np.argsort(elogits)[-1 : -top - 1 : -1].tolist()
            
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if is_invalid_index(start_index,end_index,offset_map,max_answer_length):
                        continue
                    start_char = offset_map[start_index][0]
                    end_char = offset_map[end_index][1]
                    valid_answers.append({"score": slogits[start_index] + elogits[end_index],
                                         "text": context[start_char:end_char]})
        
        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers,key=lambda x:x["score"],reverse=True)[0]
        predictions.append(best_answer['text'])
    df['PredictionString'] = predictions
    return df


## tokenize the data using given tokenizer. #main function
## the argument are fixed for now. change the function in future to make it dynamic. 
def get_tokenized_data(tokenizer,data):
    data['context'] = [context.strip() for context in data['context']]
    data['question'] = [question.strip() for question in data['question']]
    return tokenizer(data['question'],
                     data['context'],
                     truncation="only_second",
                     max_length=MAX_SEQ_LENGTH,
                     stride=DOC_STRIDE,
                     return_overflowing_tokens=True,
                     return_offsets_mapping=True,
                     padding="max_length")

def dataprep_withlabel(data,tokenized_data,offsets,sq_id,sample_idx,cidx,inp_len):
    answer_start = data['answer_start'][sample_idx]
    
    ## default start and end index
    sidx = cidx
    eidx = cidx
    ## if there is start index
    if len(str(answer_start)) != 0:
        answer_text = data['answer_text'][sample_idx]
        answer_end = data['answer_end'][sample_idx]
        ## search for context start and end in sequence
        token_start_index = 0
        while sq_id[token_start_index] != 1:
            token_start_index += 1
            
        token_end_index = inp_len - 1
        while sq_id[token_end_index] != 1:
                token_end_index -= 1
                
        ## if answer is between offset  start and end:
        ## then get precision index
        if offsets[token_start_index][0] <= answer_start and offsets[token_end_index][1] >= answer_end:
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= answer_start:
                token_start_index += 1
            sidx = token_start_index - 1
            while offsets[token_end_index][1] >= answer_end:
                token_end_index -= 1
            eidx = token_end_index + 1
    
    tokenized_data["start_positions"].append(sidx)
    tokenized_data["end_positions"].append(eidx)
    return tokenized_data
    
    
## function to create token out of each question and context pair.
## context and question pair are concat together
def data_preproc(data,tokenizer,eid2idx,labels=True):
    context_index = 1
    tokenized_data = get_tokenized_data(tokenizer,data)
   
    sample_mapping = tokenized_data.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_data.pop("offset_mapping")
    tokenized_data["offset_mapping"] = offset_mapping
    tokenized_data["example_id"] = []
    if labels:
        tokenized_data["start_positions"] = []
        tokenized_data["end_positions"] = []
        
    for idx, offsets in enumerate(offset_mapping):
        sample_index = sample_mapping[idx]
        tokenized_data["example_id"].append(eid2idx[data["id"][sample_index]])
        
        input_ids = tokenized_data["input_ids"][idx]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_data.sequence_ids(idx)
        
        if labels:
            tokenized_data= dataprep_withlabel(data,tokenized_data,offsets,sequence_ids,sample_index,cls_index,len(input_ids))

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        updated_offset_list = []
        for k, o in enumerate(offsets):
            if sequence_ids[k] == context_index:
                updated_offset_list.append(o)
            else:
                updated_offset_list.append((-1111,-1111))
        tokenized_data["offset_mapping"][idx] = updated_offset_list
    
    return tokenized_data