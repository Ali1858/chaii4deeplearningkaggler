from chaiiConfig import *
from dataprocessing import data_postproc
from transformers import get_scheduler
from transformers import AdamW

from tqdm.auto import tqdm


def single_jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def jacard(df):
    score = 0
    for _,row in df.iterrows():
        y = row['answer_text']
        y_hat = row['PredictionString']
        score += single_jaccard(y,y_hat)
    return score/df.shape[0]


def get_optim(model,n):
    optimizer = AdamW(model.parameters(), lr=LR)
    num_training_steps = NUM_EPOCHS * n
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    return optimizer,num_training_steps,lr_scheduler


def flat_batch(all_batches,batch):
    all_batches['example_id'].extend(batch['example_id'].flatten().tolist())
    all_batches['offset_mapping'].extend(batch['offset_mapping'])
    all_batches['input_ids'].extend(batch['input_ids'])
    
def flat_outputs(all_outputs,outputs):
    all_outputs['slogits'].append(outputs.start_logits.cpu().detach().numpy())
    all_outputs['elogits'].append(outputs.end_logits.cpu().detach().numpy())

## training
class Trainer:
    def __init__(self,idx2eid,cls_token_id,device,model,N):
        
        ## learning rate decay scheduler
        self.optim,self.steps,self.scheduler = get_optim(model,N)
        self.epochs = NUM_EPOCHS
        self.idx2eid = idx2eid
        self.cls_token_id = cls_token_id
        self.device = device
        self.model = model
        self.exclude = ['example_id','offset_mapping']
                            
    def fit(self,traindata,evaldata):
        progress_bar = tqdm(range(self.steps))
        self.model.train()

        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch in traindata:
                batch_ = {k: v.to(self.device) for k, v in batch.items() if k not in self.exclude}

                self.optim.zero_grad()
                outputs = self.model(**batch_)

                loss = outputs.loss
                loss.backward()
                self.optim.step()
                self.scheduler.step()
                epoch_loss += loss.item()
                progress_bar.update(1)
            log.info(f'training loss for epoch {epoch} is {epoch_loss/len(traindata)}')
            self.model_eval(evaldata[0],evaldata[1])
            
    ## evaluation 
    def model_eval(self,eval_dataloader,evaldf,is_eval=True):
        self.model.eval()
        loss = 0
        n = len(eval_dataloader)
        all_batches = {'example_id':[],'offset_mapping':[],'input_ids':[]}
        all_outputs = {'slogits':[],'elogits':[]}
        for batch in eval_dataloader:
            batch_ = {k: v.to(self.device) for k, v in batch.items() if k not in self.exclude}
            with torch.no_grad():
                outputs = self.model(**batch_)
            flat_batch(all_batches,batch)
            flat_outputs(all_outputs,outputs)
            
            if is_eval: loss+= outputs.loss.item()

        
        evaldf = data_postproc(evaldf,all_batches,all_outputs,self.idx2eid,self.cls_token_id)
        if is_eval:
            jac_score = jacard(evaldf)
            log.info(f'valid loss is {loss/n} and jaccard score is {jac_score}')
        return evaldf
    
    def predict(self,data_loader,df,idx2eid):
        train_data_mapper = self.idx2eid
        self.idx2eid = idx2eid
        df = self.model_eval(data_loader,df,False)
        self.idx2eid = train_data_mapper
        return df