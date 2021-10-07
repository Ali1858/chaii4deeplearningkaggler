import logging as log
import torch


## Basic Config 
## confirm this path before running code

PRETRAIN_PATH = 'ai4bharat/indic-bert'
# PRETRAIN_PATH = '/kaggle/input/indicbert/indic-bert-v1'


DATA_DIR = '/home/ubuntu/repo/chaii4deeplearningkaggler/data/'
OUTPUT_DIR = DATA_DIR+'outputs/'

TRAIN_FN = 'train.csv'
TEST_FN = 'test.csv'
SUB_FN = 'sample_submission.csv'

SPLIT_RATION = 0.999
BATCH_SIZE = 8
MAX_SEQ_LENGTH = 384
DOC_STRIDE = 128
LR = 5e-5
NUM_EPOCHS = 1


LOG_FORMAT = '%(asctime)s %(levelname)s %(message)s'
DATE_FORMAT = '%m/%d/%Y %I:%M:%S %p'

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
log.basicConfig(filename='chaii.log', level=log.DEBUG,datefmt=DATE_FORMAT,format=LOG_FORMAT )