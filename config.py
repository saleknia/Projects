import os
import torch
import logging
from utils import color
from tabulate import tabulate
import ml_collections

def logger_config(log_path):
    loggerr = logging.getLogger()
    loggerr.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    loggerr.addHandler(handler)
    loggerr.addHandler(console)
    return loggerr

# get_CTranS_config

SEED = 666

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['PYTHONHASHSEED'] = str(SEED)

##########################################################################
# Log Directories
##########################################################################
tensorboard = False
tensorboard_folder = './logs/tensorboard'
log = True
logging_folder = './logs/logging'

if log:
    logging_log = logging_folder
    if not os.path.isdir(logging_log):
        os.makedirs(logging_log)
    logger = logger_config(log_path = logging_log + '/training_log.log')
    logger.info(f'Logging Directory: {logging_log}')   
##########################################################################


# Hyperparameters etc.
LEARNING_RATE = 0.01
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 6
NUM_EPOCHS = 60
NUM_WORKERS = 4
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True
early_stopping = 200

LOAD_MODEL = True
CONTINUE = True

TEACHER = True
# TEACHER = False


SAVE_MODEL = True
COSINE_LR = False
DOWNLOAD = False

os.environ['PYTHONHASHSEED'] = str(SEED)

task_ids = ['1','2','3','4','5','6','7']
task_table = tabulate(
                    tabular_data=[
                        ['COVID-19', 1],
                        ['Synapse', 2],
                        ['ACDC', 3],
                        ['CT-1K', 4],
                        ['SSL', 5],
                        ['TCIA',6],
                        ['camvid',7]],
                    headers=['Task Name', 'ID'],
                    tablefmt="fancy_grid"
                    )

print(task_table)
task_id = input('Enter Task ID:  ')
assert (task_id in task_ids),'ID is Incorrect.'
task_id = int(task_id)

if task_id==1:
    NUM_CLASS = 3
    TASK_NAME = 'COVID-19'
    if os.path.isdir('./data'):
        DOWNLOAD = False
    else:
        DOWNLOAD = True

elif task_id==2:
    NUM_CLASS = 9
    TASK_NAME = 'Synapse'

elif task_id==3:
    NUM_CLASS = 4
    TASK_NAME = 'ACDC'

elif task_id==4:
    NUM_CLASS = 11
    TASK_NAME = 'CT-1K'

elif task_id==5:
    NUM_CLASS = 1
    TASK_NAME = 'SSL'

elif task_id==6:
    NUM_CLASS = 2
    TASK_NAME = 'TCIA'

elif task_id==7:
    NUM_CLASS = 11
    TASK_NAME = 'camvid'

# CONTINUE_ids = ['1','2']
# CONTINUE_table = tabulate(
#                     tabular_data=[
#                         ['True', 1],
#                         ['False', 2]],
#                     headers=['Continue Training', 'ID'],
#                     tablefmt="fancy_grid"
#                     )

# print(CONTINUE_table)
# CONTINUE_id = input('Enter Training ID:  ')
# assert (CONTINUE_id in CONTINUE_ids),'ID is Incorrect.'
# CONTINUE_id = int(CONTINUE_id)

# if CONTINUE_id==1:
#     CONTINUE = True
# elif CONTINUE_id==2:
#     CONTINUE = False


model_ids = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25']
model_table = tabulate(
                    tabular_data=[
                        ['UCTransNet', 1],
                        ['UCTransNet_GT', 2],
                        ['UNet', 3],
                        ['GT_UNet', 4],
                        ['GT_CTrans', 5],
                        ['AttUNet', 6],
                        ['UNet_loss', 7],
                        ['AttUNet_loss', 8],
                        ['MultiResUnet', 9],
                        ['MultiResUnet_loss', 10],
                        ['UNet++', 11],
                        ['UNet++_loss', 12],
                        ['U', 13],
                        ['U_loss', 14],
                        ['ENet', 15],
                        ['ENet_loss', 16],
                        ['ERFNet', 17],
                        ['ERFNet_loss', 18],
                        ['Mobile_netV2', 19],
                        ['Mobile_netV2_loss', 20],
                        ['Fast_SCNN', 21],
                        ['Fast_SCNN_loss', 22],
                        ['ESPNet',23],
                        ['ESPNet_loss',24],
                        ['DABNet_loss',25]],
                    headers=['Model Name', 'ID'],
                    tablefmt="fancy_grid"
                    )

print(model_table)
model_id = input('Enter Model ID:  ')
assert (model_id in model_ids),'ID is Incorrect.'
model_id = int(model_id)


if model_id==1:
    MODEL_NAME = 'UCTransNet'

elif model_id==2:
    MODEL_NAME = 'UCTransNet_GT'

elif model_id==3:
    MODEL_NAME = 'UNet'

elif model_id==4:
    MODEL_NAME = 'GT_UNet'

elif model_id==5:
    MODEL_NAME = 'GT_CTrans'

elif model_id==6:
    MODEL_NAME = 'AttUNet'

elif model_id==7:
    MODEL_NAME = 'UNet_loss'

elif model_id==8:
    MODEL_NAME = 'AttUNet_loss'

elif model_id==9:
    MODEL_NAME = 'MultiResUnet'

elif model_id==10:
    MODEL_NAME = 'MultiResUnet_loss'

elif model_id==11:
    MODEL_NAME = 'UNet++'

elif model_id==12:
    MODEL_NAME = 'UNet++_loss'

elif model_id==13:
    MODEL_NAME = 'U'

elif model_id==14:
    MODEL_NAME = 'U_loss'

elif model_id==15:
    MODEL_NAME = 'ENet'

elif model_id==16:
    MODEL_NAME = 'ENet_loss'

elif model_id==17:
    MODEL_NAME = 'ERFNet'

elif model_id==18:
    MODEL_NAME = 'ERFNet_loss'

elif model_id==19:
    MODEL_NAME = 'Mobile_NetV2'

elif model_id==20:
    MODEL_NAME = 'Mobile_NetV2_loss'

elif model_id==21:
    MODEL_NAME = 'Fast_SCNN'

elif model_id==22:
    MODEL_NAME = 'Fast_SCNN_loss'

elif model_id==23:
    MODEL_NAME = 'ESPNet'

elif model_id==24:
    MODEL_NAME = 'ESPNet_loss'

elif model_id==25:
    MODEL_NAME = 'DABNet_loss'

CKPT_NAME = MODEL_NAME + '_' + TASK_NAME


# if model_id==1:
#     MODEL_NAME = 'UCTransNet'
#     if task_id==1:
#         CKPT_NAME = 'UCTransNet_Covid'
#     elif task_id==2:
#         CKPT_NAME = 'UCTransNet_Synapse'   
#     elif task_id==3:
#         CKPT_NAME = 'UCTransNet_ACDC'   
# elif model_id==2:
#     MODEL_NAME = 'UCTransNet_GT'
#     if task_id==1:
#         CKPT_NAME = 'UCTransNet_GT_Covid'
#     elif task_id==2:
#         CKPT_NAME = 'UCTransNet_GT_Synapse'   
# elif model_id==3:
#     MODEL_NAME = 'UNet'
#     if task_id==1:
#         CKPT_NAME = 'UNet_Covid'
#     elif task_id==2:
#         CKPT_NAME = 'UNet_Synapse'  
#     elif task_id==3:
#         CKPT_NAME = 'UNet_ACDC' 
# elif model_id==4:
#     MODEL_NAME = 'GT_UNet'
#     if task_id==1:
#         CKPT_NAME = 'GT_UNet_Covid'
#     elif task_id==2:
#         CKPT_NAME = 'GT_UNet_Synapse'  
# elif model_id==5:
#     MODEL_NAME = 'GT_CTrans'
#     if task_id==1:
#         CKPT_NAME = 'GT_CTrans_Covid'
#     elif task_id==2:
#         CKPT_NAME = 'GT_CTrans_Synapse'  
# elif model_id==6:
#     MODEL_NAME = 'AttUNet'
#     if task_id==1:
#         CKPT_NAME = 'AttUNet_Covid'
#     elif task_id==2:
#         CKPT_NAME = 'AttUNet_Synapse'  
#     elif task_id==3:
#         CKPT_NAME = 'AttUNet_ACDC'  
# elif model_id==7:
#     MODEL_NAME = 'UNet_loss'
#     if task_id==1:
#         CKPT_NAME = 'UNet_loss_Covid'
#     elif task_id==2:
#         CKPT_NAME = 'UNet_loss_Synapse'
#     elif task_id==3:
#         CKPT_NAME = 'UNet_loss_ACDC' 
# elif model_id==8:
#     MODEL_NAME = 'AttUNet_loss'
#     if task_id==1:
#         CKPT_NAME = 'AttUNet_loss_Covid'
#     elif task_id==2:
#         CKPT_NAME = 'AttUNet_loss_Synapse' 
#     elif task_id==3:
#         CKPT_NAME = 'AttUNet_loss_ACDC' 
# elif model_id==9:
#     MODEL_NAME = 'MultiResUnet'
#     if task_id==1:
#         CKPT_NAME = 'MultiResUnet_Covid'
#     elif task_id==2:
#         CKPT_NAME = 'MultiResUnet_Synapse' 
# elif model_id==10:
#     MODEL_NAME = 'MultiResUnet_loss'
#     if task_id==1:
#         CKPT_NAME = 'MultiResUnet_loss_Covid'
#     elif task_id==2:
#         CKPT_NAME = 'MultiResUnet_loss_Synapse' 
# elif model_id==11:
#     MODEL_NAME = 'UNet++'
#     if task_id==1:
#         CKPT_NAME = 'UNet++_Covid'
#     elif task_id==2:
#         CKPT_NAME = 'UNet++_Synapse' 
# elif model_id==12:
#     MODEL_NAME = 'UNet++_loss'
#     if task_id==1:
#         CKPT_NAME = 'UNet++_loss_Covid'
#     elif task_id==2:
#         CKPT_NAME = 'UNet++_loss_Synapse' 
# elif model_id==13:
#     MODEL_NAME = 'U'
#     if task_id==1:
#         CKPT_NAME = 'U_Covid'
#     elif task_id==2:
#         CKPT_NAME = 'U_Synapse' 
#     elif task_id==3:
#         CKPT_NAME = 'U_ACDC' 
# elif model_id==14:
#     MODEL_NAME = 'U_loss'
#     if task_id==1:
#         CKPT_NAME = 'U_loss_Covid'
#     elif task_id==2:
#         CKPT_NAME = 'U_loss_Synapse' 
#     elif task_id==3:
#         CKPT_NAME = 'U_loss_ACDC' 

table = tabulate(
    tabular_data=[
        ['Learning Rate', LEARNING_RATE],
        ['Num Classes', NUM_CLASS],
        ['Device', DEVICE],
        ['Batch Size', BATCH_SIZE],
        ['COSINE LR', COSINE_LR],
        ['Num Epochs', NUM_EPOCHS],
        ['Num Workers', NUM_WORKERS],
        ['Image Height', IMAGE_HEIGHT],
        ['Image Width', IMAGE_WIDTH],
        ['Pin Memory', PIN_MEMORY],
        ['Early Stopping',early_stopping],
        ['Load Model', LOAD_MODEL],
        ['Save Model', SAVE_MODEL],
        ['Download Dataset', DOWNLOAD],
        ['Model Name', MODEL_NAME],
        ['Seed', SEED],
        ['Task Name', TASK_NAME],
        ['GPU', torch.cuda.get_device_name(0)],
        ['Checkpoint Name', CKPT_NAME]],
    headers=['Hyperparameter', 'Value'],
    tablefmt="fancy_grid"
    )

logger.info(table)



##########################################################################
# CTrans configs
##########################################################################
def get_CTranS_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 960  # KV_size = Q1 + Q2 + Q3 + Q4
    config.transformer.num_heads  = 4
    config.transformer.num_layers = 4
    config.expand_ratio           = 4  # MLP channel dimension expand ratio
    # config.transformer.embeddings_dropout_rate = 0.3
    # config.transformer.attention_dropout_rate  = 0.3
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate  = 0.1
    config.transformer.dropout_rate = 0
    config.patch_sizes = [16,8,4,2]
    # config.patch_sizes = [8,4,2,1]
    config.base_channel = 64 # base channel of U-Net
    config.n_classes = NUM_CLASS
    return config

n_channels = 1
n_labels = NUM_CLASS







