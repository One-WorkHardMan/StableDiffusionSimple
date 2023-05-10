
import torch
import torch.nn as nn
from torch.optim import AdamW
import torch.utils.data as tud
import torch.nn.functional as func
import ex1





#Train_Diffusion
def train_diffusion():
    default_device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    default_type = torch.float32

    #init model
    condition_unet = ex1.Attention_UNet_with_Condition(128,condition_channels=512)






















