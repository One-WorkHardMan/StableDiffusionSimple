
import torch
import torch.nn as nn
from torch.optim import AdamW
import torch.utils.data as tud
import torch.nn.functional as func
from diffusers.schedulers import DDPMScheduler
import ex1

class FakeDataset(tud.Dataset):

    def __init__(self,src_shape = (17,128,128),dst_shape = (1,128,128)):
        self.src_shape = src_shape;
        self.dst_shape = dst_shape;
        self.sample_count = 10000;

    def __len__(self):
        return self.sample_count

    def __getitem__(self, index:int):
        # 生成 随机数作为 假数据
        xx = torch.rand(self.src_shape,dtype=torch.float32)
        yy = torch.rand(self.dst_shape,dtype=torch.float32)
        return xx,yy



#Train_Diffusion
def train_diffusion():
    default_device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    default_type = torch.float32

    #init model
    condition_unet = ex1.Attention_UNet_with_Condition(128,condition_channels=512)
    condition_unet.to(default_device,default_type)
    optimozer = AdamW(condition_unet.parameters(),lr=1e-3,weight_decay=0.05)

    #条件编码器 , ldm 的条件编码和图像编码应该都是用VQ-VAE，Dalle2就是用的CLIP，显然CLIP更加聪明。
    con_ae = ex1.VQVAE(num_channels=128,latent_dim=512)
    con_ae.to(default_device,default_type)
    #加载预训练好的模型 load form checkpoints
    con_ae.requires_grad_(False)
    con_ae.eval()

    #图像编码器，编码到隐空间
    img_ae = ex1.VQVAE(num_channels=17,latent_dim=128)
    img_ae.to(default_device,default_type)
    img_ae.requires_grad_(False)
    img_ae.eval()

    #Init Dataset
    ds = FakeDataset()
    dataloader = tud.DataLoader(ds,batch_size=32,shuffle=True,drop_last=True,num_workers=8)

    #Init DDOM-Model
    DDPM_Scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        prediction_type="epsilon",
        clip_sample=False,

    )
    for batch,(xx,yy) in enumerate(dataloader):
        xx = xx.to(default_device,default_type)
        yy = yy.to(default_device,default_type)

        optimozer.zero_grad()
        with torch.no_grad():























