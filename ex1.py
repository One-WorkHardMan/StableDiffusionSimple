








import torch;
import torch.nn as nn;
#---------------------------------------------------------------------------------------------------------------------------
#现在都是些分块来进行，训练不是直接写一个网络
#常规的卷积块
class ConvBlock(nn.Module):
    def __init__(self,num_channels:int):
        super.__init__();
        self.layers = nn.Sequential(
            nn.Conv2d (num_channels,num_channels,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),

            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
        )

    def forward(self,inputs:torch.Tensor)->torch.Tensor:
        h = self.layers(inputs)
        return h
#---------------------------------------------------------------------------------------------------------------------------
#残差连接
class ResBlock(nn.Module):
    def __init__(self,num_channels:int):
        super(ResBlock, self).__init__();
        self.residual = nn.Sequential(
            # nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(num_channels),
            # nn.ReLU(),
            #
            # nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(num_channels),
            # nn.ReLU(),

            #不过后来何开明做了个改进:效果更好
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self,inputs:torch.Tensor)->torch.Tensor:
        h = inputs + self.residual(inputs)
        return h
#---------------------------------------------------------------------------------------------------------------------------
#下采样和上采样 简化版Unet ，没有 前后相加，但是有残差；
class AutoEncoder(nn.Module):
    def __init__(self,num_channels = 3,ch:int = 64):
        super(AutoEncoder, self).__init__()

        """
            模型的输入，输出，需要单独拿出来写，这样可以增加模型复用性，比如我这里Encoder和Decoder训练好了
            我可以只改变 convin 的通道，和输出通道，中间的参数不动
            做FineTune。
        """

        self.conv_in = nn.Conv2d(num_channels,1*ch,kernel_size=3,stride=1,padding=1)

        self.Encoder = nn.Sequential(
            #下采样Stage 1
            nn.Sequential(
                ResBlock(1*ch),
                ResBlock(1*ch)
            ),
            # 220 的好处就是，刚好长宽缩小一般，通道增大一倍
            #Stage2
            nn.Conv2d(1*ch,2*ch,kernel_size=2,stride=2,padding=0),
            nn.Sequential(
                ResBlock(2 * ch),
                ResBlock(2 * ch)
            ),
            # Stage3
            nn.Conv2d(2*ch,4*ch,kernel_size=2,stride=2,padding=0),
            nn.Sequential(
                ResBlock(4 * ch),
                ResBlock(4 * ch)
            ),
            # Stage4
            nn.Conv2d(4 * ch, 8 * ch, kernel_size=2, stride=2, padding=0),
            nn.Sequential(
                ResBlock(8* ch),
                ResBlock(8* ch)
            ),
        )

        #上采样
        self.Decoder = nn.Sequential(
            #Stage4
            nn.Sequential(
                ResBlock(8 * ch),
                ResBlock(8 * ch)
            ),

            nn.ConvTranspose2d(8 * ch,4*ch,kernel_size=2,stride=2,padding=0),
            nn.Sequential(
                ResBlock(4* ch),
                ResBlock(4 * ch)
            ),

            nn.ConvTranspose2d(4 * ch, 2 * ch, kernel_size=2, stride=2, padding=0),
            nn.Sequential(
                ResBlock(2 * ch),
                ResBlock(2 * ch)
            ),

            nn.ConvTranspose2d(2 * ch, 1 * ch, kernel_size=2, stride=2, padding=0),
            nn.Sequential(
                ResBlock(1 * ch),
                ResBlock(1 * ch)
            )
        )

        self.conv_out =  nn.Conv2d(1*ch,num_channels,kernel_size=3,stride=1,padding=1)

    def forward(self,inputs:torch.Tensor)->torch.Tensor:
        h = self.conv_in(inputs)
        h = self.Encoder(h)
        h = self.Decoder(h)
        h = self.conv_out(h)
        return h
#---------------------------------------------------------------------------------------------------------------------------

"""
    VAE 来试试 Diffusers这个包 导入预训练好的模型吧：
"""

from diffusers.models.vae import DiagonalGaussianDistribution

class VAE(nn.Module):
    def __init__(self,num_channels = 3,latent_dim:int=128 ,ch:int = 64):
        super(VAE, self).__init__()

        """
            模型的输入，输出，需要单独拿出来写，这样可以增加模型复用性，比如我这里Encoder和Decoder训练好了
            我可以只改变 convin 的通道，和输出通道，中间的参数不动
            做FineTune。
        """

        self.conv_in = nn.Conv2d(num_channels,1*ch,kernel_size=3,stride=1,padding=1)

        self.Encoder = nn.Sequential(
            #下采样Stage 1
            nn.Sequential(
                ResBlock(1*ch),
                ResBlock(1*ch)
            ),
            # 220 的好处就是，刚好长宽缩小一般，通道增大一倍
            #Stage2
            nn.Conv2d(1*ch,2*ch,kernel_size=2,stride=2,padding=0),
            nn.Sequential(
                ResBlock(2 * ch),
                ResBlock(2 * ch)
            ),
            # Stage3
            nn.Conv2d(2*ch,4*ch,kernel_size=2,stride=2,padding=0),
            nn.Sequential(
                ResBlock(4 * ch),
                ResBlock(4 * ch)
            ),
            # Stage4
            nn.Conv2d(4 * ch, 8 * ch, kernel_size=2, stride=2, padding=0),
            nn.Sequential(
                ResBlock(8* ch),
                ResBlock(8* ch)
            ),
        )

        #隐含变量z
        self.proj1 = nn.Conv2d(8 * ch,2*latent_dim,kernel_size=1,stride=1,padding=0),
        #z
        #为啥是 latent_dim， VAE里面，经过Encoder之后，然后参数重整化之后向量的维度就是 上面的 2*latent_dim 一半。
        self.proj2 = nn.Conv2d(latent_dim,8*ch,kernel_size=1,stride=1,padding=0),



        #上采样
        self.Decoder = nn.Sequential(
            #Stage4
            nn.Sequential(
                ResBlock(8 * ch),
                ResBlock(8 * ch)
            ),

            nn.ConvTranspose2d(8 * ch,4*ch,kernel_size=2,stride=2,padding=0),
            nn.Sequential(
                ResBlock(4* ch),
                ResBlock(4 * ch)
            ),

            nn.ConvTranspose2d(4 * ch, 2 * ch, kernel_size=2, stride=2, padding=0),
            nn.Sequential(
                ResBlock(2 * ch),
                ResBlock(2 * ch)
            ),

            nn.ConvTranspose2d(2 * ch, 1 * ch, kernel_size=2, stride=2, padding=0),
            nn.Sequential(
                ResBlock(1 * ch),
                ResBlock(1 * ch)
            )
        )

        self.conv_out =  nn.Conv2d(1*ch,num_channels,kernel_size=3,stride=1,padding=1)

    def encode(self,inputs:torch.Tensor,sampling:bool = False,return_loss:bool = False):
        h = self.conv_in(inputs)
        h = self.Encoder(h)
        h = self.proj1(h)

        # 隐变量传进去分成两部分：
        latent_dist = DiagonalGaussianDistribution(h)

        if sampling:
        # 参数重整化的方式进行采样
            return latent_dist.sample()

        # 训练阶段
        elif return_loss:
            #VAE 的 KL—loss
            kl_loss = latent_dist.kl()
            return latent_dist.sample(),kl_loss
        else:
            return latent_dist.mode()

    def decode(self,inputs)->torch.Tensor:
        h = self.proj2(inputs)
        h = self.Decoder(h)
        return h


    # def forward(self,inputs:torch.Tensor)->torch.Tensor:
    #     h = self.conv_in(inputs)
    #     h = self.Encoder(h)
    #     h = self.Decoder(h)
    #     h = self.conv_out(h)
    #     return h

#------------------------------------------------------------------------------------------------------------------------

from diffusers.models.vae import VectorQuantizer
# 引入向量量化器
class VQVAE(nn.Module):
    def __init__(self,num_channels = 3,latent_dim:int=128 ,ch:int = 64):
        super(VQVAE, self).__init__()

        """
            模型的输入，输出，需要单独拿出来写，这样可以增加模型复用性，比如我这里Encoder和Decoder训练好了
            我可以只改变 convin 的通道，和输出通道，中间的参数不动
            做FineTune。
        """

        self.conv_in = nn.Conv2d(num_channels,1*ch,kernel_size=3,stride=1,padding=1)

        self.Encoder = nn.Sequential(
            #下采样Stage 1
            nn.Sequential(
                ResBlock(1*ch),
                ResBlock(1*ch)
            ),
            # 220 的好处就是，刚好长宽缩小一般，通道增大一倍
            #Stage2
            nn.Conv2d(1*ch,2*ch,kernel_size=2,stride=2,padding=0),
            nn.Sequential(
                ResBlock(2 * ch),
                ResBlock(2 * ch)
            ),
            # Stage3
            nn.Conv2d(2*ch,4*ch,kernel_size=2,stride=2,padding=0),
            nn.Sequential(
                ResBlock(4 * ch),
                ResBlock(4 * ch)
            ),
            # Stage4
            nn.Conv2d(4 * ch, 8 * ch, kernel_size=2, stride=2, padding=0),
            nn.Sequential(
                ResBlock(8* ch),
                ResBlock(8* ch)
            ),
        )

        #隐含变量z
        self.proj1 = nn.Conv2d(8 * ch,latent_dim,kernel_size=1,stride=1,padding=0),
        #z
        self.vq = VectorQuantizer(n_e=8192,vq_embed_dim=latent_dim,beta=0.2,legacy=False)

        self.proj2 = nn.Conv2d(latent_dim,8*ch,kernel_size=1,stride=1,padding=0),



        #上采样
        self.Decoder = nn.Sequential(
            #Stage4
            nn.Sequential(
                ResBlock(8 * ch),
                ResBlock(8 * ch)
            ),

            nn.ConvTranspose2d(8 * ch,4*ch,kernel_size=2,stride=2,padding=0),
            nn.Sequential(
                ResBlock(4* ch),
                ResBlock(4 * ch)
            ),

            nn.ConvTranspose2d(4 * ch, 2 * ch, kernel_size=2, stride=2, padding=0),
            nn.Sequential(
                ResBlock(2 * ch),
                ResBlock(2 * ch)
            ),

            nn.ConvTranspose2d(2 * ch, 1 * ch, kernel_size=2, stride=2, padding=0),
            nn.Sequential(
                ResBlock(1 * ch),
                ResBlock(1 * ch)
            )
        )

        self.conv_out =  nn.Conv2d(1*ch,num_channels,kernel_size=3,stride=1,padding=1)

    def encode(self,inputs:torch.Tensor,sampling:bool = False,return_loss:bool = False):
        h = self.conv_in(inputs)
        h = self.Encoder(h)
        h = self.proj1(h)

        #return z_q, loss, (perplexity, min_encodings, min_encoding_indices) , 这个VQ 要返回几个东西，ze量化之后的结果，loss
        #min_encoding_indices 码本的索引；
        z_q,loss,_ = self.vq(h)
        if sampling:
            return z_q
        elif return_loss:
            #train 训练的时候一定要用到loss
            return z_q,loss
        else:
            return h


    def decode(self,inputs)->torch.Tensor:
        h = self.proj2(inputs)
        h = self.Decoder(h)
        return h

#-------------------------------------------------------------------------------------------------------------------------





































