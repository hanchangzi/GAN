#!/usr/bin/python3
import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.datasets import CIFAR10
from pylab import plt
#将图像规范化-1到1,生成器最后一层输出用tanth
transforms=transforms.Compose([
    transforms.Scale(64),#将图片大小放大为64
    transforms.ToTensor(),#将numpy转为tensor
    transforms.Normalize([0.5]*3,[0.5]*3)
])
dataset=CIFAR10(root='cifar10/', transform=transforms,download=True)

dataloader=torch.utils.data.DataLoader(dataset,
                                   32,
                                   shuffle = True,
                                   num_workers=2)

print(type(dataloader))
#判别器
##生成器小trick,不用使得参数稀疏的操作,比如最大赤化,可以用卷积使得矩阵变小,不用relu等用LeakyReLU
netd=nn.Sequential(
    #input n*3*64*64
    nn.Conv2d(3,64,4,2,1,bias=False),#n*64*32*32
    nn.LeakyReLU(0.2,inplace=True),

    nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # n*128*16*16
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.2, inplace=True),

    nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # n*256*8*8
    nn.BatchNorm2d(256),
    nn.LeakyReLU(0.2, inplace=True),

    nn.Conv2d(256, 512, 4, 2, 1, bias=False),  # n*512*4*4
    nn.BatchNorm2d(512),
    nn.LeakyReLU(0.2, inplace=True),

    nn.Conv2d(512,1,4,1,0,bias=False),#n*1*1*1
    nn.Sigmoid()
)
# #生成器
netg=nn.Sequential(
    #input n*100*1*1
    nn.ConvTranspose2d(100,512,4,1,0,bias=False),#n*512*4*4
    nn.BatchNorm2d(512),
    nn.ReLU(True),

    nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),  # n*256*8*8
    nn.BatchNorm2d(256),
    nn.ReLU(True),

    nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # n*128*16*16
    nn.BatchNorm2d(128),
    nn.ReLU(True),

    nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),  # n*64*32*32
    nn.BatchNorm2d(64),
    nn.ReLU(True),

    nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),  # n*3*64*64
    nn.Tanh()


)


#损失函数
criterion=nn.BCELoss()
##损失
def D_loss(real,fake):
    size=real.shape[0]
    true_label = Variable(torch.ones(size)).float().cuda()
    false_label = Variable(torch.zeros(size)).float().cuda()
    return criterion(real,true_label)+criterion(fake,false_label)

def G_loss(fake):
    size = fake.shape[0]
    true_label = Variable(torch.ones(size)).float().cuda()
    return criterion(fake, true_label)


#优化器,用默认参数
optimizerD = Adam(netd.parameters(), 0.0002,betas=(0.5,0.999))
optimizerG = Adam(netg.parameters(), 0.0002,betas=(0.5,0.999))


#将网络放入GPU
netd.cuda()
netg.cuda()
criterion.cuda()

fix_noise = Variable(torch.FloatTensor(32,100,1,1).normal_(0,1)).cuda()
import time
start=time.time()
for epoch in range(10):
    print(epoch)
    for ii,data in enumerate(dataloader,0):
        #训练分类器
        ##真实数据
        realdata,_ =data
        input=Variable(realdata).cuda()
        reallabel=netd(input)
        ##伪造数据
        noise=Variable(torch.randn(input.size(0),100,1,1)).cuda()
        fakedata=netg(noise).detach()
        fakelabel=netd(fakedata)
        #计算分类器损失
        d_loss=D_loss(reallabel.squeeze(),fakelabel.squeeze())
        optimizerD.zero_grad()
        d_loss.backward()
        optimizerD.step()

        ##训练生成器
        noise2 = Variable(torch.randn(input.size(0),100,1,1)).cuda()
        fakedata2 = netg(noise2)
        fakelabel2 = netd(fakedata2)
        #计算生成器损失
        g_loss=G_loss(fakelabel2.squeeze())
        optimizerG.zero_grad()
        g_loss.backward()
        optimizerG.step()
    if epoch % 2 == 0:
        end = time.time()
        print(end - start)
        start=time.time()
        fake_u = netg(fix_noise)
        imgs = make_grid(fake_u.data * 0.5 + 0.5).cpu()  # CHW
        plt.imshow(imgs.permute(1, 2, 0).numpy())  # HWC
        plt.show()


