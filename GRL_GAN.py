#!/usr/bin/python3
from torch import nn
from torch.autograd import Variable,Function

import torchvision.transforms as tfs
from torch.utils.data import DataLoader, sampler
from torchvision.datasets import MNIST
import torch
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
def show_images(images): # 定义画图工具
    images = np.reshape(images, [images.shape[0], -1])
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtimg,sqrtimg]))
    return

def preprocess_img(x):
    x = tfs.ToTensor()(x)
    return (x - 0.5) / 0.5

def deprocess_img(x):
    return (x + 1.0) / 2.0


class ChunkSampler(sampler.Sampler): # 定义一个取样的函数
    """Samples elements sequentially from some offset.
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples

NUM_TRAIN = 50000
NUM_VAL = 5000

NOISE_DIM = 96
batch_size = 128

train_set = MNIST('./mnist', train=True, download=True, transform=preprocess_img)

train_data = DataLoader(train_set, batch_size=batch_size, sampler=ChunkSampler(NUM_TRAIN, 0))

val_set = MNIST('./mnist', train=True, download=True, transform=preprocess_img)

val_data = DataLoader(val_set, batch_size=batch_size, sampler=ChunkSampler(NUM_VAL, NUM_TRAIN))


imgs = deprocess_img(train_data.__iter__().next()[0].view(batch_size, 784)).numpy().squeeze() # 可视化图片效果
show_images(imgs)


class ReverseLayerF(Function):
    def __init__(self):
        super(ReverseLayerF,self).__init__()
    def forward(ctx, x):
        return x

    def backward(ctx, grad_output):
        output = grad_output* -1

        return output#, None
relf = ReverseLayerF()

class build_dc_classifier(nn.Module):
    def __init__(self):
        super(build_dc_classifier,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(1,32,5,1),# d 32 24 24
            nn.LeakyReLU(0.01),
            nn.AvgPool2d(2,2),
            #nn.MaxPool2d(2,2),#d 32 12 12
            nn.Conv2d(32,64,5,1),#d 64 8 8
            nn.AvgPool2d(2, 2)
            #nn.MaxPool2d(2,2) # d 64 4 4
        )
        self.fc=nn.Sequential(
            nn.Linear(1024,1024),
            nn.LeakyReLU(0.01),
            nn.Linear(1024,1)
        )
    def forward(self, x):
            x=self.conv(x)
            x=x.view(x.shape[0],-1)
            x=self.fc(x)
            return x


class build_dc_generator(nn.Module):
    def __init__(self, noise_dim=NOISE_DIM):
        super(build_dc_generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(noise_dim, 1024),
            nn.ReLU(True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 7 * 7 * 128),
            nn.ReLU(True),
            nn.BatchNorm1d(7 * 7 * 128)
        )

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 1, 4, 2, padding=1),
            nn.Tanh()
        )
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0], 128, 7, 7)  # reshape 通道是 128，大小是 7x7
        x = self.conv(x)
        x=relf(x)
        #x = ReverseLayerF.apply(x,-1)
        return x




bce_loss=nn.BCEWithLogitsLoss()#二分类的损失函数,类似的还有BCELoss,但BCELoss需要在最后一层手动的加入sigmoid层,而BCEWithLogitsLoss不需要
#ylogx+(1-y)logx  y和x是穿进去的参数
def discriminator_loss(logits_real,logits_fake):
    size=logits_real.shape[0]
    true_labels=Variable(torch.ones(size,1)).float().cuda()
    false_labels=Variable(torch.zeros(size,1)).float().cuda()
    loss=bce_loss(logits_real,true_labels) + bce_loss(logits_fake,false_labels)
    #loss = bce_loss(logits_real, true_labels) + bce_loss(logits_real, false_labels)
    return loss





import time
print(torch.cuda.is_available())
def train_a_gan(D_net, G_net, optimizer, discriminator_loss, show_every=250,
                noise_size=96, num_epochs=20,Type=0):
    iter_count = 0
    for epoch in range(num_epochs):
        start=time.time()
        for x, _ in train_data:
            bs = x.shape[0]
            # 判别网络
            if Type==0:
                real_data = Variable(x).view(bs, -1).cuda()  # 真实数据
            else:
                real_data = Variable(x).cuda()  # 真实数据
            logits_real = D_net(real_data)  # 判别网络得分
            sample_noise = (torch.rand(bs, noise_size) - 0.5) / 0.5  # -1 ~ 1 的均匀分布
            g_fake_seed = Variable(sample_noise).cuda()
            fake_images = G_net(g_fake_seed)  # 生成的假的数据
            logits_fake = D_net(fake_images)  # 判别网络得分

            total_error = discriminator_loss(logits_real, logits_fake)  # 判别器的 loss
            optimizer.zero_grad()
            total_error.backward()
            optimizer.step()  # 优化判别网络


            if (iter_count % show_every == 0):
                print('Iter: {}, D: {:.4}'.format(iter_count, total_error.item()))
                imgs_numpy = deprocess_img(fake_images.data.cpu().numpy())
                show_images(imgs_numpy[0:16])
                plt.show()
                print()
            iter_count += 1
        end=time.time()
        print(end-start)
#---CNN网络---#
D_DC = build_dc_classifier().cuda()
G_DC = build_dc_generator().cuda()



optimizer = torch.optim.Adam(list(D_DC.parameters())+list(G_DC.parameters()), lr=1e-3, betas=(0.9, 0.999))

train_a_gan(D_DC, G_DC, optimizer , discriminator_loss, num_epochs=10,Type=1)
