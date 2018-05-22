#!/usr/bin/python3
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
    transforms.Resize(32),#将图片大小放大为64
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
    #input n*3*32*32
    nn.Conv2d(3,64,4,2,1,bias=False),#n*64*16*16
    nn.LeakyReLU(0.2,inplace=True),

    nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # n*128*8*8
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.2, inplace=True),

    nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # n*256*4*4
    nn.BatchNorm2d(256),
    nn.LeakyReLU(0.2, inplace=True),

    nn.Conv2d(256,1,4,1,0,bias=False),#n*1*1*1
    nn.Sigmoid()
)
# #生成器
netg=nn.Sequential(
    #input n*100*1*1
    nn.ConvTranspose2d(100,256,4,1,0,bias=False),#n*256*4*4
    nn.BatchNorm2d(256),
    nn.ReLU(True),

    nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # n*128*8*8
    nn.BatchNorm2d(128),
    nn.ReLU(True),

    nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),  # n*64*16*16
    nn.BatchNorm2d(64),
    nn.ReLU(True),

    nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),  # n*3*32*32
    nn.Tanh()


)
#优化器,用默认参数
# optimizerG = torch.optim.RMSprop(netg.parameters(),
#                                   lr=2e-4)
# optimizerD = torch.optim.RMSprop(netd.parameters(),
#                                   lr=2e-4)

optimizerG = torch.optim.Adam(netg.parameters(), lr = 0.0001 , betas=(0.5,0.9))
optimizerD = torch.optim.Adam(netd.parameters(), lr = 0.0001 , betas=(0.5,0.9))

def weight_init(m):
    # weight_initialization: important for wgan
    class_name=m.__class__.__name__
    if class_name.find('Conv')!=-1:
        m.weight.data.normal_(0,0.02)
    elif class_name.find('Norm')!=-1:
        m.weight.data.normal_(1.0,0.02)
#     else:print(class_name)

netd.apply(weight_init)
netg.apply(weight_init)
#将网络放入GPU
netd.cuda()
netg.cuda()

from torchvision.utils import save_image
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
        ##伪造数据
        noise=Variable(torch.randn(input.size(0),100,1,1)).cuda()
        fakedata=netg(noise)
        #求得gradient_penalty
        alpha = torch.Tensor(input.size(0), 1, 1, 1).uniform_(0, 1).cuda()
        alpha_ex = alpha.expand(input.shape)
        interpolates = alpha_ex * input + fakedata - alpha_ex * fakedata  # 真实和伪造数据平面之间的数据
        D_interpolates = netd(interpolates)
        gradients = torch.autograd.grad(D_interpolates, interpolates, grad_outputs=torch.ones(D_interpolates.size()).cuda(),
                            create_graph=True)[0]
        gradients = gradients.view(gradients.shape[0], -1)
        slopes = torch.norm(gradients, 2, 1)
        gradient_penalty = torch.mean(slopes - 1.) ** 2
        #求得损失
        fakedata = fakedata.detach()
        d_loss = -netd(input).mean() + netd(fakedata).mean() + 10 * gradient_penalty
        #优化梯度
        optimizerD.zero_grad()
        d_loss.backward()
        optimizerD.step()



        if (ii+1)%5==0:
        ##训练生成器
            noise2 = Variable(torch.randn(input.size(0),100,1,1)).cuda()
            fakedata2 = netg(noise2)
            fakelabel2 = netd(fakedata2)
            #计算生成器损失
            g_loss=-fakelabel2.mean()
            optimizerG.zero_grad()
            g_loss.backward()
            optimizerG.step()
    if epoch % 2 == 0:
        end = time.time()
        print(end - start)
        start=time.time()
        fake_u = netg(fix_noise)
        imgs = make_grid(fake_u.data * 0.5 + 0.5).cpu()  # CHW
        # plt.imshow(imgs.permute(1, 2, 0).numpy())  # H
        # plt.show()
        imgs = make_grid(fake_u.data * 0.5 + 0.5).cpu()  # CHW
        save_image(imgs, 'W '+str(epoch) + '.jpg')