#mnist数据集上

import argparse
import os

import torch
from torch import autograd, nn
from torch.utils import data
from torchvision import datasets, transforms
import torchvision
from bokeh import plotting

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='../data')
parser.add_argument('--dir', type=str, default='results2')
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--workers', type=int, default=0)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--log-interval', type=int, default=10)
parser.add_argument('--no-cuda', action='store_true')
parser.add_argument('--nrow', type=int, default=16)
parser.add_argument('--c', type=float, default=0.01)
args = parser.parse_args()
print(args)

use_cuda = torch.cuda.is_available() and not args.no_cuda
if use_cuda:
    print("Use CUDA.")


class Generator(nn.Module):
    def __init__(self, ch=8):
        super(Generator, self).__init__()

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(100, ch * 8, 4, 1, 0, 0, bias=False),
            nn.BatchNorm2d(ch * 8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ch * 8, ch * 4, 5, 2, 2, 1, bias=False),
            nn.BatchNorm2d(ch * 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ch * 4, ch * 2, 5, 2, 2, 1, bias=False),
            nn.BatchNorm2d(ch * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ch * 2, ch, 5, 2, 2, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ch, 1, 5, 2, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input_):
        return self.conv(input_.view(input_.size(0), 100, 1, 1))


class Discriminator(nn.Module):
    def __init__(self, ch=8):
        super(Discriminator, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, ch, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ch, ch * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ch * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ch * 2, ch * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ch * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ch * 4, ch * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ch * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ch * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, input_):
        return self.conv(input_).view(input_.size(0))


transform = transforms.Compose([
    transforms.Scale(64),
    transforms.ToTensor(),
])


mnist_loader = data.DataLoader(datasets.MNIST(args.root,
                                              train=True,
                                              transform=transform,
                                              download=True),
                               batch_size=args.batch_size,
                               shuffle=True,
                               num_workers=args.workers)


net_g = Generator()
net_d = Discriminator()

if use_cuda:
    net_g.cuda()
    net_d.cuda()


optimizer_g=torch.optim.Adam(net_g.parameters(), lr = 0.0001 , betas=(0.5,0.9))
optimizer_d=torch.optim.Adam(net_d.parameters(), lr = 0.0001 , betas=(0.5,0.9))

log = {'loss_g': [], 'loss_d': []}


def train(epoch):
    for i, (x, _) in enumerate(mnist_loader):
        # train discriminator
        x = autograd.Variable(x)
        z = autograd.Variable(torch.randn(x.size(0), 100))

        if use_cuda:
            x = x.cuda()
            z = z.cuda()

        fake = net_g(z)#.detach()
        alpha = torch.Tensor(x.size(0), 1, 1, 1).uniform_(0, 1).cuda()
        alpha_ex = alpha.expand(x.size(0), x.shape[1], x.shape[2], x.shape[3])
        interpolates = alpha_ex * x + fake - alpha_ex * fake  # 真实和伪造数据平面之间的数据
        D_interpolates = net_d(interpolates)
        gradients = torch.autograd.grad(D_interpolates, interpolates, grad_outputs=torch.ones(D_interpolates.size()).cuda(),
                            create_graph=True)[0]
        gradients = gradients.view(gradients.shape[0], -1)
        slopes = torch.norm(gradients, 2, 1)
        gradient_penalty = torch.mean(slopes - 1.) ** 2

        fake = fake.detach()
        loss_d = -net_d(x).mean() + net_d(fake).mean() + 10* gradient_penalty

        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step()

        # for p in net_d.parameters():
        #     p.data.clamp_(-args.c, args.c)

        # train generator
        z = autograd.Variable(torch.randn(x.size(0), 100))

        if use_cuda:
            z = z.cuda()
        if i%2==0:
            fake = net_g(z)
            loss_g = -net_d(fake).mean()

            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()

        # log
        log['loss_g'].append(float(loss_g.data))
        log['loss_d'].append(float(loss_d.data))

        if (i + 1) % 40 == 0:
            print('Epoch: {}/{},'.format(epoch + 1, args.epochs),
                  'iter: {}/{},'.format(i + 1,
                                        len(mnist_loader.dataset) // args.batch_size),
                  'loss g: {:.4f},'.format(log['loss_g'][-1]),
                  'loss d: {:.4f}.'.format(log['loss_d'][-1]))
            plot_loss()

z = autograd.Variable(torch.randn(args.nrow ** 2, 100), volatile=True)
if use_cuda:
    z = z.cuda()

def plot_sample(epoch):
    net_g.eval()

    os.makedirs(args.dir, exist_ok=True)
    f = os.path.join(args.dir, 'sample_{}.jpg'.format(str(epoch + 1).zfill(2)))
    torchvision.utils.save_image(net_g(z).data, f, normalize=True, nrow=args.nrow)

    net_g.train()


def plot_loss():
    p = plotting.figure(sizing_mode='stretch_both')
    x = range(len(log['loss_g']))

    p.line(x, log['loss_g'], line_color='green', alpha=0.5, line_width=5, legend='loss g')
    p.line(x, log['loss_d'], line_color='blue', alpha=0.5, line_width=5, legend='loss d')

    os.makedirs(args.dir, exist_ok=True)
    f = os.path.join(args.dir, 'loss.html')
    plotting.output_file(f)
    plotting.save(p)

import time
for epoch in range(args.epochs):
    start=time.time()
    train(epoch)
    end=time.time()
    print(end-start)
    plot_sample(epoch)