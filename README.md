# GAN
自己学习实现的DCGAN WGAN LSGAN WGAN-GP(分别对mnist和cifar做实验)
使用gradient reversal layer（Unsupervised Domain Adaptation by Backpropagation） 的trick实验最基础的GAN,　相当于生成器优化的是log(D(1-G(x))).原始GAN中生成器优化的是log(D(G(x)))，因为该损失能在训练初期提供较大的梯度,但并不能把优化目标改成这个形式，实验中效果很差
cDCGAN和cGAN是学习的别人的代码，conditional GAN  其中cDCGAN的实现蛮用意思
