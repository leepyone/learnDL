# for DCGAN

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from IPython.display import HTML

# 设置一下随机种子 作用？ todo
manualSeed = 9999
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# 参数的定义
dataroot = './data/celeba'

workers = 2
batch_size = 128
image_size = 128
# 图像的通道数
nc = 3
# 生成器从噪声中获取的向量的维度
nz = 100
# 生成器与构造器 featuremap的大小应该是卷积的，最后提取的特征个数？todo
ngf = 64
ndf = 64

num_epoch = 5
lr = 0.0002
# Adam 优化器的参数？
beta1 = 0.5
ngpu = 1


def getData():
    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([  # Compose的作用就是把下面的操作连在一起
                                   transforms.Resize(image_size),  # 调整大小
                                   transforms.CenterCrop(image_size),  # 从中心裁剪
                                   transforms.ToTensor(),  # 把每个值缩放到 0-1
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 进行归一化从操作
                               ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    # print(f'dataloader len {len(dataloader)}')
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print(f'device is:{device}')
    # 把图画出来
    real_batch = next(iter(dataloader))
    # print("real_batch size")
    # print(len(real_batch))
    # print(f'real_batch is {real_batch}')
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("训练的样本")
    plt.imshow(np.transpose(  # transpose 应该是交换不同维度的值 ，像下面的就是把第一维度的移到最后
        vutils.make_grid(
            real_batch[0].to(device)[:64],
            padding=2,
            normalize=True
        ).cpu()
        , (1, 2, 0)))
    plt.show()

    return device, dataloader


# 权重初始化
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# 生成器模型
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu

        self.main = nn.Sequential(
            #
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 0, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ndf * 2, ngf, 4, 2, 0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 0, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


# 鉴别器模型
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        self.main(input)


if __name__ == '__main__':
    device, dataloader = getData()
    # 创建网络
    netG = Generator(ngpu).to(device=device)
    netG.apply(weights_init)
    print(netG)
    netD = Discriminator(ngpu).to(device=device)
    netD.apply(weights_init)
    print(netD)
    # 构建损失函数
    # 二值交叉熵损失函数
    criterion = nn.BCELoss()
    # 高斯分布 从中获取 原始的向量
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    real_label = 1.
    fake_label = 0.
    # 创建优化器
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # 训练的过程

    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    print("开始训练")
    for epoch in range(num_epoch):
        print(f"第{epoch + 1}次迭代")
        # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中
        # 0是设置起始下表的位置
        # 这边会取出所有的样本进行训练
        for i, data in enumerate(dataloader, 0):
            # 鉴别器的更新
            # 清空梯度
            netD.zero_grad()
            # 拿到数据
            real_cpu = data[0].to(device)
            # 获取到这一批的大小
            b_size = real_cpu.size(0)
            # 获取标签
            # 这个是构造标签的tensor full用法就是用某个数字充满
            label = torch.full((b_size,),
                               real_label, dtype=torch.float, device=device)
            # 获得正向传播的结果
            # view函数作用为重构张量的维度 这里将tensor 的维度调整到了一维
            output = netD(real_cpu).view(-1)
            # 获得误差
            errD_real = criterion(output, label)
            # 计算梯度
            errD_real.backwark()
            # item() 函数获取具体的数值
            D_x = output.mean().item()

            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            # 加了_ 就是原地修改
            # 修改label成fake的标签
            label.fill_(fake_label)
            # 返回一个新的tensor，从当前计算图中分离下来的，但是仍指向原变量的存放位置,不同之处只是requires_grad为false，得到的这个tensor永远不需要计算其梯度，不具有grad
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()
            # 以上使用了 正负两种样本训练了模型，用模型正向传播了两个，然后更新一次参数

            #下面是生成器的训练部分
            netD.zero_grad()
            label.fill_(real_label)
            output = netD(fake).view(-1)
            errG = criterion(output,label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            if(i%50==0):
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epoch, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            # True 的条件是 这次迭代每取出了500个batch 或者是 这是最后一次迭代并且是最后一个样本
            if(iters%500 ==0) or ((epoch==num_epoch-1) and (i== len(dataloader-1))):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake,padding=2,normalize=True))

            iters+=1

    print("训练结束")
    plt.figure(figsize=(10,5))
    plt.title("loss 随着迭代的变化图")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("迭代次数")
    plt.ylim("Loss")
    plt.legend()
    plt.show()


    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("生成的图片")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)),animation=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig,ims,interval=1000,repeat_delay=1000,blit=True)
    HTML(ani.to_jshtml())

