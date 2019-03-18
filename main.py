# -*- coding: utf-8 -*-

# mkdir check_points
# %matplotlib inline
# pip install -U -q kaggle

# kaggle datasets download jessicali9530/celeba-dataset -f img_align_celeba.zip -q
# unzip -q img_align_celeba.zip

# mkdir img_align_celeba/faces
# rsync -r --include='*.jpg' --exclude='*' img_align_celeba/ img_align_celeba/faces
# find ./img_align_celeba -maxdepth 1 -type f -delete

# ls img_align_celeba/

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from pathlib import Path
from collections import OrderedDict
from math import ceil

data_folder = 'img_align_celeba'
# plt.imshow(Image.open(next(Path(data_folder+'/faces').iterdir())))

samples_count = len(sorted(Path(data_folder + '/faces').iterdir()))

torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

n_z = 64
cnl = 3
n = 128

lr = 0.0001
weight_decay = 0.0005

lam = 0.001
gamma = 0.4
eta = 1
k = 0

batch_size = 32
input_size = 128
epochs = 10


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        self.dec_h0 = nn.Sequential(
            OrderedDict([
                ('lin_0', nn.Linear(n_z, 8 * 8 * n)),
            ]))
        self.dec_block_1 = nn.Sequential(
            OrderedDict([
                ('conv_1_1', nn.Conv2d(n, n, 3, padding=1)),
                ('elu_1_1', nn.ELU()),
                ('conv_1_2', nn.Conv2d(n, n, 3, padding=1)),
                ('elu_1_2', nn.ELU()),
            ]))
        self.dec_block_2 = nn.Sequential(
            OrderedDict([
                ('bn_2', nn.BatchNorm2d(n)),
                ('conv_2_1', nn.Conv2d(n, n, 3, padding=1)),
                ('elu_2_1', nn.ELU()),
                ('conv_2_2', nn.Conv2d(n, n, 3, padding=1)),
                ('elu_2_2', nn.ELU()),
            ]))
        self.dec_block_3 = nn.Sequential(
            OrderedDict([
                ('bn_3', nn.BatchNorm2d(n)),
                ('conv_3_1', nn.Conv2d(n, n, 3, padding=1)),
                ('elu_3_1', nn.ELU()),
                ('conv_3_2', nn.Conv2d(n, n, 3, padding=1)),
                ('elu_3_2', nn.ELU()),
            ]))
        self.dec_block_4 = nn.Sequential(
            OrderedDict([
                ('bn_4', nn.BatchNorm2d(n)),
                ('conv_4_1', nn.Conv2d(n, n, 3, padding=1)),
                ('elu_4_1', nn.ELU()),
                ('conv_4_2', nn.Conv2d(n, n, 3, padding=1)),
                ('elu_4_2', nn.ELU()),
            ]))
        self.dec_block_5 = nn.Sequential(
            OrderedDict([
                ('bn_5', nn.BatchNorm2d(n)),
                ('conv_5_1', nn.Conv2d(n, n, 3, padding=1)),
                ('elu_5_1', nn.ELU()),
                ('conv_5_2', nn.Conv2d(n, n, 3, padding=1)),
                ('elu_5_2', nn.ELU()),
            ]))
        self.dec_out = nn.Sequential(
            OrderedDict([
                ('conv_out', nn.Conv2d(n, cnl, 3, padding=1)),
            ]))

    def forward(self, x):
        x = self.dec_h0(x).view(-1, n, 8, 8)

        x_1 = F.interpolate(self.dec_block_1(x), scale_factor=2)
        #res_1 = F.interpolate(x, scale_factor=2)
        out_1 = x_1# + res_1

        x_2 = F.interpolate(self.dec_block_2(out_1), scale_factor=2)
        res_2 = F.interpolate(out_1, scale_factor=2)
        out_2 = x_2 + res_2

        x_3 = F.interpolate(self.dec_block_3(out_2), scale_factor=2)
        res_3 = F.interpolate(out_2, scale_factor=2)
        out_3 = x_3 + res_3

        x_4 = F.interpolate(self.dec_block_4(out_3), scale_factor=2)
        res_4 = F.interpolate(out_3, scale_factor=2)
        out_4 = x_4 + res_4

        x_5 = self.dec_block_5(out_4)
        out = self.dec_out(x_5)

        return out


gen = Decoder().to(device)
gen.apply(weights_init_normal)

test = gen(torch.randn(1, n_z, device=device))
print(test.shape)


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.enc_in = nn.Sequential(
            OrderedDict([
                ('conv_in', nn.Conv2d(cnl, n, 3, padding=1)),
            ]))
        self.enc_block_1 = nn.Sequential(
            OrderedDict([
                ('conv_1_1', nn.Conv2d(n, n, 3, padding=1)),
                ('elu_1_1', nn.ELU()),
                ('conv_1_2', nn.Conv2d(n, 2 * n, 3, padding=1)),
                ('elu_1_2', nn.ELU()),
                ('conv_sub_1', nn.Conv2d(2 * n, 2 * n, 2, stride=2, padding=0)),
            ]))
        self.enc_block_2 = nn.Sequential(
            OrderedDict([
                ('bn_2', nn.BatchNorm2d(2*n)),
                ('conv_2_1', nn.Conv2d(2 * n, 2 * n, 3, padding=1)),
                ('elu_2_1', nn.ELU()),
                ('conv_2_2', nn.Conv2d(2 * n, 3 * n, 3, padding=1)),
                ('elu_2_2', nn.ELU()),
                ('conv_sub_2', nn.Conv2d(3 * n, 3 * n, 2, stride=2, padding=0)),
            ]))
        self.enc_block_3 = nn.Sequential(
            OrderedDict([
                ('bn_3', nn.BatchNorm2d(3*n)),
                ('conv_3_1', nn.Conv2d(3 * n, 3 * n, 3, padding=1)),
                ('elu_3_1', nn.ELU()),
                ('conv_3_2', nn.Conv2d(3 * n, 4 * n, 3, padding=1)),
                ('elu_3_2', nn.ELU()),
                ('conv_sub_3', nn.Conv2d(4 * n, 4 * n, 2, stride=2, padding=0)),
            ]))
        self.enc_block_4 = nn.Sequential(
            OrderedDict([
                ('bn_4', nn.BatchNorm2d(4*n)),
                ('conv_4_1', nn.Conv2d(4 * n, 4 * n, 3, padding=1)),
                ('elu_4_1', nn.ELU()),
                ('conv_4_2', nn.Conv2d(4 * n, 5 * n, 3, padding=1)),
                ('elu_4_2', nn.ELU()),
                ('conv_sub_4', nn.Conv2d(5 * n, 5 * n, 2, stride=2, padding=0)),
            ]))
        self.enc_block_5 = nn.Sequential(
            OrderedDict([
                ('bn_5', nn.BatchNorm2d(5*n)),
                ('conv_5_1', nn.Conv2d(5 * n, 5 * n, 3, padding=1)),
                ('elu_5_1', nn.ELU()),
                ('conv_5_2', nn.Conv2d(5 * n, 5 * n, 3, padding=1)),
                ('elu_5_2', nn.ELU()),
            ]))
        self.enc_out = nn.Sequential(
            OrderedDict([
                ('enc_out', nn.Linear(8 * 8 * 5 * n, n_z))
            ]))

    def forward(self, x):
        x = self.enc_in(x)

        x_1 = self.enc_block_1(x)

        x_2 = self.enc_block_2(x_1)

        x_3 = self.enc_block_3(x_2)

        x_4 = self.enc_block_4(x_3)

        x_5 = self.enc_block_5(x_4).view(-1, 8 * 8 * 5 * n)
        out = self.enc_out(x_5)

        return out


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(Encoder(), Decoder())

    def forward(self, x):
        return self.model(x)


dis = Discriminator().to(device)
dis.apply(weights_init_normal)

test = dis(torch.randn(16, 3, input_size, input_size, device=device))
print(test.shape)


class BEGANLoss(nn.Module):

    def forward(self, target,generator=False):
        loss = torch.mean(torch.pow(torch.abs(target - (dis(target).detach() if generator else dis(target))), eta))
        return loss


loss_func = BEGANLoss()

optimizerD = optim.Adam(dis.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))

dataset = dset.ImageFolder(root=data_folder,
                           transform=transforms.Compose([
                               transforms.Resize(input_size),
                               transforms.CenterCrop(input_size),
                               #                                      transforms.RandomVerticalFlip(p=1),
                               #                                      transforms.RandomHorizontalFlip(p=1),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=4)

fixed_noise = torch.randn(1, n_z, device=device)

convergense_track = []
gen_loss_track = []
dis_loss_track = []
iterations = 0

for epoch in range(1, epochs + 1):
    for i, data in enumerate(dataloader, 0):

        dis.zero_grad()
        gen.zero_grad()

        batch = data[0].to(device)

        #
        random_noise = torch.randn(batch.shape[0], n_z, device=device)
        d_loss_real = loss_func(batch)
        d_loss = d_loss_real - k * loss_func(gen(random_noise).detach())

        d_loss.backward()
        optimizerD.step()

        #
        random_noise = torch.randn(batch.shape[0], n_z, device=device)
        g_loss = loss_func(gen(random_noise), True)

        g_loss.backward()
        optimizerG.step()

        # Calculate convergence
        M = d_loss_real.item() + abs(gamma * d_loss_real.item() - g_loss.item())
        # Update k
        k = k + lam * (gamma * d_loss_real.item() - g_loss.item())

        iterations += batch.shape[0]

        if i % 100 == 0:
            dis_loss_track.append(d_loss.item())
            gen_loss_track.append(g_loss.item())
            convergense_track.append(M)

        if i % 1e3 == 0:
            print(
                'Epoch {} [{}/{}]Convergence: {}, G_Loss: {}, D_Loss: {}, k: {}'
                .format(epoch, iterations % samples_count, samples_count, M, g_loss.item(), d_loss.item(), k)
            )

    fake = gen(fixed_noise)
    vutils.save_image(fake.detach(),
                      '%s/fake_samples_epoch_%03d.png' % ('check_points', epoch),
                      normalize=True)
    # plt.imshow(Image.open('./check_points/fake_samples_epoch_{}.png'.format(str(epoch).zfill(3))))
    # plt.show()

    torch.save(gen.state_dict(), '%s/netG_epoch_%d.pth' % ('check_points', epoch))
    torch.save(dis.state_dict(), '%s/netD_epoch_%d.pth' % ('check_points', epoch))

# plt.imshow(Image.open('./check_points/fake_samples_epoch_002.png'))
