from pdb import set_trace as T
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class Discriminator(nn.Module):
   def __init__(self):
      super().__init__()
      self.criterion = nn.BCELoss()

   def forward(self, x):
      shape = x.shape
      batch = shape[0]
      x = x.view(batch, -1)
      return x

   def loss(self, x, noise, G):
      D, batch = self, x.size(0)
      zero = torch.zeros(batch, 1)#.cuda()
      one = torch.ones(batch, 1)#.cuda()

      real = self.criterion(D(x), one)
      fake = self.criterion(D(G(noise)), zero)
      loss = real + fake
      return loss

class Generator(nn.Module):
   def __init__(self):
      super().__init__()
      self.criterion = nn.BCELoss()

   def loss(self, x, noise, D):
      G, batch = self, x.size(0)
      one = torch.ones(batch, 1)#.cuda()

      loss = self.criterion(D(G(noise)), one)
      return loss

class SimpleGenerator(Generator):
   def __init__(self, xdim, zdim, h):
      super().__init__()
      self.inp = nn.Linear(zdim, h)
      self.hidden = nn.Linear(h, h)
      self.out = nn.Linear(h, xdim)
      self.xdim = xdim
      self.zdim = zdim

   def forward(self, z):
      batch, x = z.size(0), z
      x = F.leaky_relu(self.inp(x), 0.2)
      x = F.leaky_relu(self.hidden(x), 0.2)
      x = torch.tanh(self.out(x))

      x = x.view(batch, self.xdim)
      return x

class SimpleDiscriminator(Discriminator):
   def __init__(self, xdim, h):
      super().__init__()
      self.inp = nn.Linear(xdim, h)
      self.hidden = nn.Linear(h, h)
      self.out = nn.Linear(h, 1)

   def forward(self, x):
      shape = x.shape
      batch = shape[0]
      x = x.view(batch, -1)

      x = F.leaky_relu(self.inp(x), 0.2)
      x = F.leaky_relu(self.hidden(x), 0.2)
      x = torch.sigmoid(self.out(x))
      return x

class DCGenerator(Generator):
    # initializers
    def __init__(self, zdim=64, h=16, mean=0.0, std=0.02):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(zdim, h*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(h*8)
        self.deconv2 = nn.ConvTranspose2d(h*8, h*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(h*4)
        self.deconv3 = nn.ConvTranspose2d(h*4, h*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(h*2)
        self.deconv4 = nn.ConvTranspose2d(h*2, 1, 4, 2, 1)
        #self.deconv4 = nn.ConvTranspose2d(h*2, h, 4, 2, 1)
        #self.deconv4_bn = nn.BatchNorm2d(h)
        #self.deconv5 = nn.ConvTranspose2d(h, 1, 4, 2, 1)
        self.weight_init(mean, std)
        self.zdim = zdim

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, z):
        batch = z.size(0)
        x = z.view(*z.shape, 1, 1)
        x = F.relu(self.deconv1_bn(self.deconv1(x)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = torch.tanh(self.deconv4(x))
        #x = F.relu(self.deconv4_bn(self.deconv4(x)))
        #x = F.tanh(self.deconv5(x))

        return x

class DCDiscriminator(Discriminator):
    # initializers
    def __init__(self, h=128, mean=0.0, std=0.02):
        super().__init__()
        self.conv1 = nn.Conv2d(1, h, 4, 2, 1)
        self.conv2 = nn.Conv2d(h, h*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(h*2)
        self.conv3 = nn.Conv2d(h*2, h*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(h*4)
        self.conv4 = nn.Conv2d(h*4, 1, 4, 2, 0)
        #self.conv4 = nn.Conv2d(h*4, h*8, 4, 2, 1)
        #self.conv4_bn = nn.BatchNorm2d(h*8)
        #self.conv5 = nn.Conv2d(h*8, 1, 4, 1, 0)
        self.weight_init(mean, std)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = torch.sigmoid(self.conv4(x))
        #x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        #x = F.sigmoid(self.conv5(x))
        return x

class GAN(nn.Module):
   def __init__(self, zdim):
      super().__init__()
      self.zdim = zdim

   def forward(self, z):
      x = self.generator(z)
      a = self.discriminator(x)
      return a

   def sample(self, z):
      x = self.generator(z)
      x = x.detach().cpu().numpy()
      if x.shape[1] == 1:
         x = x.squeeze(1)
      return x

   def noise(self, batch):
      return torch.randn(batch, self.zdim)#.cuda()

   def zero_grad(self):
      self.dOpt.zero_grad()
      self.gOpt.zero_grad()

   def train_step(self, x):
      self.zero_grad()
      gLoss = self.generator.stepG(x)
      gLoss.backward()
      self.gOpt.step()
      return dLoss + gLoss

class SimpleGAN(GAN):
   def __init__(self, xdim, zdim=64, h=256, lr=2e-4):
      super().__init__(zdim)
      self.discriminator = SimpleDiscriminator(xdim, h)
      self.generator = SimpleGenerator(xdim, zdim, h)

class DCGAN(GAN):
   def __init__(self, zdim=16, h=8, lr=2e-4):
      super().__init__(zdim)
      self.discriminator = DCDiscriminator(h)
      self.generator = DCGenerator(zdim, h)
 
