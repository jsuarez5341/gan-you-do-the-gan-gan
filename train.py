from pdb import set_trace as T
from matplotlib import pyplot as plt

import numpy as np
import scipy.stats as stats
import skvideo.io
import sys, os

import torch
from torch import nn

from visualize import visualize, visGANGAN
from gan import SimpleGAN, DCGAN
import data, utils

class GANTrainer:
   def __init__(self, gan, loader, datadir, lr=2e-4): 
       D, G = gan.discriminator, gan.generator
       self.dOpt = torch.optim.Adam(D.parameters(), lr)
       self.gOpt = torch.optim.Adam(G.parameters(), lr)

       self.gan, self.loader = gan, loader
       self.batch = loader.batch_size
       self.datadir = datadir

       self.noise = gan.noise(self.batch)
       self.loss = utils.GANLoss(self.batch)

       plt.ion()
       plt.show()

   def save(self, epoch):
       loss, datadir = self.loss, self.datadir
       print('Epoch: ' + str(epoch) + ', ' + str(loss))
       loss.epoch()
       np.save(datadir+'loss.npy', loss.epochs)
       torch.save(self.gan.state_dict(), datadir+'model_'+str(epoch)+'.pt')

   def step(self, x):
      gan, dOpt, gOpt = self.gan, self.dOpt, self.gOpt
      D, G = gan.discriminator, gan.generator

      noise = gan.noise(self.batch)
      gLoss = G.loss(x, noise, D)
      dOpt.zero_grad()
      gOpt.zero_grad()
      gLoss.backward()
      gOpt.step()

      noise = gan.noise(self.batch)
      dLoss = D.loss(x, noise, G)
      dOpt.zero_grad()
      gOpt.zero_grad()
      dLoss.backward()
      dOpt.step()

      return dLoss, gLoss

class MNISTTrainer(GANTrainer):
   def __init__(self, gan, loader, datadir, lr=2e-4): 
       super().__init__(gan, loader, datadir, lr)
       self.writer = skvideo.io.FFmpegWriter(datadir + 'demo.mp4',
               inputdict={'-r':'5'})

   def train(self, epochs=25):
      for epoch in range(epochs):
          for x, _ in self.loader:
             if x.size(0) < self.batch:
                 continue
             x = x.cuda()
             x = 2*(x - 0.5)

             dLoss, gLoss = self.step(x)
          self.loss.update(float(dLoss), float(gLoss))
          self.save(epoch)
      self.writer.close()

   def save(self, epoch):
      frame = visualize(self.gan, self.noise)
      super().save(epoch)
      self.writer.writeFrame(frame)

class GANGANTrainer(GANTrainer):
   def __init__(self, gan, loader, lr=2e-4): 
      super().__init__(gan, loader, lr)
      self.noise = torch.randn(32, 64)

   def save(self, epoch):
      super().save(epoch)
      z = self.noise[0:1, :]
      z = np.linspace(-2, 2, 16)
      z = torch.Tensor(z).cuda().view(-1, 1)
      ganParams = self.gan.sample(z)
      frame = visGANGAN(ganParams, self.noise)

   def train(self, epochs=25):
      for epoch in range(epochs):
          for x in self.loader:
             x = x[0]
             if x.size(0) < self.batch:
                 continue
             x = x.cuda()

             dLoss, gLoss = self.step(x)
          self.loss.update(float(dLoss), float(gLoss))
          self.save(epoch)

def trainGANs(n=100, datadir='data/gan/'):
    for i in range(11, n):
       try:
          os.mkdir(datadir+str(i))
       except FileExistsError:
          pass
       loader = data.MNIST(batch=128)
       model = SimpleGAN(28*28, zdim=64, hd=64, hg=64, lr=2e-4).cuda()
       print('Network: ' + str(i) + ', Params: ' + str(utils.count_parameters(model)))
       #model = DCGAN(zdim=16, h=4, lr=2e-4).cuda()
       trainer = MNISTTrainer(model, loader, datadir+str(i)+'/')
       trainer.train(epochs=100)

def trainGANGAN(loaddir='data/gan/', savedir='data/gangan/'):
   print('Loading data...')
   loader = data.GANLoader(35, 100, loaddir)
   print('Loaded.')
   model = SimpleGAN(113745, zdim=1, hd=8, hg=64, lr=2e-4).cuda()
   trainer = GANGANTrainer(model, loader, savedir)
   trainer.train(epochs=250)

if __name__ == '__main__':
    assert len(sys.argv) == 2
    exp = sys.argv[1]

    cuda = torch.cuda.is_available()
    print('Found CUDA? :: ', cuda)

    if exp == 'gan':
       trainGANs()
    elif exp == 'gangan':
       trainGANGAN()
    else:
       'Specify a network to train (gan, gangan)'
       exit(0)




