from pdb import set_trace as T
import numpy as np
import os

import torch
from torch import nn
from torch.nn import functional as F

import gan as GAN
import main

def train(gan, loader, epochs=1000, lr=2e-4):
   D, G = gan.discriminator, gan.generator
   dOpt = torch.optim.Adam(D.parameters(), lr)
   gOpt = torch.optim.Adam(G.parameters(), lr)
   batch = loader.batch_size
   loss = main.GANLoss(batch)
   datadir = 'data/gangan/'

   for epoch in range(epochs):
      for x in loader:
         x = x[0] 
         if x.size(0) < batch:
             continue
         x = x.cuda()

         noise = gan.noise(batch)
         gLoss = G.loss(x, noise, D)
         dOpt.zero_grad()
         gOpt.zero_grad()
         gLoss.backward()
         gOpt.step()

         noise = gan.noise(batch)
         dLoss = D.loss(x, noise, G)
         dOpt.zero_grad()
         gOpt.zero_grad()
         dLoss.backward()
         dOpt.step()

         loss.update(float(dLoss), float(gLoss))
      print('Epoch: ' + str(epoch) + ', ' + str(loss))
      loss.epoch()
      np.save(datadir+'loss.npy', loss.epochs)
      torch.save(gan.state_dict(), datadir+'model.pt')

if __name__ == '__main__':
   print('Loading data...')
   batch = 32
   loader = GANLoader(45, 25, batch)
   print('Loaded.')

   model = GANGAN().cuda()
   train(model, loader)

