from pdb import set_trace as T
import numpy as np

import torch, torchvision
from torchvision import transforms
from torch.utils.data.dataset import Dataset, TensorDataset

import utils

def shuffle(t):
   idx = torch.randperm(t.size(0))
   return t[idx]

def MNIST(train=True, batch=128, sz=32):
   trans = transforms.Compose(
         [transforms.Resize(sz),
         transforms.ToTensor()] )
   #TODO: add normalize 2*(x-0.5)
   data = torchvision.datasets.MNIST(root='./data',
         train=train, download=True, transform=trans)
   loader = torch.utils.data.DataLoader(data, batch_size=batch,
         shuffle=True, num_workers=2)
   return loader

def GANData(n, epochs, datadir, flatten=True):
   params = {}
   for i in range(n):
      modeldir = datadir + str(i) + '/'
      modeldir = datadir + '0/'
      params[i] = {}
      for e in range(epochs):
         f = modeldir + 'model_' + str(e) + '.pt'
         model = torch.load(f, map_location='cpu')
         params[i][e] = utils.getParameters(model)

   if not flatten:
      return params

   rets = []
   for idx, snapshot in params.items():
      for epoch, vec in snapshot.items():
         #rets.append(vec)
         rets.append(params[0][epoch])
   rets = torch.stack(rets)
   return shuffle(rets)

def GANLoader(n, epochs, datadir, batch=32, flatten=True):
   dat = GANData(n, epochs, datadir, flatten).cpu()
   data = TensorDataset(dat)
   #data = GANData(n, epochs, flatten)
   return torch.utils.data.DataLoader(data, batch_size=batch,
         shuffle=True, num_workers=2)

'''
class GANData(Dataset):
   def __init__(self, n, epochs, flatten=True):
       super().__init__()
       self.data = loadData(n, epochs, flatten)

   def __getitem__(self, index):
       return self.data[index]

   def __len__(self):
       return self.data.size(0)
'''
