from pdb import set_trace as T
import numpy as np

import torch
from matplotlib import pyplot as plt

from gangan import GANGAN, getParameters
from gan import DCGan as GAN
import main

def setParameters(ann, meanVec):
   ind = 0
   meanVec = meanVec.ravel()
   stateDict = {}
   for k, e in ann.state_dict().items():
      #if e.data.dtype != torch.float32:
      #   continue
      shape = e.size()
      nParams = e.numel()
      assert e.data.dtype in (torch.float32, torch.long)
      if len(shape) != 0:
         ary = np.array(meanVec[ind:ind+nParams]).reshape(*shape)
         ary = torch.Tensor(ary)
         if e.data.dtype == torch.float32:
            e.data = ary.float()
         elif e.data.dtype == torch.long:
            e.data = ary.long()
      else:
         ary = meanVec[ind]
      stateDict[k] = e
      ind += nParams
   ann.load_state_dict(stateDict)
   
def loadModel():
   params = torch.load('data/gangan/model.pt')
   gangan = GANGAN()
   gangan.load_state_dict(params)
   gangan = gangan.cuda()

   gan = GAN().cuda()
   return gan, gangan

def visualize(gan, vals, sz=8):
   ary = np.empty((sz, sz), dtype=object)
   i = 0
   #vals = np.linspace(-1, 1, sz)
   for r in range(sz):
      for c in range(sz):
         z = vals[i].view(1, -1)
         #z = torch.Tensor([vals[r], vals[c]]).view(1, -1).cuda()
         x = gan.sample(z)[0]
         ary[r, c] = x#[r*sz+c]
         i += 1
   ary = np.vstack([np.hstack(e) for e in ary])
   plt.imshow(ary)
   plt.pause(0.001)
   return ary

def evaluate(gan, gangan):
   zsamples = np.linspace(-5, 5, 25)
   noise = gan.noise(64)
   plt.ion()
   plt.show()
   for z in zsamples:
      #z = torch.Tensor([z]).cuda()
      z = gangan.noise(1)
      ganParams = gangan.sample(z)
      model = torch.load('data/model0/model_24.pt')
      model = getParameters(model).cpu().data.numpy().reshape(1, -1)
      setParameters(gan, ganParams)
      gan = gan.cuda()
      frame = visualize(gan, noise)
      T()


if __name__ == '__main__':
   gan, gangan = loadModel()
   evaluate(gan, gangan)

