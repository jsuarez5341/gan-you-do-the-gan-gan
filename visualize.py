from matplotlib import pyplot as plt
from pdb import set_trace as T
import numpy as np

from gan import SimpleGAN
import utils

def visualize(gan, vals, sz=8, show=True):
   ary = np.empty((sz, sz), dtype=object)
   vals  = gan.sample(vals)
   i = 0
   for r in range(sz):
      for c in range(sz):
         ary[r, c] = vals[i].reshape(28, 28)
         i += 1
   ary = np.vstack([np.hstack(e) for e in ary])
   if show:
      plt.imshow(ary)
      plt.pause(0.001)
   return ary

def visGANGAN(ganParams, vals, show=True):
   imgs = vals.size(0)
   nets = ganParams.shape[0]
   ary = np.empty((nets, imgs), dtype=object)
   z = vals.cuda()
   for r in range(nets):
      gan = SimpleGAN(28*28, zdim=64, hd=64, hg=64, lr=2e-4).cuda()
      utils.setParameters(gan, ganParams[r])
      gan = gan.cuda()
      x = gan.sample(z)
      for c in range(imgs):
         ary[r, c] = x[c].reshape(28, 28)

   ary = np.vstack([np.hstack(e) for e in ary])
   if show:
      plt.imshow(ary)
      plt.pause(0.001)
   return ary

