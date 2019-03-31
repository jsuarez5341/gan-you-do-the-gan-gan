from matplotlib import pyplot as plt
from pdb import set_trace as T
import numpy as np

def visualize(gan, vals, sz=8):
   ary = np.empty((sz, sz), dtype=object)
   i = 0
   for r in range(sz):
      for c in range(sz):
         z = vals[i].view(1, -1)
         ary[r, c] = gan.sample(z).reshape(32, 32)
         i += 1
   ary = np.vstack([np.hstack(e) for e in ary])
   plt.imshow(ary)
   plt.pause(0.001)
   return ary

def visGANGAN(gan, vals, sz=8):
   ary = np.empty((sz, sz), dtype=object)
   i = 0
   for r in range(sz):
      for c in range(sz):
         z = vals[i].view(1, -1)
         x = gan.sample(z)[0]
         ary[r, c] = x#[r*sz+c]
         i += 1
   ary = np.vstack([np.hstack(e) for e in ary])
   plt.imshow(ary)
   plt.pause(0.001)
   return ary

