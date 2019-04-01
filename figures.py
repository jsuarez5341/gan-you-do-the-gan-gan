from pdb import set_trace as T
import numpy as np

from matplotlib import pyplot as plt
import torch

from visualize import visualize, visGANGAN
from gan import SimpleGAN

gandir = 'data/gan/0/'
gangandir = 'data/gangan/'

fig = plt.figure(frameon=False)
#fig.set_size_inches(16, 4)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)

#GAN Figure
img = []
buf = 1+np.zeros((28*8, 2))
noise = torch.randn(64, 64).cuda()
for i in (0, 1, 10, 25, 27, 30, 32, 35, 40, 49):
   params = torch.load(gandir+'model_'+str(i)+'.pt')
   gan = SimpleGAN(28*28, zdim=64, hd=64, hg=64, lr=2e-4)
   gan.load_state_dict(params)
   gan = gan.cuda()

   ary = visualize(gan, noise, show=False)
   img.append(ary)

img1 = [img[0], buf, img[1], buf, img[2], buf, img[3], buf, img[4]]
hbuf = 1+np.zeros((2, 28*8*5+8))
img2 = [img[5], buf, img[6], buf, img[7], buf, img[8], buf, img[9]]
img = np.vstack([np.hstack(img1), hbuf, np.hstack(img2)])
ax.imshow(img)
#plt.show()
fig.savefig('gan_training.png', dpi=200, bbox_inches='tight') 

#GAN-GAN Figure
gannoise = torch.randn(40, 64).cuda()
gangannoise = torch.Tensor(np.linspace(-2, 2, 32)).cuda().view(-1, 1)

params = torch.load(gangandir+'model_249.pt')
gangan = SimpleGAN(113745, zdim=1, hd=8, hg=64, lr=2e-4).cuda()
gangan.load_state_dict(params)
gangan = gangan.cuda()

gan = gangan.sample(gangannoise)
img = visGANGAN(gan, gannoise, show=False)
ax.imshow(img)
#plt.show()
fig.savefig('gangan_sample', dpi=200, bbox_inches='tight') 





