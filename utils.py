from pdb import set_trace as T
import numpy as np
import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def getParameters(paramDict):
   ret = []
   for k, v in paramDict.items():
      v = v.data.view(-1).float()
      ret.append(v)
   return torch.cat(ret)

def setParameters(ann, meanVec):
   ind = 0
   meanVec = meanVec.ravel()
   stateDict = {}
   for k, e in ann.state_dict().items():
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

#Continuous moving average
class CMA():
   def __init__(self):
      self.t = 1.0
      self.cma = None

   def update(self, x):
      if self.cma is None:
         self.cma = x
         return
      self.cma = (x + self.t*self.cma)/(self.t+1)
      self.t += 1.0

class GANLoss():
   def __init__(self, batch):
      self.dLoss = CMA()
      self.gLoss = CMA()
      self.epochs = {'D':[], 'G':[]}

   def update(self, d, g):
      self.dLoss.update(d)
      self.gLoss.update(g)

   def epoch(self):
      self.epochs['D'].append(self.dLoss.cma)
      self.epochs['G'].append(self.gLoss.cma)
      self.dLoss = CMA()
      self.gLoss = CMA()

   def __str__(self):
       return 'D: ' + str(self.dLoss.cma)[:5] + ', G: ' + str(self.gLoss.cma)[:4]
