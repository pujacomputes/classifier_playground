import torch
import numpy as np
import skimage
from torchvision import transforms
class RandMixFFTDataset(torch.utils.data.Dataset):
  """Dataset wrapper to perform PixMix."""

  def __init__(self, dataset, preprocess):
    self.dataset = dataset
    self.preprocess = preprocess
    self.counter = 0
    self.rand_crop = transforms.RandomCrop(32, padding=4)
  def __getitem__(self, i):
    x, y = self.dataset[i]
    rnd_idx = np.random.choice(len(self.dataset))
    rand_img,_ = self.dataset[rnd_idx]
    return self.pixmix_fft(x, rand_img), y

  def __len__(self):
    return len(self.dataset)
  
  def low_freq_mutate(self, amp_src, amp_trg, L=0.05 ):
      _, h, w = amp_src.size()
      b = (  np.floor(np.amin((h,w))*L)  ).astype(int)     # get b
      assert b > 0 
      amp_src[:,0:b,0:b]     = amp_trg[:,0:b,0:b]      # top left
      amp_src[:,0:b,w-b:w]   = amp_trg[:,0:b,w-b:w]    # top right
      amp_src[:,h-b:h,0:b]   = amp_trg[:,h-b:h,0:b]    # bottom left
      amp_src[:,h-b:h,w-b:w] = amp_trg[:,h-b:h,w-b:w]  # bottom right
      return amp_src

  def pixmix_fft(self,orig, rand_img,k=4):
      tensorize, normalize = self.preprocess['tensorize'], self.preprocess['normalize']
      orig_fft = torch.fft.fft2(tensorize(orig))
      rand_fft = torch.fft.fft2(tensorize(rand_img))
      orig_amp, orig_phase = torch.abs(orig_fft), torch.angle(orig_fft)
      rand_amp, rand_phase = torch.abs(rand_fft), torch.angle(rand_fft)
      
      #new_amp = self.low_freq_mutate(amp_src=orig_amp,amp_trg=rand_amp,L=0.1) 
      new_amp = self.low_freq_mutate(amp_src=rand_amp,amp_trg=orig_amp,L=0.4)
      new_fft = torch.polar(abs=new_amp,angle=orig_phase)   
      mixed = torch.abs(torch.fft.ifft2(new_fft))
      #jmixed_fractal = mixed / mixed.max()
      mixed_fractal = torch.clip(mixed,0,1) #/ mixed.max()
      """
      Having mixed now do the random crop!
      """
      orig = self.rand_crop(orig)
      orig_aug = tensorize(augment_input(orig))
      mixed_op = np.random.choice(utils.mixings)
      mixed = mixed_op(mixed_fractal, orig_aug, 3)
      mixed = torch.clip(mixed, 0, 1) # mixed.max()
      
      return normalize(mixed)