import numpy as np
import nipype.interfaces.fsl as fsl

def preprocess_star(rec):
  '''
  rec_clean = preprocess_star(rec)

  Preprocess data for startGAN synthesis network 

  The images will be phase aligned and the imaginary part thrown away.
  
  The image will be normalized such that the maximum = 1

  FSL BET will be used to remove the background.

  Inputs:
    rec (Numpy array): the output of "long" offline recon in clinical pipeline. 
                       Shape = [256, 256, 256, 3]

  Returns:
    rec_clean (Array): Processed volume.
  '''
  x = (rec * np.exp(-1j * np.angle(rec[:, :, :, 0])[..., None])).real
  x = x / np.max(np.abs(x.ravel()))
  return np.transpose(x, [3, 0, 1, 2])
