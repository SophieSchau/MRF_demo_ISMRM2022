import numpy as np
import sigpy as sp
import sigpy.mri as mr

def linop(trj, phi, mps):

  dev = sp.get_device(mps)

  F = sp.linop.NUFFT(mps.shape[1:], trj)
  outer_A = []
  for k in range(mps.shape[0]):
    S = sp.linop.Multiply(mps.shape[1:], mps[k, ...]) * \
        sp.linop.Reshape( mps.shape[1:], [1] + list(mps.shape[1:]))
    lst_A = [sp.linop.Reshape([1] + list(F.oshape), F.oshape)   * \
             sp.linop.Multiply(F.oshape, phi[k, :, None, None]) * \
             F * S for k in range(phi.shape[0])]
    inner_A = sp.linop.Hstack(lst_A, axis=0)
    D1 = sp.linop.ToDevice(inner_A.ishape, dev, sp.cpu_device)
    D2 = sp.linop.ToDevice(inner_A.oshape, sp.cpu_device, dev)
    outer_A.append(D2 * inner_A * D1) 
  A = sp.linop.Vstack(outer_A, axis=0)

  return A
