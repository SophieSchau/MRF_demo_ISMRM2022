import sys
import time
import argparse
from distutils.util import strtobool

import numpy as np
import sigpy as sp

from rovir import rovir


def get_gre_ksp(raw):
  ksp = np.reshape(raw, (raw.shape[0], raw.shape[1], 64, 64)) # After reshape: (kx, nc, ky, z)
  ksp = np.transpose(ksp, (0, 2, 3, 1))
  tmp = np.zeros_like(ksp)
  tmp[:, :, 0::2, :] = ksp[:, :, :32, :]
  tmp[:, :, 1::2, :] = ksp[:, :, 32:, :]
  img = sp.ifft(tmp, axes=(0, 1))
  img = np.transpose(img, (2, 1, 0, 3))
  img = img[:, ::-1, ::-1, :]
  return sp.fft(img, axes=(0, 1, 2))


def main(args):

  ## Load data.
  raw = np.load(args.ksp)
  nc  = np.shape(raw)[1]

  if raw.shape[0] != 64 or raw.shape[2] != 4096:
    raise Exception("Currently, only GRE calibration supported.")
  print("Assumed GRE parameters:", flush=True)
  print("\tMatrix size:  64 x  64 x  64", flush=True)
  print("\tFOV:         440 x 440 x 440 mm^3", flush=True)

  ksp = get_gre_ksp(raw)
  img = sp.ifft(ksp, axes=(0, 1, 2))
  mtx_shape = list(img.shape[:3])

  # Load noise matrix.
  nmat = np.load(args.nse)

  if args.idx is not None:
    idx = np.load(args.idx)
    nmat = nmat[idx,...]
    nc = len(idx)
    ksp = ksp[..., idx]
    img = img[..., idx]

  #################################### ROVIR ####################################
  # Link: https://onlinelibrary.wiley.com/doi/epdf/10.1002/mrm.28706
  # Based on Section 2.3.
  #
  # 1. Pre-whiten k-space.
  # 2. Calculate A & B
  # 3. Calculate generalized eigenvalue decomposition.
  # 4. Select number of virtual coils.
  # 5. Normalize generalized eigenvectors to have unit l2-norm.
  # 6. Ortho-normalize chosen coils => virtual channels are whitened.

  # 1. Whitening.
  covm = (nmat @ nmat.conj().T)/(nmat.shape[1] - 1)
  wht  = np.linalg.pinv(np.linalg.cholesky(covm)) # Whitening matrix.
  img  = np.reshape((wht @ np.reshape(img, (-1, nc)).T).T, mtx_shape + [nc])

  # 2. Calculate A and B of ROVir.
  roi = {}
  roi["x"] = np.arange(16, 48)
  roi["y"] = np.arange(16, 48)
  roi["z"] = np.arange(16, 48)
  rtf = None

  # 3. Calculate generalized eigenvalue decomposition.
  # 4. Select number of virtual coils.
  eig = rovir(img, roi, rtf)[:, :args.nrc]
  # 5. Normalize to unit l2-norm.
  for k in range(args.nrc):
    eig[:, k] = eig[:, k]/np.linalg.norm(eig[:, k])
  # 6. Ortho-normalize.
  (rcc, _, _) = np.linalg.svd(eig, full_matrices=False)

  # Coil processing matrix so far.
  ccm = rcc.T @ wht

  # Apply processing to k-space, and take SVD decomposition.
  ksp_shape = list(ksp.shape)
  ksp_shape[-1] = args.nsv
  ksp = ccm @ np.reshape(ksp.T, (nc, -1))

  (u, _, _) = np.linalg.svd(ksp, full_matrices=False)
  u  = u[:, :args.nsv]
  uH = u.conj().T

  ksp = np.reshape((uH @ ksp).T, ksp_shape)
  ccm = uH @ ccm

  img = sp.ifft(ksp, axes=(0, 1, 2))

  ## Save results
  if args.rci is not None:
    np.save(args.rci, img)
  np.save(args.ccm, ccm)


def create_arg_parser():
  parser = argparse.ArgumentParser(description="Calibration.")

  # Required parameters.
  parser.add_argument("--ksp", type=str, required=True,  \
    help="GRE k-space data.")
  parser.add_argument("--nse", type=str, required=True,  \
    help="GRE noise measurements.")
  parser.add_argument("--ccm", type=str, required=True, \
    help="Location to save coil processing matrix.")

  # Optional arguments.
  parser.add_argument("--rci", type=str, required=False, \
    default=None, help="Location to save ROVir coil images.")
  parser.add_argument("--nrc", type=int, required=False, \
    default=40, help="Number of ROVir coils.")
  parser.add_argument("--nsv", type=int, required=False, \
    default=4, help="Number of SVD coils.")
  parser.add_argument("--idx", type=str, required=False, \
    default=None, help="Indecies of coils to use.")

  return parser


if __name__ == "__main__":
  start_time = time.perf_counter()
  args = create_arg_parser().parse_args(sys.argv[1:])
  main(args)
  end_time = time.perf_counter()
  print("> Total time: %0.2f seconds." % (end_time - start_time))
