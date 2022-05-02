import os
import sys
import time
import argparse
import numpy as np
import sigpy as sp
import sigpy.mri as mr
from distutils.util import strtobool

import mrf
import llr
import optalg

from load_data  import load_data
from autofov    import autofov
from smilr      import smilr

def main(args):

  dev = sp.Device(args.dev)
  xp = dev.xp

  mvd = lambda x: sp.to_device(x, dev)
  mvc = lambda x: sp.to_device(x, sp.cpu_device)

  print("> Loading data... ", end="", flush=True)
  start_time = time.perf_counter()
  (trj, ksp, phi) = load_data(args.trj, args.ksp, args.phi, args.rnk, \
                              bool(strtobool(args.akp)), args.ptt)
  end_time = time.perf_counter()
  print("done. Time taken: %0.2f seconds." % (end_time - start_time), \
        flush=True)

  (sx, sy, sz) = [args.mtx]*3

  print("> Dimensions: ", flush=True)
  print(">> trj:", trj.shape, flush=True)
  print(">> ksp:", ksp.shape, flush=True)
  print(">> phi:", phi.shape, flush=True)


  if args.ccm is not None:
    print("> Coil processing... ", end="", flush=True)
    start_time = time.perf_counter()
    with dev:
      ccm      = xp.load("/mnt/" + args.ccm)
      shape    = list(ksp.shape)
      shape[0] = ccm.shape[0]

      ksp     = np.reshape(ksp, (ksp.shape[0], -1))
      new_ksp = np.zeros((shape[0], ksp.shape[-1]), dtype=ksp.dtype)
      batch   = 1024
      for k in range(0, ksp.shape[-1], batch):
        new_ksp[:, k:(k + batch)] = mvc(ccm @ mvd(ksp[:, k:(k + batch)]))
      del ksp
    ksp = np.reshape(new_ksp, shape)
    end_time = time.perf_counter()
    print("done. Time taken: %0.2f seconds." % (end_time - start_time), \
          flush=True)

  if args.shf is not None:
      print("> Shifting ksp:", flush=True)
      shifts = np.load('/mnt/' + args.shf)
      (shift_x, shift_y, shift_z) = shifts
      print(f'Shifts: {shift_x}, {shift_y}, {shift_z}')
      px = np.exp(1j * 2 * np.pi * trj[0, ...] * shift_x)[None, ...]
      py = np.exp(1j * 2 * np.pi * trj[1, ...] * shift_y)[None, ...]
      pz = np.exp(1j * 2 * np.pi * trj[2, ...] * shift_z)[None, ...]
      ksp = ksp * px * py * pz
      end_time = time.perf_counter()
      print("done. Time taken: %0.2f seconds." % (end_time - start_time), flush=True)
  else:
    if args.akp == False and args.tfv is not None:
      print("> AutoFOV (with testing):", flush=True)
      (ksp, res) = autofov(ksp, trj, (sx, sy, sz), mit=args.fit, save=True, devnum=args.dev)

      print("> Saving AutoFOV test... ", end="", flush=True)
      start_time = time.perf_counter()
      np.save("/mnt/" + args.tfv, res)
      del res
      end_time = time.perf_counter()
      print("done. Time taken: %0.2f seconds." % (end_time - start_time), flush=True)
    elif args.akp == False:
      print("> AutoFOV:", flush=True)
      ksp = autofov(ksp, trj, (sx, sy, sz), mit=args.fit, save=False, devnum=args.dev)

  if args.svk is not None:
    print("> Saving processed k-space... ", end="", flush=True)
    start_time = time.perf_counter()
    np.save("/mnt/" + args.svk, ksp)
    end_time = time.perf_counter()
    print("done. Time taken: %0.2f seconds." % (end_time - start_time), flush=True)

  # Scaling trajectory
  trj[0, ...] = trj[0, ...] * sx
  trj[1, ...] = trj[1, ...] * sy
  trj[2, ...] = trj[2, ...] * sz

  # Coil calibration.
  print("> Coil Estimation:", flush=True)
  with dev:
    start_time = time.perf_counter()
    calib_ksp = ksp[   :, :64, ...].transpose((1, 2, 3, 0)).T
    calib_trj = trj[::-1, :64, ...].T
    calib_ksp = sp.to_device(calib_ksp, dev)
    mps = mr.app.JsenseRecon(calib_ksp, coord=calib_trj, \
                             device=dev, img_shape=(sz, sy, sx)).run()

    if args.nco is not None:
      mps = mps[:args.nco,...]
      ksp = ksp[:args.nco,...]
    end_time = time.perf_counter()
    print("Time taken: %0.2f seconds." % (end_time - start_time), flush=True)


  # Preparing data for full reconstruction.
  print("> Permuting data... ", flush=True, end="")
  start_time = time.perf_counter()
  trj = trj[::-1, ...].T
  ksp = np.transpose(ksp, (1, 2, 3, 0)).T
  ksp = ksp/np.linalg.norm(ksp)
  phi = phi.T
  end_time = time.perf_counter()
  print("done. Time taken: %0.2f seconds." % (end_time - start_time), flush=True)

  with dev:
    phi = sp.to_device(phi, dev)
    mps = sp.to_device(mps, dev)

    # MRF forward operator.
    A = mrf.linop(trj, phi, mps)

    # Full reconstruction.
    if args.c:
      print("> CG Reconstruction:", flush=True)
      recon = mvc(sp.app.LinearLeastSquares(A, ksp, max_iter=args.mit).run()).T

    if args.p:
      print("> Polynomial Preconditioned LLR Reconstruction:", flush=True)
      LL = args.eig if args.eig is not None else \
           sp.app.MaxEig(A.N, dtype=xp.complex64, \
                         device=sp.cpu_device).run() * 1.01
      print(">> Maximum eigenvalue estimated:", LL)
      A = np.sqrt(1/LL) * A
      proxg = llr.ProxLLR(A.ishape, args.lam, args.blk) if args.blk > 0 else \
              sp.prox.NoOp(A.ishape)
      recon = mvc(optalg.unconstrained(args.mit, A, ksp, proxg, pdeg=args.pdg)).T

    if args.s:
      print("> SMILR:", flush=True)

      print(">> Initial reconstruction:", flush=True)
      start_time = time.perf_counter()
      A_approx = mrf.linop(trj, phi[:args.srn, :], mps[:args.sco, ...])
      init  = mvc(sp.app.LinearLeastSquares(A_approx, ksp[:args.sco, ...], max_iter=args.sit).run())
      end_time = time.perf_counter()
      print("Time taken: %0.2f seconds." % (end_time - start_time), flush=True)

      print(">> Inference:", flush=True)
      start_time = time.perf_counter()
      bias = smilr(args.dev, init)
      end_time = time.perf_counter()
      print("Time taken: %0.2f seconds." % (end_time - start_time), flush=True)

      print(">> Scaling... ", flush=True, end="")
      start_time = time.perf_counter()
      scale = sp.to_device(xp.linalg.norm(ksp) / xp.linalg.norm(A * bias), sp.cpu_device)
      bias  = bias * scale
      end_time = time.perf_counter()
      print("done. Time taken: %0.2f seconds." % (end_time - start_time), flush=True)

      # Preparing weights.
      coeff_weights = [1, 3.5, 6, 6.5, 7]
      I  = sp.linop.Identity([1] + list(A.ishape[1:]))
      W  = sp.linop.Diag([elm * I     for elm in coeff_weights], iaxis=0, oaxis=0)
      iW = sp.linop.Diag([(1/elm) * I for elm in coeff_weights], iaxis=0, oaxis=0)

      # Weighting vectors.
      print(">> Refinement:", flush=True)
      start_time = time.perf_counter()
      A     = A * iW
      bias  = W(bias)
      recon = iW(sp.app.LinearLeastSquares(A, ksp, x=bias, z=bias, lamda=args.slm, max_iter=args.mit).run()).T
      end_time = time.perf_counter()
      print("Time taken: %0.2f seconds." % (end_time - start_time), flush=True)

  print("> Saving reconstruction... ", end="", flush=True)
  start_time = time.perf_counter()
  np.save("/mnt/" + args.res, recon)
  end_time = time.perf_counter()
  print("done. Time taken: %0.2f seconds." % (end_time - start_time), \
         flush=True)


def create_arg_parser():
  parser = argparse.ArgumentParser(description="MRF Reconstruction.")

  # Required parameters.
  parser.add_argument("--trj", type=str, required=True, \
    help="Trajectory.")
  parser.add_argument("--ksp", type=str, required=True, \
    help="k-space data.")
  parser.add_argument("--phi", type=str, required=True, \
    help="Temporal basis")
  parser.add_argument("--res", type=str, required=True, \
    help="Location to save the result.")

  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument('-c', action='store_true', \
    help="Conjugate Gradient Reconstruction.")
  group.add_argument('-p', action='store_true', \
    help="Polynomial Preconditioned LLR Reconstruction.")
  group.add_argument('-s', action='store_true', \
    help="SMILR Reconstruction.")

  # Optional parameters.
  parser.add_argument("--ccm", type=str, required=False, default=None, \
    help="Coil processing matrix.")
  parser.add_argument("--akp", type=str, required=False, default="True", \
    help="Assume k-space processed already.")
  parser.add_argument("--yfv", type=str, required=False, default="True", \
    help="Perform AutoFOV.")
  parser.add_argument("--tfv", type=str, required=False, default=None, \
    help="Save AutoFOV results.")
  parser.add_argument("--svk", type=str, required=False, default=None, \
    help="Save processed k-space.")
  parser.add_argument("--rnk", type=int, required=False, default=5, \
    help="Rank of temporal subspace.")
  parser.add_argument("--mtx", type=int, required=False, default=256, \
    help="Matrix size: [mtx, mtx, mtx]")
  parser.add_argument("--ptt", type=int, required=False, default=10, \
    help="Number of readout points to throw away.")
  parser.add_argument("--dev", type=int, required=False, default=0, \
    help="Device to use for reconstruction.")
  parser.add_argument("--fit", type=int, required=False, default=20, \
    help="Number of CG iterations for AutoFOV.")
  parser.add_argument("--mit", type=int, required=False, default=40,  \
    help="Number of iterations for reconstruction.")
  parser.add_argument("--nco", type=int, required=False, default=None, \
    help="Number of coils to use (after coil compression)")
  parser.add_argument("--shf", type=str, required=False, default=None,\
    help="(x, y, z) shifts in mm.")

  parser.add_argument("--pdg", type=int, required=False, default=9,    \
    help="(-p) Degree of polynomial preconditioner.")
  parser.add_argument("--blk", type=int, required=False, default=8,    \
    help="(-p) LLR block size.")
  parser.add_argument("--lam", type=float, required=False, default=5e-5, \
    help="(-p) Regularization value for LLR.")
  parser.add_argument("--eig", type=float, required=False, default=None, \
    help="(-p) Lipchitz constant, if known.")

  parser.add_argument("--srn", type=int, required=False, default=3, \
    help="(-s) Initial reconstruction rank.")
  parser.add_argument("--sco", type=int, required=False, default=3, \
    help="(-s) Initial reconstruction number of coils.")
  parser.add_argument("--sit", type=int, required=False, default=5, \
    help="(-s) Initial reconstruction number of iterations.")
  parser.add_argument("--slm", type=float, required=False, default=1e-2, \
    help="(-s) Scale regularization.")


  return parser

if __name__ == "__main__":
  start_time = time.perf_counter()
  args = create_arg_parser().parse_args(sys.argv[1:])
  main(args)
  end_time = time.perf_counter()
  print("> Total time: %0.2f seconds." % (end_time - start_time))
