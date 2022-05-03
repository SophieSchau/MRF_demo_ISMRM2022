import os
import sys
import time
import argparse

import numpy as np

from preprocess import preprocess_star
from synthesize import synthesize_star

def main(args):

  print("> Loading data... ", end="", flush=True)
  start_time = time.perf_counter()
  recon = np.load(args.inp)
  end_time = time.perf_counter()
  print("done. Time taken: %0.2f seconds." % (end_time - start_time), \
        flush=True)

  print("> Preprocessing... ", end="", flush=True)
  start_time = time.perf_counter()
  recon = preprocess_star(recon)
  end_time = time.perf_counter()
  print("done. Time taken: %0.2f seconds." % (end_time - start_time), \
        flush=True)

  print("> Synthesizing contrasts... ", end="", flush=True)
  start_time = time.perf_counter()
  synth = synthesize_star(recon, args.idx, args.wgt)
  synth = np.transpose(synth, (0,2,3,1))
  end_time = time.perf_counter()
  print("done. Time taken: %0.2f seconds." % (end_time - start_time), \
        flush=True)

  print("> Saving synthesized image.. ", end="", flush=True)
  start_time = time.perf_counter()
  contrast_names = ["t1w", "t1mprage", "t2w", "t2flair", "dir"]
  for k in range(len(args.idx)):
    filename = contrast_names[args.idx[k]]
    np.save(args.sth + '/' + filename, synth[k, ...])
  end_time = time.perf_counter()
  print("done. Time taken: %0.2f seconds." % (end_time - start_time), \
         flush=True)


def create_arg_parser():
  parser = argparse.ArgumentParser(description="MRF Synthesis.")

  # Required parameters.
  parser.add_argument("--inp", type=str, required=True, \
    help="MRF reconstruction as NIFTI file.")
  parser.add_argument("--wgt", type=str, required=True, \
    help="Path to stored network weights.")
  parser.add_argument("--sth", type=str, required=True, \
    help="Path to save synthesized images.")

  # Optional parameters.
  parser.add_argument("--idx", type=int, nargs="+", required=False, default=[0], \
    help="Index for contrasts to synthesize.")

  return parser

if __name__ == "__main__":
  start_time = time.perf_counter()
  args = create_arg_parser().parse_args(sys.argv[1:])
  main(args)
  end_time = time.perf_counter()
  print("> Total time: %0.2f seconds." % (end_time - start_time))
