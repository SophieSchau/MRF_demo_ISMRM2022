import numpy as np
import sigpy as sp
import time
import optpoly

from tqdm.auto import tqdm

def unconstrained(num_iters, A, b, proxg, pdeg=None,
                  norm="l_2", l=0, verbose=True):

  device = sp.get_device(b)
  xp = device.xp

  P = sp.linop.Identity(A.ishape) if pdeg is None else  \
      optpoly.create_polynomial_preconditioner(pdeg, A.N, l, 1, \
                                               norm=norm, verbose=verbose)

  with device:

    AHb = A.H(b)
    x = AHb.copy()
    z = x.copy()

    lst_time  = []
    calc_tol = -1
    if verbose:
      pbar = tqdm(total=num_iters, desc="Unconstrained Optimization", \
                  leave=True)
    for k in range(1, num_iters + 1):
      start_time = time.perf_counter()
        
      x_old = x.copy()
      x = z.copy()
        
      gr = A.N(x) - AHb
      x = proxg(1, x - P(gr))
            
      # DOI: 10.1007/s10957-015-0746-4
      step = (k - 1)/(k + 4)

      z = x + step * (x - x_old)

      end_time = time.perf_counter()
      lst_time.append(end_time - start_time)

      pbar.update()
      pbar.refresh()

    if verbose:
      pbar.close()

    return x
