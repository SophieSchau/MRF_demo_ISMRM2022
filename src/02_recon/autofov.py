import time
import numpy as np
import sigpy as sp
import sigpy.mri as mr


def gaussian(n, sig):
  x = np.arange(-int(n/2), -int(n/2) + n, 1)
  return np.exp(-0.5 * (x/sig)**2)


def autofov(ksp, trj, mtx_size, scale=1.25, mit=20, std=24, bins=16384,
            p=0.125, penalty=5, save=False, devnum=sp.cpu_device):

  (sx, sy, sz) = mtx_size
  (SX, SY, SZ) = [int(scale * elm) for elm in mtx_size]
  N = max(int(ksp.shape[1]/2), 128)

  dev = sp.Device(devnum)
  xp  = dev.xp

  calib_ksp = np.copy(ksp[:, :N, ...]).transpose((1, 2, 3, 0)).T

  calib_trj = np.copy(trj[:, :N, ...])
  calib_trj[0, ...] = calib_trj[0, ...] * SX
  calib_trj[1, ...] = calib_trj[1, ...] * SY
  calib_trj[2, ...] = calib_trj[2, ...] * SZ
  calib_trj = calib_trj[::-1, ...].T

  with dev:
    calib_trj = sp.to_device(calib_trj, dev)
    calib_ksp = sp.to_device(calib_ksp, dev)

    print(">> Low resolution high FOV reconstruction... ", \
          end="", flush=True)
    start_time = time.perf_counter()
    mps = mr.app.JsenseRecon(calib_ksp, coord=calib_trj, \
                             device=dev, img_shape=(SZ, SY, SX), \
                             show_pbar=False).run()
    x = mr.app.SenseRecon(calib_ksp, mps, coord=calib_trj, device=dev, \
                              max_iter=mit, coil_batch_size=1,
                              show_pbar=False).run().T
    del mps
    end_time = time.perf_counter()
    print("done. Time taken: %0.2f seconds." % (end_time - start_time), \
          flush=True)

    print(">> Binary center-of-mass processing... ", \
          end="", flush=True)
    start_time = time.perf_counter()
    X = sp.fft(x, axes=(0, 1, 2)) 
    h = gaussian(SX, std)[:, None, None] * \
        gaussian(SY, std)[None, :, None] * \
        gaussian(SZ, std)[None, None, :]
    Y = X * sp.to_device(h, dev)
    y = sp.ifft(Y, axes=(0, 1, 2))

    del Y
    del X
    del h

    z = np.abs(y)
    z = z - np.min(z.ravel())
    z = sp.to_device(z/np.max(z.ravel()), sp.cpu_device)
    del y

  (hist, bins) = np.histogram(z.ravel(), bins=bins, range=(0, 1))
  hist = hist/np.linalg.norm(hist, ord=1)
  bins = (bins[1:] + bins[:-1])/2

  cdf = np.cumsum(hist) # cumulative distribution
  thresh = bins[np.argwhere(cdf > 1 - p).min()]
  binary_image = z > thresh

  x_axis = np.arange(-int(SX/2), SX - int(SX/2), 1)
  y_axis = np.arange(-int(SY/2), SY - int(SY/2), 1)
  z_axis = np.arange(-int(SZ/2), SZ - int(SZ/2), 1)

  # Outside FOV is "really far away" from the center.
  x_axis[np.abs(x_axis) >= int(sx/2)] = penalty * x_axis[np.abs(x_axis) >= int(sx/2)]
  y_axis[np.abs(y_axis) >= int(sy/2)] = penalty * y_axis[np.abs(y_axis) >= int(sy/2)]
  z_axis[np.abs(z_axis) >= int(sz/2)] = penalty * z_axis[np.abs(z_axis) >= int(sz/2)]

  # Calculate shifts.
  area = np.sum(binary_image.ravel())
  dx = int(np.round(np.sum((x_axis[:, None, None] * binary_image).ravel())/area))
  dy = int(np.round(np.sum((y_axis[None, :, None] * binary_image).ravel())/area))
  dz = int(np.round(np.sum((z_axis[None, None, :] * binary_image).ravel())/area))

  end_time = time.perf_counter()
  print("done. Time taken: %0.2f seconds." % (end_time - start_time), \
        flush=True)
  print(">> Calculated shifts: (%d, %d, %d)" % (dx, dy, dz), flush=True)

  print(">> Applying shifts... ", end="", flush=True)
  start_time = time.perf_counter()
  px = np.exp(1j * 2 * np.pi * trj[0, ...] * dx)[None, ...]
  py = np.exp(1j * 2 * np.pi * trj[1, ...] * dy)[None, ...]
  pz = np.exp(1j * 2 * np.pi * trj[2, ...] * dz)[None, ...]

  ksp = sp.to_device(ksp, sp.cpu_device) * px * py * pz
  end_time = time.perf_counter()
  print("done. Time taken: %0.2f seconds." % (end_time - start_time), \
        flush=True)

  if save is True:
    print(">> Re-doing reconstruction for testing... ", end="", flush=True)
    start_time = time.perf_counter()

    calib_ksp = np.copy(ksp[:, :N, ...]).transpose((1, 2, 3, 0)).T
    with dev:
      calib_ksp = sp.to_device(calib_ksp, dev)
      mps = mr.app.JsenseRecon(calib_ksp, coord=calib_trj, \
                               device=dev, img_shape=(SZ, SY, SX), \
                               show_pbar=False).run()
      z = mr.app.SenseRecon(calib_ksp, mps, coord=calib_trj, device=dev, \
                            max_iter=mit, coil_batch_size=1,
                            show_pbar=False).run().T
      x = sp.to_device(x, sp.cpu_device)[..., None]
      z = sp.to_device(z, sp.cpu_device)[..., None]
      res = np.concatenate((x, z), axis=-1)

    end_time = time.perf_counter()
    print("done. Time taken: %0.2f seconds." % (end_time - start_time), \
           flush=True)

  if save:
    return (ksp, res)
  return ksp
