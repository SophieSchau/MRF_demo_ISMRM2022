import numpy as np
from scipy.io import loadmat

def check_values(arr):
  assert np.sum(np.isnan(np.abs(arr.ravel()))) == 0, \
    ">>>>> Unexpected nan in array."
  assert np.sum(np.isinf(np.abs(arr.ravel()))) == 0, \
    ">>>>> Unexpected inf in array."

def load_data(trj_file: str, ksp_file: str, phi_file: str, phi_rank: int, ptt: int):
  '''
  (trj, ksp, phi) = load_data(trj_file, ksp_file, phi_file, phi_rank):

  Load data to return NumPy arrays. This function uses the trajectory
  dimensions to remove rewinder points from the loaded k-space array.
  
  The trj within the 'mat' file pointed to by 'trj_file' is expected to
  be of the form trj[a, b, c, d]:
    a -> Readout dimension (data acquired during a spiral) NOT including
         rewinder points.
    b -> Non-cartesian coordinates of the bth volumetric dimension. b
         here varies from 0, 1 and 2.
    c -> Interleave dimension.
    d -> TR dimension.

  The ksp dimensions within the 'npy' file pointed to by 'ksp_file' is
  expected to be of the form ksp[e, f, g]:
    e -> Readout dimension (data acquired during a spiral) including
         rewinder points.
    f -> Coil dimension.
    g -> Combined TR and Interleave dimension (that is, c * d from trj).
         The data is expected to be ordered as ksp[:, :, 0:c] is the first
         TR, ksp[:, :, c:2*c] is the next and so on.
    
  The phi within the 'mat' file pointed to by 'phi_file' is expected to
  be of the form phi[d, h]
    h -> Number of subspace vectors.

  Inputs:
    trj_file (String): Path to trajectory as a MATLAB 'mat' file. Expects
                       an entry denoted 'k_3d' within the MATLAB file.
    ksp_file (String): Path to k-space file as a NumPy array.
    phi_file (String): Path to subspace basis as a MATLAB 'mat' file.
                       Expects an entry denoted 'phi' within the MATLAB
                       file.
    phi_rank (Int): Rank of subspace to use for reconstruction. If bigger
                    than h, use h.

  Returns:
    trj (Array): Non-cartesian k-space coordinates.
    ksp (Array): Acquired k-space data.
    phi (Array): Temporal subspace.
  '''
  trj = loadmat(trj_file)['k_3d'].transpose((1, 0, 2, 3))
  check_values(trj)
  assert np.abs(trj.ravel()).max() < 0.5, \
    "Trajectory must be scaled between -1/2 and 1/2."

  ksp = np.load(ksp_file, mmap_mode='r')


  # Remove rewinder points
  num_points = trj.shape[1]
  ksp = np.transpose(ksp[:num_points, ...], (1, 0, 2))

  # Split interleaves and time points
  ksp = np.reshape(ksp, (ksp.shape[0], ksp.shape[1], \
                          trj.shape[2], trj.shape[3]))
  ksp = ksp[:, ptt:, ...]

  trj = trj[:, ptt:, ...]
  phi = loadmat(phi_file)['phi'][:, :phi_rank]

  check_values(phi)
  check_values(ksp)
  check_values(phi)

  return (trj, ksp, phi)
