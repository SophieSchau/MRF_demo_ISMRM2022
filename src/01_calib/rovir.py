"""
ROVir Coil Compression.

@author: myurt@stanford.edu
@author: ssi@stanford.edu
"""

import numpy as np

def rovir(gre, roi, rtf=None):
  """
  ROVir Coil Compression.

  Inputs:
    gre: Dimensions: (sx, sy, sz, nc).
    roi: Region of interest.
    rtf: Region of interference.

  Returns:
    u: Dimensions (nc, nc).
       Coil compression matrix.
  """
  (sx, sy, sz, nc) = gre.shape

  # Compute A.
  # TODO: Move to not using loops.
  A = np.zeros((nc, nc), dtype=np.complex64)
  for i_x in roi["x"]:
    for i_y in roi["y"]:
      for i_z in roi["z"]:
        g_xyz = gre[i_x, i_y, i_z,:]
        g_xyz_1 = np.expand_dims(np.conjugate(g_xyz), 1)
        g_xyz_2 = np.expand_dims(g_xyz, 0)
        A = A + np.matmul(g_xyz_1, g_xyz_2)

  # Compute B.
  # TODO: Move to not using loops.
  B = np.zeros((nc, nc), dtype=np.complex64)
  if rtf is None:
    print("Estimating interference from ROI.")
    for i_x in range(sx):
      for i_y in range(sy):
        for i_z in range(sz):
          if (i_x not in roi["x"]) or \
             (i_y not in roi["y"]) or \
             (i_z not in roi["z"]):
            g_xyz = gre[i_x, i_y, i_z, :]
            g_xyz_1 = np.expand_dims(np.conjugate(g_xyz), 1)
            g_xyz_2 = np.expand_dims(g_xyz,0)
            B = B + np.matmul(g_xyz_1,g_xyz_2)
  else:
    print("Using specified interference.")
    for i_x in rtf["x"]:
      for i_y in rtf["y"]:
        for i_z in rtf["z"]:
          g_xyz = gre[i_x, i_y, i_z, :]
          g_xyz_1 = np.expand_dims(np.conjugate(g_xyz), 1)
          g_xyz_2 = np.expand_dims(g_xyz,0)
          B = B + np.matmul(g_xyz_1,g_xyz_2)

  # Eigenvalue decomposition
  cor = np.matmul(np.linalg.pinv(B), A)
  (eigvals, eigvecs) = np.linalg.eig(cor)
  descending_indices = np.argsort(eigvals)[::-1]
  eigvecs = eigvecs[:, descending_indices]
  return eigvecs
