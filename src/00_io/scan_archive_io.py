#!/root/anaconda3/bin/python3


import sys
import time
import argparse


import numpy as np
import os
import glob
import h5py


from GERecon import Archive


def get_frame_table(archive):

  ksp = []
  try:
    while True:
      ksp.append(np.array(archive.NextFrame()[..., None], dtype=np.complex64))
  except:
    None

  if (len(ksp) > 1) and (ksp[0].shape != ksp[1].shape):
    ksp.pop(0)

  ksp = np.concatenate(ksp, axis=-1)

  return ksp

def get_noise(noise_file):
  noisedata = h5py.File(noise_file, 'r')
  noise = noisedata['Data/NoiseData'][:]
  noise = noise.view('complex')
  return noise


def main(args):
  

  print("> Loading scan archive... ", end="", flush=True)
  start_time = time.perf_counter()
  archive = Archive(args.scn)
  end_time = time.perf_counter()
  print("done. Time taken: %0.2f seconds." % (end_time - start_time), flush=True)


  dct = archive.Header()

  hdr = {}
  hdr["system_name"]           = "%s / %s" % (dct["rdb_hdr_exam"]["hospname"], \
                                              dct["rdb_hdr_exam"]["ex_sysid"])
  hdr["psd_name"]              = dct["rdb_hdr_image"]["psdname"].split("/")[-1]
  hdr["exam_num"]              = int(dct["rdb_hdr_exam"]["ex_no"])
  hdr["number_of_interleaves"] = int(dct["rdb_hdr_image"]["user2"])
  hdr["in_plane_acceleration"] = int(dct["rdb_hdr_image"]["user1"])
  hdr["trj_rot_xita"]          = np.pi * float(dct["rdb_hdr_image"]["user3"])
  hdr["trj_rot_beta"]          = np.pi * float(dct["rdb_hdr_image"]["user4"])

  print("> Sequence information:")
  print(">> System:                    %s"     % hdr["system_name"])
  print(">> PSD name:                  %s"     % hdr["psd_name"])
  print(">> Exam number:               %s"     % hdr["exam_num"])
  print(">> Number of interleaves:     %d"     % hdr["number_of_interleaves"])
  print(">> In-plane acceleration:     %d"     % hdr["in_plane_acceleration"])
  print(">> Trj rotation angle (Xita): %f rad" % hdr["trj_rot_xita"])
  print(">> Trj rotation angle (Beta): %f rad" % hdr["trj_rot_beta"])


  # Saving parameters as text file.
  if args.txt is not None:
    with open(args.txt, 'w') as f:
      f.write(f'{hdr["psd_name"]}\n')
      f.write(f'{hdr["exam_num"]}\n')
      f.write(f'{hdr["number_of_interleaves"]}\n')
      f.write(f'{hdr["in_plane_acceleration"]}\n')
      

  # Saving parameters as pickle file.
  if args.pkl is not None:

    import pickle

    name = args.pkl
    if name[-4:] != ".pkl":
      name = name + ".pkl"

    with open( name, 'wb') as f:
      pickle.dump(hdr, f)


  # Parsing and saving k-space as numpy array.
  if args.ksp is not None:

    print("> Parsing frame table... ", end="", flush=True)
    start_time = time.perf_counter()
    ksp = get_frame_table(archive)
    end_time = time.perf_counter()
    print("done. Time taken: %0.2f seconds." % (end_time - start_time), flush=True)

    print("> Saving NPY... ", end="", flush=True)
    start_time = time.perf_counter()
    np.save( args.ksp, ksp)
    end_time = time.perf_counter()
    print("done. Time taken: %0.2f seconds." % (end_time - start_time), flush=True)

  # Parsing and saving noise measurement
  if args.nse is not None:
    
    if args.nsc is None:
      basePath   = os.getcwd()
      noise_path = glob.glob(os.path.join(basePath + "/" + str(hdr["exam_num"]), \
                                          "{0}*.h5".format("NoiseStatistic")))[0]
    else:
      noise_path =  args.nsc

    print("> Parsing and saving noise measurement... ", end="", flush=True)
    start_time = time.perf_counter()
    noise = get_noise(noise_path)
    np.save( args.nse, noise)
    end_time = time.perf_counter()
    print("done. Time taken: %0.2f seconds." % (end_time - start_time), flush=True)


  # Saving Dicom.
  if args.rec is not None:
    assert args.dcm is not None, "'--rec' and '--dcm' both require values."

    from GERecon import Dicom
    import pydicom as dcm
    import sigpy as sp

    dicom = Dicom(archive)
    res = np.load( args.rec)

    # Resizing to get around DICOM's inbuilt interpolation.
    R = sp.linop.Resize((256, 256, res.shape[2]), res.shape) 
    im = np.abs(sp.ifft(R(sp.fft(res))))
    im = np.transpose(im, (2,1,0))
    x = sp.to_device(im, -1)

    # Image windowing.
    x = x - np.min(x.ravel())
    x = 1e3 * (x/np.max(x.ravel()))

    # DICOM information.
    corners     = archive.Corners(0)
    orientation = archive.Orientation(0)
    zlocs       = np.arange(-res.shape[0]/2, res.shape[0]/2, 1)

    series_uid = dcm.uid.generate_uid()
    series_tag = args.tag
    if series_tag.lower() == 'offline':
      tag_num = 1
    elif series_tag.lower() == 't2flair':
      tag_num = 2
    elif series_tag.lower() == 't2cube':
      tag_num = 3
    elif series_tag.lower() == 't1cube':
      tag_num = 4
    elif series_tag.lower() == 't1mprage':
      tag_num = 5
    elif series_tag.lower() == 'dircube':
      tag_num = 6


    print("> Writing DICOM in directory: %s" % args.dcm)

    for slc_idx in range(x.shape[0]):
      write_slc_idx = x.shape[0] - slc_idx - 1

      corners["lowerLeft_z"]  = zlocs[slc_idx]
      corners["upperLeft_z"]  = zlocs[slc_idx]
      corners["upperRight_z"] = zlocs[slc_idx]


      slc = np.rot90(np.abs(x[write_slc_idx, ...].squeeze()).astype(np.float32), k=2)

      save_path = "%s/mrf_%s%05d.dcm" % (args.dcm, series_tag.lower(), write_slc_idx)
      dicom.Write(save_path, slc, write_slc_idx, corners, orientation)

      # Load DICOM into pydicom for additional manipulation.
      ds = dcm.dcmread(save_path)

      # Set parameters specific to current MRF implementation.
      ds[0x0018, 0x1100].value = 256 # Set reconstruction diameter.
      ds[0x0019, 0x101e].value = 256 # Set display FOV.
      ds[0x0018, 0x0050].value = 1   # Set slice thickness.
      ds[0x0018, 0x0088].value = 1   # Set slice spacing.
      ds[0x0028, 0x0030].value = 1   # Set pixel spacing.

      # Change patient oritentation -- affects how dicoms are rendered in DICOM viewers.
      ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]

      # sets tag of image
      series_desc = ds.SeriesDescription
      series_desc_new = ' '.join(series_desc.split())
      series_desc_new += ':%s' % series_tag
      ds.SeriesDescription = series_desc_new

      # Modify series number (i.e. add 100 to the existing series number)
      ds.SeriesNumber = str(int(ds.SeriesNumber) * 1000 + (tag_num))

      # Modify series UID
      ds.SeriesInstanceUID = series_uid

      # Save modified dicom.
      ds.save_as(save_path)


  return archive


def create_arg_parser():

  parser = argparse.ArgumentParser(description="Scan archive to .npy or .npy to DICOM (using ScanArchive for metadata).")

  # Required parameters.
  parser.add_argument('--scn', type=str, required=True,  help="Absolute path to scan archive h5 file.")

  # Optional parameters.
  parser.add_argument('--pkl', type=str, required=False, default=None, help="Absolute path to parameter dictionary as Pickle binary file.")
  parser.add_argument('--ksp', type=str, required=False, default=None, help="Absolute path to save k-space NPY file.")
  parser.add_argument('--nsc', type=str, required=False, default=None, help="Absolute path to noise scan archive h5 file.")
  parser.add_argument('--nse', type=str, required=False, default=None, help="Absolute path to save noise NPY file.")
  parser.add_argument('--rec', type=str, required=False, default=None, help="Absolute path to input reconstruction (as NPY) to save as DICOM.")
  parser.add_argument('--dcm', type=str, required=False, default=None, help="Absolute path to save DICOM file from input reconstruction (from 'rec' param).")
  parser.add_argument('--tag', type=str, required=False, default='offline', help="Tag to add to saved DICOM, e.g. MPRAGE, T2-FLAIR...")
  parser.add_argument('--txt', type=str, required=False, default=None, help="Textfile to store some header info in")

  return parser


if __name__ == '__main__':

  start_time = time.perf_counter()

  args = create_arg_parser().parse_args(sys.argv[1:])

  if ((args.rec is not None and args.dcm is None) or (args.rec is None and args.dcm is not None)):
    raise RuntimeError("'--rec' and '--dcm' both require values.")

  archive = main(args)

  end_time = time.perf_counter()
  print("> Total time: %0.2f seconds." % (end_time - start_time))
