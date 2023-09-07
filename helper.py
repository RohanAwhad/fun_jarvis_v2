import bz2
import pickle

def save_data(data_tuple, fn):
  bin_data = pickle.dumps(data_tuple)
  cmprsd_bin_data = bz2.compress(bin_data)
  logger.debug(f'- orginal size: {len(bin_data) / 1e6} MB')
  logger.debug(f'- compressed size: {len(cmprsd_bin_data) / 1e6} MB')
  with open(fn, 'wb') as f: f.write(cmprsd_bin_data)

def load_data(fn):
  with open(fn, 'rb') as f: cmprsd_bin_data = f.read()
  bin_data = bz2.decompress(cmprsd_bin_data)
  return pickle.loads(bin_data)
