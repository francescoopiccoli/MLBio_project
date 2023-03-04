import os

def ensure_dir_exists(directory):
  if not os.path.exists(directory):
    try:
      os.makedirs(directory)
    except OSError:
      if not os.path.isdir(directory):
        raise
  return

def get_fn(string):
  return string.split('/')[-1].split('.')[0]