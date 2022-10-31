__author__ = 'Wendong Xu'
import argparse
import utils
import logging
from PIL import Image
import os


def get_image_path(raw_input_dir: str) -> list:
  """
  get image path and id from root resource path.
  :return: a list contains all images' path.
  """
  result = []
  for root, dirs, files in os.walk(raw_input_dir):
    for file in files:
      result.append(os.path.join(root, file))
  return result

def remove_below_specified_resolution(raw_img_dir: str, resolution: tuple) -> None:
  """
  remove the image which is smaller than specified resolution.
  :param raw_img_dir:
  :return:
  """
  img_paths = get_image_path(raw_img_dir)
  for path in img_paths:
    size = Image.open(path).size
    if size[0] < resolution[0] or size[0] < resolution[1]:
      os.remove(path)
      print('{} doesn\'t satisfied. deleted.'.format(path))


if __name__ == '__main__':
  logging.basicConfig(filename=None, level=logging.INFO, format='%(levelname)s:%(message)s',
                      datefmt='%d-%m-%Y %I:%M:%S %p')
  logging.getLogger().addHandler(logging.StreamHandler())

  parser = argparse.ArgumentParser()
  parser.add_argument("--raw_img_dir", type=str,
                      default="../../../resource/getchu_raw_img/",
                      help='''''')
  parser.add_argument("--resolution", type=tuple,
                      default=(101, 101),
                      help='''''')
  FLAGS, unparsed = parser.parse_known_args()
  raw_img_dir = FLAGS.raw_img_dir
  resolution = FLAGS.resolution

  remove_below_specified_resolution(raw_img_dir, resolution)
