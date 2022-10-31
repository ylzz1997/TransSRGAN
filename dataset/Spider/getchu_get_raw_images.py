__author__ = 'Wendong Xu'
import argparse
import re
import logging
import os
from pyquery import PyQuery
import requests
import urllib
from PIL import Image
from io import BytesIO
import multiprocessing as mp


__DEBUG__ = False
URL_PREFIX = 'http://www\.getchu\.com'
URL_PATTERN = re.compile('<.+=".*(/brandnew/(\d+)/c\d+chara.*?)"')
proxies = {
  'https': 'https://127.0.0.1:10809',
  'http': 'http://127.0.0.1:10809'
}
headers = {
  'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36',
  'host':'www.getchu.com'
}

def get_html(url: str,url_p:str = None) -> str:
  """
  Get the html file using proxy in order to access the demo server
  over 'GFW' using your shadowsocks.
  :param url:
  :return: the html source code
  """
  print('Now processing: {}'.format(url))
  opener = urllib.request.build_opener(urllib.request.ProxyHandler(proxies))

  urllib.request.install_opener(opener)
  if url_p!=None: 
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36',
        'host':'www.getchu.com',
        'referer':url_p
    }
  else:
    headers = {
      'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36',
      'host':'www.getchu.com'
    }
  req = urllib.request.Request(url, headers=headers)
  try:
    response = urllib.request.urlopen(req)
    return response.read()
  except:
    return None


def get_img(url_p:str,img_url: str, save_path: str) -> None:
  """
  Get charater's images using proxy.
  :param img_url:
  :param save_path:
  :return:
  """
  img_str = get_html(img_url,url_p)
  if img_str==None: return
  print("save:{}".format(img_url))
  image = Image.open(BytesIO(bytes(img_str)))
  print("差不多SAVE了")
  image.save(save_path)
  print("这下可SAVE了")


def get_img_urls(html, save_dir: str,url_p: str) -> None:
  """
  Get image urls.
  :param html:
  :param save_dir:
  :return:
  """
  if html==None: return
  doc = PyQuery(html)
  chara_img_urls = re.findall(URL_PATTERN, doc.html())
  if len(chara_img_urls) == 0:
    return
  print("imglen:{}".format(len(chara_img_urls)))
  for i in range(0, len(chara_img_urls)):
    url, file_name = str(URL_PREFIX + chara_img_urls[i][0]).replace('\\', ''), chara_img_urls[i][1] + ('_{}.jpg'.format(i))
    print('{} {}'.format(url, file_name))
    print('host: {}'.format(url_p))   
    get_img(url_p,url, save_dir + file_name)


def main():
  logging.basicConfig(filename=None, level=logging.INFO, format='%(levelname)s:%(message)s',
                      datefmt='%d-%m-%Y %I:%M:%S %p')
  logging.getLogger().addHandler(logging.StreamHandler())

  parser = argparse.ArgumentParser()
  parser.add_argument("--input_path", type=str,
                      default="resource/getchu_urls.txt",
                      help='''''')

  parser.add_argument("--output_dir", type=str,
                      default="resource/getchu_raw_img/",
                      help='''''')

  parser.add_argument("--temp_dir", type=str,
                      default="resource/getchu_raw_img/",
                      help='''''')
  FLAGS, unparsed = parser.parse_known_args()
  input_path = FLAGS.input_path
  output_dir = FLAGS.output_dir
  temp_dir = FLAGS.temp_dir
  logging.info('--input_path: {}'.format(os.path.abspath(FLAGS.input_path)))
  logging.info('--output_path : {}'.format(os.path.abspath(FLAGS.output_dir)))
  logging.info('--temp_path : {}'.format(os.path.abspath(FLAGS.temp_dir)))

  if not os.path.exists(output_dir):
    os.mkdir(output_dir)
  if not os.path.exists(temp_dir):
    os.mkdir(temp_dir)

  print(input_path)

  with open(input_path) as fin:
    urls = fin.readlines()
    pool = mp.Pool(max(mp.cpu_count() - 2, 1))
    for url in urls:
      url = url.replace("\n","")
      pool.apply_async(get_img_urls, (get_html(url).decode('EUC-JP', 'ignore'), output_dir,url))
      # get_img_urls(get_html(url).decode('EUC-JP', 'ignore'), output_dir,url)
      if __DEBUG__:
        break
    pool.close()
    pool.join()

if __name__ == '__main__':
  main()
