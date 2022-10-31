import os
import re
import shutil
from tqdm import tqdm

source = '../../resource/getchu_avatar'
target = '../../resource/avatar.list'
if __name__ == "__main__":
    get_dir = os.listdir(source)
    with open(target,'w') as file:
        for i in tqdm(get_dir):
            sub_dir = os.path.join(source,i)
            file.write("{} {}\n".format(i,sub_dir))
    