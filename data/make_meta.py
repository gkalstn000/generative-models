import os
import json
from glob import glob
from collections import defaultdict
from PIL import Image

def find_image_paths(root_dir):
    image_paths = []
    # 이미지 파일로 간주할 확장자 목록

    return image_paths

root = '/home/gkalstn000/dataset/base_funetuning'

meta_file = os.path.join(root, 'meta.json')
bucket_root = os.path.join(root, 'bucket')

info_dict = defaultdict(dict)

with open(meta_file, 'r') as file:
    data = json.load(file)

for filename, caption in zip(data['images'], data['captions']):
    info_dict[filename]['caption'] = caption


image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}

for dirpath, dirnames, filenames in os.walk(bucket_root):
    for filename in filenames:
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            file_path = os.path.join(dirpath, filename)
            image = Image.open(file_path)
            resolution = list(image.size[::-1])

            info_dict[filename]['train_resolution'] = resolution
            info_dict[filename]['path'] = file_path


with open(os.path.join(root, 'metafile.json'), 'w') as file:
    json.dump(info_dict, file, indent=4)