from PIL import Image
import os
from collections import defaultdict
from tqdm import tqdm
import math
import matplotlib.pyplot as plt

img_root = '/home/gkalstn000/dataset/connectbrick'
file_list = os.listdir(os.path.join(img_root, 'images'))
bucket = defaultdict(list)
os.makedirs(os.path.join(img_root, 'bucket'), exist_ok=True)

for filename in tqdm(file_list):
    filepath = os.path.join(img_root, 'images', filename)
    image = Image.open(filepath).convert('RGB')
    w, h = image.size

    aspect_ratio = round(h / w, 2)
    max_pixels = 1024**2

    w_prime_max = int(math.floor(math.sqrt(max_pixels / aspect_ratio)))
    h_prime_max = int(math.floor(aspect_ratio * w_prime_max))

    image_resize = image.resize((w_prime_max, h_prime_max))
    aspect_ratio = str(round(h_prime_max / w_prime_max, 2))

    os.makedirs(os.path.join(img_root, 'bucket', aspect_ratio), exist_ok=True)
    image_resize.save(os.path.join(img_root, 'bucket', aspect_ratio, filename))



def image_show(pil_image) :
    plt.imshow(image)
    plt.show()