from PIL import Image
from concurrent.futures import ProcessPoolExecutor
import os
from tqdm import tqdm
import math

def process_image(filename):
    filepath = os.path.join(img_root, 'images', filename)
    image = Image.open(filepath).convert('RGB')
    w, h = image.size

    aspect_ratio = round(h / w, 2)
    max_pixels = 1024**2

    w_prime_max = int(math.floor(math.sqrt(max_pixels / aspect_ratio))) // 32 * 32
    h_prime_max = int(math.floor(aspect_ratio * w_prime_max)) // 32 * 32

    image_resize = image.resize((w_prime_max, h_prime_max))
    aspect_ratio_str = str(round(h_prime_max / w_prime_max, 2))

    output_dir = os.path.join(img_root, 'bucket', aspect_ratio_str)
    os.makedirs(output_dir, exist_ok=True)
    image_resize.save(os.path.join(output_dir, filename))

img_root = '/home/gkalstn000/dataset/connectbrick'
file_list = os.listdir(os.path.join(img_root, 'images'))

# 멀티 프로세싱을 사용하여 이미지 처리
with ProcessPoolExecutor() as executor:
    list(tqdm(executor.map(process_image, file_list), total=len(file_list)))


import matplotlib.pyplot as plt

def image_show(pil_image) :
    plt.imshow(pil_image)
    plt.show()