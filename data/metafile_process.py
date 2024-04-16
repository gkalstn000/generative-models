import json
import os

root = '/home/gkalstn000/dataset/connectbrick'
metafile = os.path.join(root, 'meta.json')

with open(metafile, 'r') as file :
    data = json.load(file)

metalist = [[filename, caption] for filename, caption in zip(data['images'], data['captions'])]
metalist = sorted(metalist, key=lambda x: len(x[1].split()), reverse=True)


def save_meta(metalist) :
    jsonfile = {'images' : [],
                'captions': []}

    for filename, caption in metalist :
        jsonfile['images'].append(filename)
        jsonfile['captions'].append(caption)

    with open(metafile, 'w') as json_file:
        json.dump(jsonfile, json_file)

    print('metafile saved')