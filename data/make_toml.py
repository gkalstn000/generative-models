import toml
import os
import json
root = '/home/gkalstn000/dataset/base_funetuning'
bucket_root = os.path.join(root, 'bucket')


config = {
    "general": {
        "shuffle_caption": False,
        "keep_tokens": 1,
    },
    "datasets": [
        {
            "resolution": 1024,
            "batch_size": 2,
            'subsets': []
        }
    ]
}

# "subsets": [
#     {
#         "image_dir": "C:\\piyo",
#         "metadata_file": "C:\\piyo\\piyo_md.json"
#     }
# ]
# TOML 파일로 저장


image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}

with open(os.path.join(root, 'metafile.json'), 'r') as file:
    data = json.load(file)


for dirpath, dirnames, filenames in os.walk(bucket_root):
    json_filename = dirpath.split('/')[-1] + '.json'
    jsonfile = {}
    for filename in filenames:
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            file_path = os.path.join(dirpath, filename)
            if data[filename]['caption'] == '' : continue
            jsonfile[os.path.join(dirpath, filename)] = data[filename]

    if len(jsonfile) > 0 :
        with open(os.path.join(dirpath, json_filename), 'w') as file:
            json.dump(jsonfile, file, indent=4)

        subsets = {'image_dir': dirpath,
                    'metadata_file': os.path.join(dirpath, json_filename)}

        config['datasets'][0]['subsets'].append(subsets)


with open(os.path.join('/home/gkalstn000/sd-scripts/configs', 'data.toml'), 'w') as toml_file:
    toml.dump(config, toml_file)