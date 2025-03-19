"""
Reorganize the celebA dataset to:
root/attr_obj/img1.jpg
root/attr_obj/img2.jpg
root/attr_obj/img3.jpg
...

obj -> target class
attr -> spurious attribute
"""

import pandas as pd
import os
from itertools import product
import shutil
from tqdm import tqdm
import torch

DATA_FOLDER= "data"

old_root = DATA_FOLDER + '/waterbird_complete95_forest2water2'
new_root = DATA_FOLDER + '/waterbirds'

objs = ['landbird', 'waterbird']
attrs = ['land', 'water']
pairs = list(product(attrs, objs))
sets = ['train', 'val', 'test']

os.mkdir(new_root)
for p in pairs:
    os.makedirs(os.path.join(new_root, 'images', '_'.join(p)), exist_ok=True)

data = pd.read_csv(os.path.join(old_root, 'metadata.csv')).to_dict('records')
new_data = []
for instance in tqdm(data, 'Reorganizing waterbirds '):
    filename = instance['img_filename']
    attr = attrs[instance['place']]
    obj = objs[instance['y']]
    set = sets[instance['split']]

    file_path = os.path.join(old_root, filename)
    new_dir = os.path.join(new_root, 'images', f'{attr}_{obj}')
    shutil.move(file_path, new_dir)
    new_data.append({
        'image': f"{attr}_{obj}/{filename.split('/')[1]}",
        'attr': attr,
        'obj': obj,
        '_image': filename,
        'set': set
    })

split_name = 'compositional-split-natural'
torch.save(new_data, os.path.join(new_root, f'metadata_{split_name}.t7'))
os.makedirs(os.path.join(new_root, split_name), exist_ok=True)
for s in sets:
    with open(os.path.join(new_root, split_name, f'{s}_pairs.txt'), 'w+') as file:
        file.writelines([a + ' ' + o + '\n' for a, o in pairs])
