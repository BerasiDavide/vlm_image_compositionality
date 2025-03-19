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

old_root = DATA_FOLDER + '/celeba-dataset'
new_root = DATA_FOLDER + '/celebA'

objs = ['blonde', 'dark']
attrs = ['female', 'male']
pairs = list(product(attrs, objs))
sets = ['train', 'val', 'test']

os.mkdir(new_root)
for p in pairs:
    os.makedirs(os.path.join(new_root, 'images', '_'.join(p)), exist_ok=True)

data = pd.merge(
    pd.read_csv(os.path.join(old_root, 'list_attr_celeba.csv')),
    pd.read_csv(os.path.join(old_root, 'list_eval_partition.csv')),
    on='image_id')

data[['Blond_Hair', 'Male', 'partition']] = data[['Blond_Hair', 'Male', 'partition']].astype(int)
data = data.to_dict('records')

new_data = []
for instance in tqdm(data, 'Reorganizing celebA '):
    filename = instance['image_id']
    obj = 'blonde' if instance['Blond_Hair']==1 else 'dark'
    attr = 'male' if instance['Male']==1 else 'female'
    set = sets[instance['partition']]

    file_path = os.path.join(old_root, 'img_align_celeba', 'img_align_celeba', filename)
    new_dir = os.path.join(new_root, 'images', f'{attr}_{obj}')
    shutil.move(file_path, new_dir)
    new_data.append({
        'image': f"{attr}_{obj}/{filename}",
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
