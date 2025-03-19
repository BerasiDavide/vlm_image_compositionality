from itertools import product
import torch
import os
import torch.nn.functional as F
from collections import defaultdict

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import (CenterCrop, Compose, InterpolationMode,
                                    Normalize, RandomHorizontalFlip,
                                    RandomPerspective, RandomRotation, Resize,
                                    ToTensor)
from torchvision.transforms.transforms import RandomResizedCrop

BICUBIC = InterpolationMode.BICUBIC
n_px = 224


def transform_image(split="train", imagenet=False):
    if imagenet:
        # from czsl repo.
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        transform = Compose(
            [
                RandomResizedCrop(n_px),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(
                    mean,
                    std,
                ),
            ]
        )
        return transform

    if split == "test" or split == "val":
        transform = Compose(
            [
                Resize(n_px, interpolation=BICUBIC),
                CenterCrop(n_px),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
    else:
        transform = Compose(
            [
                # RandomResizedCrop(n_px, interpolation=BICUBIC),
                Resize(n_px, interpolation=BICUBIC),
                CenterCrop(n_px),
                RandomHorizontalFlip(),
                RandomPerspective(),
                RandomRotation(degrees=5),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    return transform

class ImageLoader:
    def __init__(self, root):
        self.img_dir = root

    def __call__(self, img):
        file = '%s/%s' % (self.img_dir, img)
        img = Image.open(file).convert('RGB')
        return img


class CompositionDataset(Dataset):
    def __init__(
            self,
            root,
            phase,
            split='compositional-split-natural',
            open_world=False,
            transform=None
    ):
        # General attributes
        self.root = root
        self.phase = phase
        self.split = split
        self.open_world = open_world

        self.feat_dim = None
        self.transform = transform
        self.loader = ImageLoader(self.root + '/images/')

        self.attrs, self.objs, self.pairs, \
                self.train_pairs, self.val_pairs, \
                self.test_pairs = self.parse_split()
        
        self.full_pairs = list(product(self.attrs, self.objs))
        if self.open_world:
            self.pairs = self.full_pairs
        
        # phase-specific attributes
        self.train_data, self.val_data, self.test_data = self.get_split_info()
        if self.phase == 'train':
            self.data = self.train_data
        elif self.phase == 'val':
            self.data = self.val_data
        elif self.phase == 'test':
            self.data = self.test_data
        else: # get all data
            self.data = self.train_data + self.val_data + self.test_data
        _, self.all_attrs, self.all_objs = zip(*self.data)
        self.all_pairs = list(zip(self.all_attrs, self.all_objs))

        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}
        self.attr2idx = {attr: idx for idx, attr in enumerate(self.attrs)}
        self.pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}

        self.train_pair_to_idx = dict(
            [(pair, idx) for idx, pair in enumerate(self.train_pairs)]
        )

        # Some potentially usefull info 
        seen_pairs = set(self.train_pairs)
        self.seen_mask = torch.BoolTensor(
            [pair in seen_pairs for pair in self.pairs]
            )

        self.objs_by_attr= {k: [] for k in self.attrs}
        self.attrs_by_obj = {k: [] for k in self.objs}
        for (a, o) in self.all_pairs:
            self.objs_by_attr[a].append(o)
            self.attrs_by_obj[o].append(a)

    def get_split_info(self):
        data = torch.load(self.root + '/metadata_{}.t7'.format(self.split), weights_only=False)
        train_data, val_data, test_data = [], [], []
        pairs = set(self.pairs)
        for instance in data:
            image, attr, obj, settype = instance['image'], instance[
                'attr'], instance['obj'], instance['set']

            if attr == 'NA' or (attr,
                                obj) not in pairs or settype == 'NA':
                # ignore instances with unlabeled attributes
                # ignore instances that are not in current split
                continue

            data_i = [image, attr, obj]
            if settype == 'train':
                train_data.append(data_i)
            elif settype == 'val':
                val_data.append(data_i)
            else:
                test_data.append(data_i)

        return train_data, val_data, test_data

    def parse_split(self):
        def parse_pairs(pair_list):
            with open(pair_list, 'r') as f:
                pairs = f.read().strip().split('\n')
                # pairs = [t.split() if not '_' in t else t.split('_') for t in pairs]
                pairs = [t.split() for t in pairs]
                pairs = list(map(tuple, pairs))
            attrs, objs = zip(*pairs)
            return attrs, objs, pairs

        tr_attrs, tr_objs, tr_pairs = parse_pairs(
            '%s/%s/train_pairs.txt' % (self.root, self.split))
        vl_attrs, vl_objs, vl_pairs = parse_pairs(
            '%s/%s/val_pairs.txt' % (self.root, self.split))
        ts_attrs, ts_objs, ts_pairs = parse_pairs(
            '%s/%s/test_pairs.txt' % (self.root, self.split))

        all_attrs, all_objs = sorted(
            list(set(tr_attrs + vl_attrs + ts_attrs))), sorted(
                list(set(tr_objs + vl_objs + ts_objs)))
        all_pairs = sorted(list(set(tr_pairs + vl_pairs + ts_pairs)))

        return all_attrs, all_objs, all_pairs, tr_pairs, vl_pairs, ts_pairs

    def __getitem__(self, index):
        image, attr, obj = self.data[index]
        img = self.loader(image)
        if self.transform is not None:
            img = self.transform(img)

        if self.phase == 'train':
            data = [
                img, self.attr2idx[attr], self.obj2idx[obj], self.train_pair_to_idx[(attr, obj)]
            ]
        else:
            data = [
                img, self.attr2idx[attr], self.obj2idx[obj], self.pair2idx[(attr, obj)]
            ]

        return data

    def __len__(self):
        return len(self.data)
    
    def __str__(self) -> str:
        n_seen_val = len(set(self.train_pairs) & set(self.val_pairs))
        n_seen_test = len(set(self.train_pairs) & set(self.test_pairs))
        descr_pairs = ' # train pairs : {:<7} | # val pairs : {:<7} ({:^5} seen) | # test pairs : {:<7} ({:^5} seen)'.format(
            len(self.train_pairs),
            len(self.val_pairs), n_seen_val,
            len(self.test_pairs), n_seen_test)
        
        _, attr_val, obj_val = zip(*self.val_data)
        all_val_pairs = zip(attr_val, obj_val)
        seen_pairs = set(self.train_pairs)
        n_seen_img_val = sum(p in seen_pairs for p in all_val_pairs)
        _, attr_test, obj_test = zip(*self.test_data)
        all_test_pairs = zip(attr_test, obj_test)
        n_seen_img_test = sum(p in seen_pairs for p in all_test_pairs)
        descr_n_img = ' # train images: {:<7} | # val images: {:<7} ({:^5} seen) | # test images: {:<7} ({:^5} seen)'.format(
            len(self.train_data),
            len(self.val_data), n_seen_img_val,
            len(self.test_data), n_seen_img_test)
        descr_ao = ' # attrs : {:<10} # objs : {:<10} # full pairs {:<10}'.format(
            len(self.attrs), len(self.objs), len(self.full_pairs))
        return descr_pairs + '\n' + descr_n_img + '\n' + descr_ao



class CompositionDatasetEmbeddings(CompositionDataset):
    '''
    Class for loading pre-computed embeddings of a compositional dataset.
    '''
    def __init__(
            self,
            root,
            phase,
            split='compositional-split-natural',
            open_world=False,
            model_architecture: str='ViT-L-14',
            model_pretraining: str='openai',
            normalize: bool=True,
            ):
        super().__init__(root, phase, split, open_world)
        loadfile_id = f"emb_{model_architecture}_{model_pretraining}.pt"
        self.text_embs_path = os.path.join(root, 'TEXT' + loadfile_id)
        self.image_embs_path = os.path.join(root, 'IMG' + loadfile_id)
        self.normalize = normalize
    
    def load_text_embs(self, pairs):
        '''
        Loads text embeddings for given pairs.
        '''
        text_embs_data = torch.load(self.text_embs_path, weights_only=False)  # All text embeddings
        pair2idx = defaultdict(list)
        for i, pair in enumerate(text_embs_data['pairs']):
            pair2idx[pair].append(i)

        indices = [i for pair in pairs
                     for i in pair2idx[pair]]
        text_embs = text_embs_data['embeddings'][indices]

        if self.normalize:
            text_embs = F.normalize(text_embs, p=2, dim=-1)
        
        if len(pairs)==len(text_embs):
            return text_embs
        else:
            all_pairs = [text_embs_data['pairs'][i] for i in indices]
            return text_embs, all_pairs

    def load_all_image_embs(self):
        '''
        Loads all image embeddings for `self.phase`.
        '''
        image_embs_data = torch.load(self.image_embs_path, weights_only=False)  # All image embeddings
        image_id2idx = {id: i for i, id in enumerate(image_embs_data['image_ids'])}

        all_image_id, all_attrs, all_objs = zip(*self.data)
        all_pairs = list(zip(all_attrs, all_objs))
        indices = [image_id2idx[id] for id in all_image_id]
        image_embs = image_embs_data['embeddings'][indices]

        if self.normalize:
            image_embs = F.normalize(image_embs, p=2, dim=-1)

        return image_embs, all_pairs