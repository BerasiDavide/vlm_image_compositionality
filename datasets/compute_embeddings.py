import torch
import os
import argparse
import json

from open_clip import create_model_and_transforms, tokenizer
from tqdm import tqdm
from itertools import product

from datasets.read_datasets import DATASET_PATHS
from datasets.composition_dataset import CompositionDataset


def chunks(l, n):
    '''
    Yield successive n-sized chunks from l.
    '''
    for i in range(0, len(l), n):
        yield l[i:i + n]


def get_class_prompts_debiasing(dataset_name):
    # Prompts are the same used in https://arxiv.org/abs/2302.00070 (see https://github.com/chingyaoc/debias_vl/blob/main/discriminative/main.py)
    """
    Zero-Shot Prompts
    """
    if dataset_name == 'waterbirds':
        class_prompt = ['This is a picture of a landbird.', 'This is a picture of a waterbird.']
        objs = ['landbird', 'waterbird']
    elif dataset_name == 'celebA':
        class_prompt = ['A photo of a celebrity with blond hair.', 'A photo of a celebrity with dark hair.']
        objs = ['blonde', 'dark']
    return class_prompt, objs


def get_pair_prompts_debiasing(dataset_name):
    # Prompts are the same used in https://arxiv.org/abs/2302.00070 (see https://github.com/chingyaoc/debias_vl/blob/main/discriminative/main.py)
    if dataset_name == 'waterbirds':
        class_prompt = ['This is a picture of a landbird.', 'This is a picture of a waterbird.']
        class_label = ['landbird','waterbird']
        
        spurious_prompt = ['This is a land background.', 'This is a picture of a forest.',
                    'This is a picture of a moutain.', 'This is a picture of a wood.',
                    'This is a water background.', 'This is a picture of an ocean.',
                    'This is a picture of a beach.', 'This is a picture of a port.']
        spurious_label = ['land']*4 + ['water']*4

    elif dataset_name == 'celebA':
        class_prompt = ['A photo of a celebrity with blond hair.', 'A photo of a celebrity with dark hair.']
        class_label = ['blonde', 'dark']
        spurious_prompt = ['A photo of a female.', 'A photo of a female celebrity.', 'A photo of a woman.',
                           'A photo of a male.', 'A photo of a male celebrity.', 'A photo of a man.']
        spurious_label = ['female']*3  + ['male']*3

    prompt = [s + ' ' + c for s, c in product(spurious_prompt, class_prompt)]
    pair = list(product(spurious_label, class_label))
    return prompt, pair


def get_text_prompts(dataset, dataset_name):
    if dataset_name in {'waterbirds', 'celebA'}:
        class_prompt, objs = get_class_prompts_debiasing(dataset_name)
        pair_prompts, pair_pairs = get_pair_prompts_debiasing(dataset_name)
        prompts = class_prompt + pair_prompts
        pairs = [(None, o) for o in objs] + pair_pairs
        template = 'same used in https://arxiv.org/abs/2302.00070'
    else:
        def prompt_template(attr, obj):
            attr = attr.replace(".", " ").lower()
            obj = obj.replace(".", " ").lower()
            return f"an image of a {attr} {obj}"

        pairs = dataset.full_pairs
        prompts = [prompt_template(a, o) for a, o in pairs]
        template = prompt_template('ATTR', 'OBJ')
    
    return prompts, pairs, template


def compute_text_embeddings(dataset: CompositionDataset, dataset_name, model, tokenizer, device, output_file):
    '''
    Computes and saves on disk the embeddings of text prompts for all (attr, obj) pairs in the dataset.
    The embeddings are organized in a tensor of shape (n_attr, n_objs, embedding_dim)
    '''
    output_path = os.path.join(dataset.root, output_file)
    if os.path.exists(output_path):
        print(f"The file {output_path} already exists. Delete or rename it to recompute embeddings.")
        return

    # Define text prompts
    prompts, pairs, template = get_text_prompts(dataset, dataset_name)

    # Compute embeddings
    prompts_chunks = chunks(prompts, 128)
    text_embeddings = []
    with torch.no_grad():
        for text in tqdm(prompts_chunks,
                         total=len(prompts)//128+1,
                         desc='Computing text embeddings '):
            text_tokens = tokenizer.tokenize(text)
            text_emb = model.encode_text(
                text_tokens.to(device), normalize=False
                ).float()

            text_embeddings.append(text_emb.cpu())

    text_embeddings = torch.cat(text_embeddings)

    # Save embeddings on disk
    torch.save(
        {'embeddings': text_embeddings,
         'pairs': pairs,
         'prompt template': template},
        output_path)
    print(f"\nStored {len(pairs)} text embeddings in {output_path} .")


def compute_image_embeddings(dataset: CompositionDataset, model, preprocess, device, output_file):
    '''
    Computes and saves on disk the embeddings of all jpg images in the dataset.
    '''
    output_path = os.path.join(dataset.root, output_file)
    if os.path.exists(output_path):
        print(f"The file {output_path} already exists. Delate or rename it to recompute embeddings.")
        return

    # Retrieve all images paths
    #img_paths = glob(os.path.join(dataset.root, 'images', '**', '*.jpg'), recursive=True)
    all_images, all_attrs, all_objs = zip(*dataset.data)
    all_pairs = zip(all_attrs, all_objs)

    image_embeddings = []
    with torch.no_grad():
        for img_chunk in tqdm(chunks(all_images, 64), total=len(all_images)//64+1,
                                desc=f'Computing image embeddings '):
            imgs = list(map(dataset.loader, img_chunk))
            imgs = list(map(preprocess, imgs))
            imgs = torch.stack(imgs, 0)
            imgs_emb = model.encode_image(imgs.to(device),
                                          normalize=False).float()
            image_embeddings.append(imgs_emb)
    image_embeddings = torch.cat(image_embeddings, 0).to('cpu')
    torch.save({'image_ids': all_images,
                'embeddings': image_embeddings,
                'pairs': all_pairs},
                output_path)

    print(f"\nStored {len(all_images)} image embeddings in {output_path} .")   


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-nI", "--no_image",
                        help="do not compute image embeddings",
                        action="store_true")
    parser.add_argument("-nT", "--no_text",
                        help="do not compute text embeddings",
                        action="store_true")
    parser.add_argument("dataset",
                        help="dataset name")
    parser.add_argument("model_architecture",
                        help="architecture of CLIP image encoder")
    parser.add_argument("model_pretraining",
                        help="pretrainig set")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    

    # Instantiate model
    model, _, preprocess = create_model_and_transforms(
        model_name=args.model_architecture,
        pretrained=args.model_pretraining,
        device=device,
        quick_gelu=True
        )
    model.eval()

    # Instantiate dataset
    dataset_path = DATASET_PATHS[args.dataset]
    dataset = CompositionDataset(dataset_path,
                                 phase='all',
                                 transform=preprocess) # =transform_image('test', imagenet=False)
    
    # Compute embeddings
    outputfile_id = f"emb_{args.model_architecture}_{args.model_pretraining}.pt"
    if not args.no_text:
        output_file = 'TEXT' + outputfile_id
        compute_text_embeddings(dataset, args.dataset, model, tokenizer, device, output_file)

    if not args.no_image:
        output_file = 'IMG' + outputfile_id
        compute_image_embeddings(dataset, model, preprocess, device, output_file)