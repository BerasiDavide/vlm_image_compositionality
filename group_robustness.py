import argparse
import json
import os
import torch
import numpy as np
import random
from tqdm import tqdm
from math import exp
from collections import defaultdict
from itertools import product
from open_clip import create_model_and_transforms, tokenizer
import torch.nn.functional as F

from datasets.read_datasets import DATASET_PATHS
from datasets.composition_dataset import CompositionDataset, CompositionDatasetEmbeddings
from utils.utils import set_seed, chunks
from factorizers import FACTORIZERS


def summarize_acc(correct_by_groups, total_by_groups, 
                stdout=False, return_groups=True):
    all_correct = 0
    all_total = 0
    min_acc = 101.
    min_correct_total = [None, None]
    groups_accs = np.zeros([len(correct_by_groups), 
                            len(correct_by_groups[-1])])
    if stdout:
        print('Accuracies by groups:')
    for yix, y_group in enumerate(correct_by_groups):
        for aix, a_group in enumerate(y_group):
            acc = a_group / total_by_groups[yix][aix] * 100
            groups_accs[yix][aix] = acc
            # Don't report min accuracy if there's no group datapoints
            if acc < min_acc and total_by_groups[yix][aix] > 0:
                min_acc = acc
                min_correct_total[0] = a_group
                min_correct_total[1] = total_by_groups[yix][aix]
            if stdout:
                print(
                    f'{yix}, {aix}  acc: {int(a_group):5d} / {int(total_by_groups[yix][aix]):5d} = {a_group / total_by_groups[yix][aix] * 100:>7.3f}')
            all_correct += a_group
            all_total += total_by_groups[yix][aix]
    if stdout:
        average_str = f'Average acc: {int(all_correct):5d} / {int(all_total):5d} = {100 * all_correct / all_total:>7.3f}'
        robust_str = f'Robust  acc: {int(min_correct_total[0]):5d} / {int(min_correct_total[1]):5d} = {min_acc:>7.3f}'
        print('-' * len(average_str))
        print(average_str)
        print(robust_str)
        print('-' * len(average_str))
        
    avg_acc = all_correct / all_total * 100
        
    if return_groups:
        return avg_acc, min_acc, groups_accs
    return avg_acc, min_acc 


def evaluate_clip(preds, targets_t, targets_s, verbose=False):
    """
    General method for classification validation
    Args:
    - clip_predictions (np.array): predictions
    - dataloader (torch.utils.data.DataLoader): (unshuffled) dataloader
    """
    correct_by_groups = np.zeros([len(np.unique(targets_t)),
                                  len(np.unique(targets_s))])
    auroc_by_groups = np.zeros([len(np.unique(targets_t)),
                                len(np.unique(targets_s))])
    total_by_groups = np.zeros(correct_by_groups.shape)
    losses_by_groups = np.zeros(correct_by_groups.shape)

    correct = (preds == targets_t)
    
    for ix, y in enumerate(targets_t):
        s = targets_s[ix]
        correct_by_groups[int(y)][int(s)] += correct[ix].item()
        total_by_groups[int(y)][int(s)] += 1
        
    avg_acc, robust_acc, groups_accs = summarize_acc(correct_by_groups,
                                        total_by_groups,
                                        stdout=False)
    
    result = {
        'avg_acc': avg_acc,
        'robust_acc': robust_acc,
        'gap': avg_acc - robust_acc,
        'groups_accs': groups_accs.flatten().tolist()
    }
    return result


def select_n_embs_per_pair(embeddings, all_pairs, n: int):
    '''Randomly selects up to n embeddings for each pair.'''
    # Create dict pair->list[embeddings]
    unique_pairs = sorted(set(all_pairs))
    pair_idx2img_embs = {pair: [] for pair in unique_pairs}
    for i, pair in enumerate(all_pairs):
        pair_idx2img_embs[pair].append(embeddings[i])

    # Select (at most) n embeddings for each pair
    selected_embs, selected_all_pairs = [], []
    for pair in unique_pairs:
        pair_reps = pair_idx2img_embs[pair]
        k = min(n['_'.join(pair)], len(pair_reps)) if isinstance(n, dict) else min(n, len(pair_reps))
        sampled_reps = random.sample(pair_reps, k)
        selected_embs += sampled_reps
        selected_all_pairs += [pair] * k
    selected_embs = torch.stack(selected_embs)
    return selected_embs, selected_all_pairs


def compute_logits(image_embs, label_embs):
    logit_scale = exp(0.07)
    logit_scale = logit_scale if logit_scale<=100.0 else 100.0
    logits = logit_scale * image_embs @ label_embs.t()
    return logits.to('cpu')


def compute_weights(embs_for_IW, all_pairs_IW, train_dataset, Factorizer, use_clip_score=False, temperature=0.01):
    device = embs_for_IW.device
    if len(set(all_pairs_IW))==len(all_pairs_IW):
        weights = None
    else:
        if use_clip_score:  # Use CLIP Weights (only in image modality)
            text_embs, text_pairs = train_dataset.load_text_embs(train_dataset.pairs)

            factorizer = Factorizer(text_embs, text_pairs)
            robust_text_embs = factorizer.compute_ideal_words_approximation(target_pairs=all_pairs_IW)

            logits = torch.sum(embs_for_IW * robust_text_embs, dim=1)  # CLIP score
            
            T = temperature
            weights = logits if T=='None' else torch.exp(logits / float(T))
        else:
            weights = torch.ones(len(all_pairs_IW)).float().to(device)  # Uniform weights within pairs

        # Normalize within pair:
        _, inverse = np.unique(all_pairs_IW, axis=0, return_inverse=True)
        inverse = torch.LongTensor(inverse).to(device)
        group_sums = torch.bincount(inverse, weights=weights).float()
        weights /= group_sums[inverse]
    return weights


def main(config: argparse.Namespace, verbose=False):
    if config.experiment_name != 'clip' and config.modality_IW is None:
        raise Exception("Argument --modality_IW is required when --experiment_name!='clip'.")
    
    set_seed(42)    # Set seed for reproducibility. Effective if config.modality_IW='image and config.n_images!=None.
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    dataset_path = DATASET_PATHS[config.dataset]
    model_info = {
        'model_architecture': config.model_architecture,
        'model_pretraining': config.model_pretraining
    }

    test_dataset = CompositionDatasetEmbeddings(dataset_path,
                                     phase=config.test_phase,
                                     open_world=config.open_world,
                                     **model_info)   
         
    train_dataset = CompositionDatasetEmbeddings(dataset_path,
                                        phase='train',
                                        **model_info)
    
    if verbose:
        scenario = 'open world' if config.open_world else 'closed world'
        print(f"Experiment name : {config.experiment_name}   IW modality : {config.modality_IW}    Scenario : {scenario}")
        print(f'Running on      : {device}')
        print(f"Dataset         : {config.dataset}")
        print(test_dataset)

    all_results = []
    for e in range(config.n_exp):
        if verbose: print(f'Running experiment {e+1}/{config.n_exp}')

        # Compute representations for test pairs
        if config.experiment_name == 'clip':
            # Representations are clip text embeddings:
            class_embs = test_dataset.load_text_embs([(None, o) for o in test_dataset.objs])
        else:
            # Representations are ideal words approximations:
            
            # 1) Prepare the embeddings that will be used to compute the ideal words (primitive directions in the optimal decomposition)
            if config.modality_IW == 'text':
                all_pairs_IW = train_dataset.full_pairs
                embs_for_IW, all_pairs_IW = train_dataset.load_text_embs(all_pairs_IW)
            elif config.modality_IW == 'image':
                embs_for_IW, all_pairs_IW = train_dataset.load_all_image_embs()
                if config.n_images is not None:
                    embs_for_IW, all_pairs_IW = select_n_embs_per_pair(
                        embs_for_IW, all_pairs_IW, n=config.n_images
                    )

            # 2) Compute noise distribution
            if 'CW' in config.experiment_name:  # Use CLIP Weights (only in image modality)
                name, _, T = config.experiment_name.split('_') # Expect name_CW_T
                weights = compute_weights(embs_for_IW, all_pairs_IW, train_dataset, FACTORIZERS[name],
                                          use_clip_score=True, temperature=T)
            else:
                name = config.experiment_name
                weights = compute_weights(embs_for_IW, all_pairs_IW, train_dataset, FACTORIZERS[name],
                                          use_clip_score=False)
            
            # 3) Select the factorizer used to compute/combine ideal words
            Factorizer = FACTORIZERS[name]
            factorizer = Factorizer(embs_for_IW, all_pairs_IW, weights)
            
            # 4) Compute class representations as ideal words
            class_embs = factorizer.combine_ideal_words(factorizer.obj_IW)
       
        # Compute predictions
        image_embs, all_pairs_true = test_dataset.load_all_image_embs()
        image_embs = image_embs.to(device)
        class_embs = class_embs.to(device)

        logits = compute_logits(image_embs, class_embs)
        preds = logits.argmax(dim=1).numpy().astype(int)

        # Evaluate predictions
        targets_t = np.array(
            [test_dataset.obj2idx[o] for _, o in all_pairs_true], dtype=int
        ) 
        targets_s = np.array(
            [test_dataset.attr2idx[a] for a, o in all_pairs_true], dtype=int
        )
        result = evaluate_clip(preds, targets_t, targets_s, verbose)
        all_results.append(result)

    # Combine results of multiple experiments
    if config.n_exp > 1:
        all_stats = list(all_results[0].keys())
        result = defaultdict(list)
        for res in all_results:
            for stat in all_stats:
                result[stat + ' (list)'].append(res[stat])
        # Compute mean and std for each statistic
        for stat in all_stats:
            result[stat + ' (mean)'] = np.mean(result[stat + ' (list)'])
            result[stat + ' (std)'] = np.std(result[stat + ' (list)'])
        result = dict(result)

    # Show and save results
    if config.result_path is not None:
        with open(config.result_path, 'w+') as fp:
            experiment_details = {'config': vars(config),
                                  'result': result}
            json.dump(experiment_details, fp, indent=4)

    if verbose:
        to_float = lambda v: [float(x) for x in v] if isinstance(v, list) else float(v)
        print({k: to_float(v)  for k, v in result.items()})
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        help="name of the dataset",
        type=str, required=True)
    parser.add_argument(
        "--model_architecture",
        help="clip model architecture",
        type=str, default="ViT-L-14")
    parser.add_argument(
        "--model_pretraining",
        help="clip model pretraining set",
        type=str, default="openai")
    parser.add_argument(
        "--experiment_name",
        help="name of the experiment",
        type=str)
    parser.add_argument(
        "--modality_IW",
        help="modality considerer for ideal words",
        choices=['text', 'valid text', 'image'],
        type=str, default=None)
    parser.add_argument(
        "--n_images",
        help="limit the number of images per pair in IW computation with image modality",
        type=int, default=None)
    parser.add_argument(
        "--open_world",
        help="evaluate on open world setup",
        action="store_true")
    parser.add_argument(
        "--n_exp",
        help="number of times the experiment is repeated. >1 makes sense only if modality_IW='image' and n_image!=None",
        default=1, type=int)
    parser.add_argument(
        "--test_phase",
        help="test or val",
        default="test", type=str)
    parser.add_argument(
        "--result_path",
        help="path to json file. Result is saved here.",
        type=str, default=None)
    
    config = parser.parse_args()
    main(config, verbose=True)
