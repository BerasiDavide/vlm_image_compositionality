'''Objects for decomposing embeddings of attribute-object compositions.'''

import torch
import numpy as np
import torch.nn.functional as F

from sphere import (calculate_intrinstic_mean,
                    logarithmic_map,
                    exponential_map,
                    parallel_transport)


def weighted_mean(embeddings, weights=None):
    '''Calculate the weighted mean of `embeddings`. The `weights` are normalized.'''
    if weights is None:
        return embeddings.mean(dim=0)
    else:
        return weights @ embeddings / weights.sum()


def compute_group_means(embeddings, group_ids, unique_groups, weights=None):
    '''
    Computes the mean vector for each group in unique_groups.
    Vector embeddings[i] and embeddings[j] belongs to the same group iff group_ids[i]=group_ids[j].
    '''
    group_id2idx= {id: [] for id in unique_groups}
    for i, group_id in enumerate(group_ids):
        group_id2idx[group_id].append(i)
    
    means = []
    for id in unique_groups:
        idx = group_id2idx[id]

        group_weights = None if weights is None else weights[idx]
        group_mean = weighted_mean(embeddings[idx], group_weights)
        means.append(group_mean)

    return torch.stack(means)


def compute_attr_obj_means(embeddings, all_pairs_gt, centered=True, weights=None):
    '''
    Computes mean for each attribute and object.
    If two or more embeddings have the same pair, a the denoising step is performed first.
    `weights` gives the weight distribution within pair. If None, uniform weights are used.
    '''

    mean_all = weighted_mean(embeddings, weights)
    
    attrs, objs = zip(*all_pairs_gt)
    attr_means = compute_group_means(embeddings, attrs, sorted(set(attrs)), weights)  # sorted wrt unique attrs
    obj_means = compute_group_means(embeddings, objs, sorted(set(objs)), weights)     # sorted wrt unique objs
    
    if centered:
        attr_means = attr_means - mean_all
        obj_means = obj_means - mean_all

    return mean_all, attr_means, obj_means


# Factorizers

class CompositionalFactorizer:

    def __init__(self, embs_for_IW, all_pairs_gt, weights=None):
        '''
        Class that represents a compositional structure for a set of embeddings.
        Input:
            dataset: dataset of the embeddings
            embs_for_IW: embeddings used to compute the Ideal Words (primitive directions in the optimal decomposition)
            all_pair_gt: (attr, obj) label for `embs_for_IW`
            weights: weights assigned to the `embs_for_IW`, if `None` uniform weights are used. Weights are automatically normalized within pair.
        '''
        self.device = embs_for_IW.device
        self.all_pairs_gt = all_pairs_gt
        self.embs_for_IW = embs_for_IW
        self.weights = weights

        attrs, objs = zip(*all_pairs_gt)
        self.attrs = sorted(set(attrs))
        self.objs = sorted(set(objs))

        self.attr2idx = {attr: idx for idx, attr in enumerate(self.attrs)}
        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}

        # Compute IW for attrs and objs in dataset
        self.context, self.attr_IW, self.obj_IW = self.compute_ideal_words(
            embeddings=embs_for_IW,
            all_pairs_gt=all_pairs_gt,
            weights=weights
        )

    def compute_ideal_words(self, embeddings, all_pairs_gt):
        '''
        Extracts ideal words from `embeddings` labeled with `all_pairs_gt`.
        '''
        raise(NotImplementedError)
    
    def combine_ideal_words(self, *ideal_words, context=None):
        '''
        Combines ideal words using `context` as center.
        If context is `None`, `self.context` is used.
        '''
        raise(NotImplementedError)
    
    def get_attr_IW(self, attr):
        attr_idx = self.attr2idx[attr]
        return self.attr_IW[attr_idx]
    
    def get_obj_IW(self, obj):
        obj_idx = self.obj2idx[obj]
        return self.obj_IW[obj_idx]

    def compute_ideal_words_approximation(self, target_pairs):
        target_attr_idx = torch.tensor(
            [self.attr2idx[attr] for attr, _ in target_pairs],
            device=self.device)
        target_obj_idx = torch.tensor(
            [self.obj2idx[obj] for _, obj in target_pairs],
            device=self.device)
        
        # Select attr_IW and obj_IW for target pairs
        attrIW_target = self.attr_IW[target_attr_idx]
        objIW_target = self.obj_IW[target_obj_idx]

        # Compute IW approximation for target pairs
        target_IWapprox = self.combine_ideal_words(attrIW_target, objIW_target)

        return target_IWapprox
        
    def __str__(self) -> str:
        return self.name


class LDE(CompositionalFactorizer):
    name = 'LDE'

    def compute_ideal_words(self, embeddings, all_pairs_gt, weights):
        return compute_attr_obj_means(embeddings, all_pairs_gt, weights=weights)

    def combine_ideal_words(self, *ideal_words, context=None):
        if context is None:
            context = self.context
        ideal_words = torch.stack(ideal_words)
        return context + torch.sum(ideal_words, dim=0)
    
    def get_denoised_pair(self):
        unique_pairs = list(set(self.all_pairs_gt))
        denoised_pair = compute_group_means(self.embs_for_IW, self.all_pairs_gt, unique_pairs)
        return unique_pairs, denoised_pair


class GDE(CompositionalFactorizer):
    name = 'GDE'
    
    def compute_ideal_words(self, embeddings, all_pairs_gt, weights):
        intrinsic_mean = calculate_intrinstic_mean(embeddings, weights, init='normalized mean')  # mu

        # 1) Map embedding to the tangent space T_muS^n
        embeddings_T = logarithmic_map(intrinsic_mean, embeddings)

        # 2) Compute IW on the tangent space
        v_c, attr_IW, obj_IW = compute_attr_obj_means(embeddings_T, all_pairs_gt, weights=weights)
        assert torch.norm(v_c, p=2) < 1e-5 # should be v_c=0

        context = intrinsic_mean
        return context, attr_IW, obj_IW

    def combine_ideal_words(self, *ideal_words, context=None):
        if context is None:
            original_contex = True
            context = self.context
        else:
            original_contex = False

        # 3) Combine ideal words in the tangent plane
        ideal_words = torch.stack(ideal_words)
        embs_approx_T = torch.sum(ideal_words, dim=0)

        # 4) Map the obtained approximation back to the sphere
        if original_contex:
            embs_approx = exponential_map(context, embs_approx_T)
        else:
            # If context is not mu, we need to transport embs_approx_T from T_muS^n to T_contextS^n
            embs_approx_T_transported = parallel_transport(self.context, context, embs_approx_T)
            embs_approx = exponential_map(context, embs_approx_T_transported)
        
        return embs_approx

    def get_denoised_pair(self):
        unique_pairs = list(set(self.all_pairs_gt))
        embs_T = logarithmic_map(self.context, self.embs_for_IW)
        denoised_pair_T = compute_group_means(embs_T, self.all_pairs_gt, unique_pairs)
        denoised_pair = exponential_map(self.context, denoised_pair_T)
        return unique_pairs, denoised_pair


FACTORIZERS = {
    'LDE': LDE,
    'GDE': GDE,
}