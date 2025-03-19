'''
Script to perform compositional classification using CLIP Weights (CW) with optimal temperature as the noise distribution.
'''

import os
from argparse import Namespace
from tqdm import tqdm
import argparse
import json
import subprocess
import numpy as np
from collections import defaultdict

from datasets.composition_dataset import CompositionDataset
from datasets.read_datasets import DATASET_PATHS

from classification import main


def run_main_with_optimal_T(config: argparse.Namespace, verbose=True):

    # Perform grid search on validation set to select the optimal temperature value
    TT = np.linspace(0, 0.3, 31)[1:] # grid of temperature values
    objective = 'auc'

    T_optim, objective_optim = 0, 0
    for T in tqdm(TT, "Searching for optimal T"):
        config_T = Namespace(
            open_world= config.open_world,
            dataset= config.dataset,
            experiment_name=f'{config.experiment_name}_CW_{T:.4f}',
            model_architecture= config.model_architecture,
            model_pretraining= config.model_pretraining,
            n_images= config.n_images,
            n_exp= config.n_exp,
            modality_IW= config.modality_IW,
            result_path= None,
            test_phase="val"
            )
        result = main(config_T, verbose=False)   
        obj = result[objective]
        if obj > objective_optim:
            T_optim, objective_optim = T, obj

 
    # Perform experiment with optimal T value
    config_optim = Namespace(
        open_world= config.open_world,
        dataset= config.dataset,
        experiment_name=f'{config.experiment_name}_CW_{T_optim:.4f}',
        model_architecture= config.model_architecture,
        model_pretraining= config.model_pretraining,
        n_images= config.n_images,
        n_exp= config.n_exp,
        modality_IW= config.modality_IW,
        result_path= config.result_path,
        test_phase=config.test_phase
        )
    print(f"*** Running experiment with optimal T = {T_optim:.4f} ***")
    main(config_optim, verbose=verbose) 



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
    run_main_with_optimal_T(config, verbose=True)