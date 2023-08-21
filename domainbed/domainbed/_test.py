# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import collections
import json
import os
import random
import sys
import time
import uuid
from itertools import chain

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader, DataParallelPassthrough
from domainbed import model_selection
from domainbed.lib.query import Q

from datetime import datetime, timedelta

from torchvision.transforms.functional import to_pil_image
from matplotlib import pyplot as plt

import clip
import pickle

def _log(text):
    now = datetime.now()
    now = now + timedelta(hours=9)
    print(f"({now.strftime('%Y-%m-%d %H:%M:%S')})\tLOG:\t",f"start {text}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--dataset_dir', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--task', type=str, default="domain_generalization",
        help='domain_generalization | domain_adaptation')
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--algorithm_path', type=str)

    # ---

    # parser.add_argument('--clip_backbone', type=str, default="None")
    args = parser.parse_args()
        
    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    # NOTE
    algorithm_path = args.algorithm_path
    if algorithm_path == None:
        algorithm_dict = None
    else:
        algorithm_dict = torch.load(algorithm_path)['model_dict']
    if 'DPLCLIPALL' in args.algorithm:
        algorithm_dict['network.input.weight'] = algorithm_dict.pop('network.module.input.weight')
        algorithm_dict['network.input.bias'] = algorithm_dict.pop('network.module.input.bias')
        algorithm_dict['network.hiddens.0.weight'] = algorithm_dict.pop('network.module.hiddens.0.weight')
        algorithm_dict['network.hiddens.0.bias'] = algorithm_dict.pop('network.module.hiddens.0.bias')
        algorithm_dict['network.output.weight'] = algorithm_dict.pop('network.module.output.weight')
        algorithm_dict['network.output.bias'] = algorithm_dict.pop('network.module.output.bias')
    
    os.makedirs(args.output_dir, exist_ok=True)

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    hparams['test_envs'] = [int(i) for i in args.test_envs]

    hparams['clip_transform'] = hparams['backbone'] == 'clip'

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir, args.test_envs, hparams)
    else:
        raise NotImplementedError

    # ---

    _log('test')

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
        len(dataset) - len(args.test_envs), hparams)

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)
    if hasattr(algorithm, 'network'):
        algorithm.network = DataParallelPassthrough(algorithm.network)
    else:
        for m in algorithm.children():
            m = DataParallelPassthrough(m)

    # ---

    os.makedirs(f'{args.output_dir}/test', exist_ok=True)

    with open(args.dataset_dir+f'/tsne_dataset_{args.dataset}.pkl', 'rb') as fr:
        dataset = pickle.load(fr)
    domain = dataset['domain'] #sum([_.tolist() for _ in dataset['domain']],[])
    images = dataset['images']
    labels = sum([_.tolist() for _ in dataset['labels']],[])


    if args.algorithm in ['CLIP']:
        deep_feature_image = []
        deep_feature_text = []

        clip_model = clip.load(hparams['clip_backbone'])[0].float()
        with torch.no_grad():
            _log('encoding')
            for x in images:
                x = x.to(device)
                image_feature = clip_model.encode_image(x)
                deep_feature_image += image_feature.cpu().numpy().tolist()

                text_feature = clip_model.encode_text(algorithm.prompt)
                deep_feature_text += text_feature.cpu().numpy().tolist()
            
            _log(f'drawing tsne, layer11')
            tsne = TSNE(n_components=2, random_state=0)
            
            # NOTE: class
            actual = np.array(labels)

            plt.figure(figsize=(10, 10))
            cluster_image = np.array(tsne.fit_transform(np.array(deep_feature_image)))
            for i, label in zip(range(len(hparams['class_names'])), hparams['class_names']):
                idx = np.where(actual == i)
                plt.scatter(cluster_image[idx, 0], cluster_image[idx, 1], marker='.', s=1, label=label)
                
            plt.legend()
            plt.xlim(-100,100)
            plt.ylim(-100,100)
            plt.savefig(f'{args.output_dir}/class/tsne_layer11.png', bbox_inches='tight')
            plt.close()

            plt.figure(figsize=(10, 10))
            cluster_text = np.array(tsne.fit_transform(np.array(deep_feature_text)))
            for i, label in zip(range(len(hparams['class_names'])), hparams['class_names']):
                idx = i
                plt.scatter(cluster_text[idx, 0], cluster_text[idx, 1], marker='+', s=30, label=label)
                
            plt.legend()
            plt.xlim(-600,600)
            plt.ylim(-600,600)
            plt.savefig(f'{args.output_dir}/class/tsne_layer11_text.png', bbox_inches='tight')
            plt.close()
    
            # NOTE: domain
            actual = np.array(domain)

            plt.figure(figsize=(10, 10))
            for i in range(4):
                idx = np.where(actual == i)
                plt.scatter(cluster_image[idx, 0], cluster_image[idx, 1], marker='.', label=i)

            plt.legend()
            plt.xlim(-100,100)
            plt.ylim(-100,100)
            plt.savefig(f'{args.output_dir}/domain/tsne_layer11.png', bbox_inches='tight')
            plt.close()

    if args.algorithm in ['CLIPALL']:
        plt.rc('font', size=14)
        plt.rc('axes', labelsize=14)   # x,y축 label 폰트 크기
        plt.rc('xtick', labelsize=14)  # x축 눈금 폰트 크기 
        plt.rc('ytick', labelsize=14)  # y축 눈금 폰트 크기
        plt.rc('legend', fontsize=14)  # 범례 폰트 크기
        plt.rc('figure', titlesize=20) # figure title 폰트 크기
        clip_model = clip.load(hparams['clip_backbone'])[0].float()
        with torch.no_grad():
            for i, x in enumerate(images):
                x = x.to(device)
                image_feature = clip_model.encode_image(x)

                image_weight = algorithm.visual_network(image_feature)
                mean_image_weight = image_weight.mean(dim=0, keepdim=True)
                text_weight = algorithm.textual_network(image_feature)
                mean_text_weight = text_weight.mean(dim=0, keepdim=True)

                wi = np.abs(mean_image_weight.cpu().numpy().squeeze(0))
                wt = np.abs(mean_text_weight.cpu().numpy().squeeze(0))
                plt.figure(figsize=(10, 5))
                plt.bar(range(12), wi)
                plt.savefig(f'{args.output_dir}/test/domain={domain[i][0]},image.png')
                plt.close()
                plt.figure(figsize=(10, 5))
                plt.bar(range(12), wt)
                plt.savefig(f'{args.output_dir}/test/domain={domain[i][0]},text.png')
                plt.close()

                # image_features = algorithm.encode_image(x)
                # image_feature = torch.einsum('da,abc->bc', mean_image_weight, image_features)

                # text_features = algorithm.encode_text(algorithm.prompt)
                # text_feature = torch.einsum('da,abc->bc', mean_text_weight, text_features)

                # image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True) # [64, 512]
                # text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)    # [ 7, 512]

                # # [64, 7]
                # score = image_feature @ text_feature.t() 
                # print(x.shape, score.shape)
                # for j in range(len(x)):
                #     plt.imshow(to_pil_image(x[j]))
                #     plt.savefig(f'{args.output_dir}/test/{score[j].tolist()}.png')

            # plt.figure(figsize=(10, 10))
            # for i, label in zip(range(len(hparams['class_names'])), hparams['class_names']):
            #     idx = np.where(actual == i)
            #     plt.scatter(cluster[idx, 0], cluster[idx, 1], marker='.', label=label)
            # plt.legend()
            # plt.xlim(-100,100)
            # plt.ylim(-100,100)
            # plt.savefig(f'{args.output_dir}/class/tsne_weighted_sum.png', bbox_inches='tight')
            # plt.close()