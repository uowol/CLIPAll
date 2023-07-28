import os
import random
import argparse
import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn

from datasets.imagenet import ImageNet
import clip
import clipall
from utils import *


def get_arguments():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings of Tip-Adapter in yaml format')
    args = parser.parse_args()

    return args


def run_clipall(cfg, class_name, template, train_loader, test_loader):
    clipall_model = clipall.build_model(cfg, class_name, template)
    best_acc, best_epoch = 0.0, 0

    for train_idx in range(cfg['train_epoch']):
        # Train
        clipall_model.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))

        for i, (images, target) in enumerate(tqdm(train_loader)):
            result = clipall_model.update((images, target))

            acc = cls_acc(result['logits'], target.cuda())
            correct_samples += acc / 100 * len(result['logits'])
            all_samples += len(result['logits'])
            loss_list.append(result['loss'])

        current_lr = result['scheduler'].get_last_lr()[0]
        print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))

        # Eval
        # NOTE: 데이터의 양이 많을 경우 여기서 시간이 엄청 오래 걸림. 중간 중간에 한 번씩 실행해주기
        if (train_idx+1) % 40 == 0:
            clipall_model.eval()
            acc_sum = 0; acc_length = 0
            for i, (images, target) in enumerate(tqdm(test_loader)):
                logits = clipall_model.predict(images)

                acc_sum += cls_acc(logits, target.cuda())
                acc_length += 1

            acc = acc_sum/acc_length

            print("**** CLIPall's test accuracy: {:.2f}. ****\n".format(acc))
            if acc > best_acc:
                best_acc = acc
                best_epoch = train_idx
                torch.save(clipall_model.visual_projection, cfg['cache_dir'] + "/best_F_vp_" + str(cfg['shots']) + "shots.pt")
                torch.save(clipall_model.textual_projection, cfg['cache_dir'] + "/best_F_tp_" + str(cfg['shots']) + "shots.pt")
                torch.save(clipall_model.visual_network, cfg['cache_dir'] + "/best_F_vn_" + str(cfg['shots']) + "shots.pt")
                torch.save(clipall_model.textual_network, cfg['cache_dir'] + "/best_F_tn_" + str(cfg['shots']) + "shots.pt")
    
    clipall_model.visual_projection = torch.load(cfg['cache_dir'] + "/best_F_vp_" + str(cfg['shots']) + "shots.pt")
    clipall_model.textual_projection = torch.load(cfg['cache_dir'] + "/best_F_tp_" + str(cfg['shots']) + "shots.pt")
    clipall_model.visual_network = torch.load(cfg['cache_dir'] + "/best_F_vn_" + str(cfg['shots']) + "shots.pt")
    clipall_model.textual_network = torch.load(cfg['cache_dir'] + "/best_F_tn_" + str(cfg['shots']) + "shots.pt")
    print(f"**** After fine-tuning, CLIPall's best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")


def main():

    # Load config file
    args = get_arguments()
    assert (os.path.exists(args.config))
    
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    cache_dir = os.path.join('./caches', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir

    print("\nRunning configs.")
    print(cfg, "\n")

    # CLIP
    clip_model, preprocess = clip.load(cfg['backbone'])
    clip_model.eval()

    # ImageNet dataset
    random.seed(1)
    torch.manual_seed(1)
    
    print("Preparing ImageNet dataset.")
    imagenet = ImageNet(cfg['root_path'], cfg['shots'], preprocess)

    test_loader = torch.utils.data.DataLoader(imagenet.test, batch_size=64, num_workers=8, shuffle=False)

    train_loader = torch.utils.data.DataLoader(imagenet.train, batch_size=256, num_workers=8, shuffle=True)

    cfg['data_length'] = len(train_loader)

    # ------------------------------------------ CLIPall ------------------------------------------
    run_clipall(cfg, imagenet.classnames, imagenet.template, train_loader, test_loader)           

if __name__ == '__main__':
    main()