import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

import matplotlib.pyplot as plt
import torchvision

import os
import random
import pandas as pd
import numpy as np
import json

import torch.utils.data as data
from torchvision import transforms
from PIL import Image

import collections
import time
from tqdm import tqdm

from datetime import datetime, timedelta


class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, 512)
        self.dropout = nn.Dropout(0.1)
        self.hiddens = nn.ModuleList([nn.Linear(512,512)])
        self.output = nn.Linear(512, n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self):
        super(Algorithm, self).__init__()

    def update(self, minibatches):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError


class CLIP(Algorithm):
    def __init__(self, clip_backbone):
        super(CLIP, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model = clip.load(clip_backbone)[0].float()

        # embedding dim for image and text encoder.
        self.EMBEDDING_DIM = 512
        
        # NOTE: Zero-shot
        print("="*50)
        print('Set self.clip_model.parameters.reguires_grad = False!')
        for name, param in self.clip_model.named_parameters():
            param.requires_grad = False
        print("="*50)
        
    def update(self, minibatches):
        return {'loss': 0}
    
    def predict(self, x):
        logits_per_image, _ = self.clip_model(x, self.prompt)
        return logits_per_image.softmax(dim=-1)


class CLIPALL(CLIP):
    def encode_image(self, image):  # image: [batch_size, W, H, 3]  -> [batch_size, self.EMBEDDING_DIM]
        num_image_layer = self.clip_model.visual.transformer.layers
        image = image.to(self.device)

        with torch.no_grad():
            ## image_feature :  [batch_size, self.EMBEDDING_DIM]
            image_feature = self.clip_model.encode_image(image) # CLIP에서 사전학습된 좋은 image의 properties를 추출한다.
        image_weight = self.visual_network(image_feature)   # 추출한 정보를 바탕으로 이미지 인코더의 각 레이어의 가중치를 결정한다.
        mean_image_weight = image_weight.mean(dim=0, keepdim=True)  # 한 번에 들어오는 minibatch의 도메인이 동일하다는 가정 하에 동일한 가중치를 사용한다.

        with torch.no_grad():
            out_list = []
            x = self.clip_model.visual.conv1(image.type(self.clip_model.dtype))
            x = x.reshape(x.shape[0], x.shape[1], -1)   # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)                      # shape = [*, grid ** 2, width]
            x = torch.cat([self.clip_model.visual.class_embedding.to(x.dtype) + 
                        torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
            x = x + self.clip_model.visual.positional_embedding.to(x.dtype)
            x = self.clip_model.visual.ln_pre(x)
            x = x.permute(1, 0, 2)          # NLD -> LND

            for i in range(num_image_layer):
                x = self.clip_model.visual.transformer.resblocks[i](x)
                tmp = x.permute(1, 0, 2)    # LND -> NLD
                tmp = tmp[:, 0, :]
                out_list.append(tmp)

            image_feature = torch.stack(out_list)               # [12, 32, 768]
            image_feature_origin = self.ln_post(image_feature)  # [12, 32, 768]
        
        image_feature_proj = torch.einsum('abc,acd->abd',   # -> [12, batch_size, self.EMBEDDING_DIM]
                            image_feature_origin,           # [12, batch_size, 768]
                            self.visual_projection)         # [12, 768, self.EMBEDDING_DIM]
        
        image_feature_final = torch.einsum('da,abc->bc',    # -> [batch_size, self.EMBEDDING_DIM]
                                     mean_image_weight,     # [1, 12]
                                     image_feature_proj)    # [12, batch_size, self.EMBEDDING_DIM]

        return image_feature_origin, image_feature_proj, image_feature_final

    def encode_text(self, text, image):    #  text: [batch_size, 77]       -> [batch_size, self.EMBEDDING_DIM]
        with torch.no_grad():
            image_feature = self.clip_model.encode_image(image)         # CLIP에서 사전학습된 좋은 image의 properties를 추출한다.
        text_weight = self.textual_network(image_feature)           # 추출한 정보를 바탕으로 텍스트 인코더의 각 레이어의 가중치를 결정한다.
        mean_text_weight = text_weight.mean(dim=0, keepdim=True)    # 한 번에 들어오는 minibatch의 도메인이 동일하다는 가정 하에 동일한 가중치를 사용한다.

        with torch.no_grad():
            # NOTE: encoding
            out_list = []
            x = self.clip_model.token_embedding(text.to(self.device)).type(self.clip_model.dtype)  # [batch_size, n_ctx, d_clip_model]
            x = x + self.clip_model.positional_embedding.type(self.clip_model.dtype)
            x = x.permute(1, 0, 2)                  # NLD -> LND
            for i in range(self.clip_model.transformer.layers):
                x = self.clip_model.transformer.resblocks[i](x)
                tmp = x.permute(1, 0, 2)            # LND -> NLD
                out_list.append(tmp)
            text_feature = torch.stack(out_list)                                # [12, batch_size, 77, 512]
            text_feature_origin = self.ln_final(text_feature)                   # [12, batch_size, 77, 512]

        # NOTE: projection
        text_feature_proj = torch.einsum('abc,acd->abd',                    # -> [12, batch_size, self.EMBEDDING_DIM]
                            text_feature_origin[:,
                                torch.arange(text_feature_origin.shape[1]), # [batch_size]
                                text.argmax(-1)                             # [batch_size, 77] -> [batch_size]
                            ],                                              # [12, batch_size, self.EMBEDDING_DIM]
                            self.textual_projection)                        # [12, self.EMBEDDING_DIM, self.EMBEDDING_DIM]
        
        text_feature_final = torch.einsum('da,abc->bc',                     # -> [batch_size, self.EMBEDDING_DIM] 
                                    mean_text_weight,                       # [1, 12]
                                    text_feature_proj)                      # [12, batch_size, self.EMBEDDING_DIM]
        
        return text_feature_origin, text_feature_proj, text_feature_final
        
    def __init__(self, cfg, class_name, template): 
        super(CLIPALL, self).__init__(cfg['backbone'])
        
        visual_width = 768       
        textual_width = 512      
        visual_scale = visual_width ** -0.5
        textual_scale = textual_width ** -0.5
        output_dim = self.EMBEDDING_DIM

        classnames = [name.replace('_', ' ') for name in class_name]
        self.prompt = torch.cat([clip.tokenize([t.format(ppt) for t in template]) for ppt in classnames]).to(self.device)
        # prompt: [batch_size, 77]

        self.logit_scale = self.clip_model.logit_scale
        self.dtype = self.clip_model.dtype
        self.num_of_visual_encoder_layers = self.clip_model.visual.transformer.layers
        self.num_of_textual_encoder_layers = self.clip_model.transformer.layers
        self.ln_post = self.clip_model.visual.ln_post#.requires_grad_(True)
        self.ln_final = self.clip_model.ln_final#.requires_grad_(True)
        self.visual_projection = nn.Parameter(  # [12, 768, 512]
            torch.stack([
                visual_scale * torch.randn((visual_width, output_dim), dtype=self.dtype).to(self.device)
                for _ in range(self.num_of_visual_encoder_layers - 1)
            ]+[self.clip_model.visual.proj])).requires_grad_(True)
        self.textual_projection = nn.Parameter( # [12, 512, 512]
            torch.stack([
                textual_scale * torch.randn((textual_width, output_dim), dtype=self.dtype).to(self.device)
                for _ in range(self.num_of_textual_encoder_layers - 1)
            ]+[self.clip_model.text_projection])).requires_grad_(True)

        self.visual_network = MLP(self.EMBEDDING_DIM, 12).to(device=self.device, dtype=self.clip_model.dtype)
        self.textual_network = MLP(self.EMBEDDING_DIM, 12).to(device=self.device, dtype=self.clip_model.dtype)
        
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
        
        self.visual_network.apply(init_weights)
        self.textual_network.apply(init_weights)
        
        print("="*50)
        for name, p in self.named_parameters():
            if p.requires_grad:
                print(f"{name} will be updated.")
        print("="*50)

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=cfg['lr'], eps=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, cfg['train_epoch'] * cfg['data_length'])

    def update(self, minibatch):        # [0]: image, [1]: prompt
        # NOTE: batch_size만큼씩 image-caption pair를 학습시키고자 함. i번째 이미지 <-> i번째 캡션
        x = minibatch[0].cuda().float()                             # [batch_size, 224, 224, 3]
        target = minibatch[1].cuda().long()

        _,_,image_feature = self.encode_image(x)                                    # [batch_size, self.EMBEDDING_DIM]
        _,_,text_feature = self.encode_text(self.prompt,x)                          # [batch_size, self.EMBEDDING_DIM]

        image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)    # [batch_size, self.EMBEDDING_DIM]
        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)       # [batch_size, self.EMBEDDING_DIM]

        score = image_feature @ text_feature.t()            # [batch_size, batch_size]

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * score

        loss = F.cross_entropy(logits, target)   # i번째 이미지 <-pairing-> i번째 캡션

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return {'loss': loss.item(), 'logits':logits, 'scheduler':self.scheduler}

    def predict(self, minibatch):
        x = minibatch.cuda().float()                             # [batch_size, 224, 224, 3]

        _,_,image_feature = self.encode_image(x)                                    # [batch_size, self.EMBEDDING_DIM]
        _,_,text_feature = self.encode_text(self.prompt,x)                          # [batch_size, self.EMBEDDING_DIM]

        image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)    # [batch_size, self.EMBEDDING_DIM]
        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)       # [batch_size, self.EMBEDDING_DIM]

        score = image_feature @ text_feature.t()            # [batch_size, batch_size]

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * score
        return logits

def build_model(clip_backbone, class_name, template, state_dict=None):
    model = CLIPALL(clip_backbone, class_name, template)
    # model.load_state_dict(state_dict)
    return model.eval()