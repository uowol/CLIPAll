# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import copy
import numpy as np

# import domainbed.captionizer as captionizer
from domainbed import networks
from domainbed.lib.misc import random_pairs_of_minibatches

import clip
import pickle

ALGORITHMS = [
    # some algorithms moved to 'other_algorithms.py'
    'CLIP',
    'CLIPALL',
    'DPLCLIP',
    'DPLCLIPALL',
]


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
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
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CLIP, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.hparams = hparams
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.clip_model = clip.load(self.hparams['clip_backbone'])[0].float()

        # embedding dim for image and text encoder.
        self.EMBEDDING_DIM = 512  # 
        
        classnames = [name.replace('_', ' ') for name in hparams['class_names']]
        self.prompt = torch.cat([clip.tokenize(f'a photo of a {ppt}') for ppt in classnames]).to(self.device)

        # NOTE: LP
        # visual_width = 768       
        # textual_width = 512      
        # visual_scale = visual_width ** -0.5
        # textual_scale = textual_width ** -0.5
        # output_dim = self.EMBEDDING_DIM

        # self.clip_model.visual.proj = nn.Parameter(
        #     visual_scale * torch.randn((visual_width, output_dim), dtype=self.clip_model.dtype))
        # self.clip_model.text_projection = nn.Parameter(
        #     textual_scale * torch.randn((textual_width, output_dim), dtype=self.clip_model.dtype))

        # print("="*50)
        # print('Set self.clip_model.parameters.reguires_grad = False!')
        # for name, param in self.clip_model.named_parameters():
        #     # NOTE: LP
        #     if name in [
        #         'text_projection',
        #         'visual.proj'
        #     ]:
        #         param.requires_grad = True
        #         print(f'Set self.clip_model.{name}.reguires_grad = True!')
        #     else: 
        #         param.requires_grad = False
        # print("="*50)

        # NOTE: Zero-shot
        print("="*50)
        print('Set self.clip_model.parameters.reguires_grad = False!')
        for name, param in self.clip_model.named_parameters():
            param.requires_grad = False
        print("="*50)
        

        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams["lr"],
            momentum=self.hparams["momentum"]
        )
        
    def update(self, minibatches, unlabeled=None):
        all_x = [data[0].cuda().float() for data in minibatches]
        all_y = torch.cat([data[1].cuda().long() for data in minibatches])

        x = torch.cat(all_x)
        logits_per_image, _ = self.clip_model(x, self.prompt)

        loss = F.cross_entropy(logits_per_image.softmax(dim=-1), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}
    
    def predict(self, x, paths):
        logits_per_image, _ = self.clip_model(x, self.prompt)
        return logits_per_image.softmax(dim=-1)

### NOTE: 텍스트 인코더와 이미지 인코더의 모든 레이어로부터의 임베딩 벡터를 모두 활용하는 모델
class CLIPALL(CLIP):
    def encode_image(self, image):  # image: [32, W, H, 3]
        # if not self.hparams['use_caption']:
        num_image_layer = self.clip_model.visual.transformer.layers
        image = image.to(self.device)   

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

        image_features = torch.stack(out_list)                          # [12, 32, 768]
        image_features = self.ln_post(image_features)                   # [12, 32, 768]
        image_features = torch.einsum('abc,acd->abd',
                            image_features,                             # [12, 32, 768]
                            self.visual_projection)                     # [12, 768, self.EMBEDDING_DIM]
        return image_features                                           # [12, 32, self.EMBEDDING_DIM]

    def encode_text(self, text):    #  text: [7, 77]
        # NOTE: encoding
        out_list = []
        x = self.clip_model.token_embedding(text.to(self.device)).type(self.clip_model.dtype)  # [batch_size, n_ctx, d_clip_model]
        x = x + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        x = x.permute(1, 0, 2)                  # NLD -> LND
        for i in range(self.clip_model.transformer.layers):
            x = self.clip_model.transformer.resblocks[i](x)
            tmp = x.permute(1, 0, 2)            # LND -> NLD
            out_list.append(tmp)
        text_features = torch.stack(out_list)
        text_features = self.ln_final(text_features)    # [12,  7, 77, 512]

        # NOTE: projection
        text_features = torch.einsum('abc,acd->abd',
                            text_features[:,
                                torch.arange(text_features.shape[1]),   # [ 7]
                                text.argmax(-1)                         # [ 7, 77] -> [ 7]
                            ],                                          # [12,  7, self.EMBEDDING_DIM]
                            self.textual_projection)                    # [12, self.EMBEDDING_DIM, self.EMBEDDING_DIM]
        return text_features                                            # [12,  7, self.EMBEDDING_DIM]
    
    def __init__(self, input_shape, num_classes, num_domains, hparams): 
        super(CLIPALL, self).__init__(input_shape, num_classes, num_domains, hparams)
        visual_width = 768       
        textual_width = 512      
        visual_scale = visual_width ** -0.5
        textual_scale = textual_width ** -0.5
        output_dim = self.EMBEDDING_DIM

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
        # self.visual_projection = nn.Parameter(  # [12, 768, 512]
        #     torch.stack([
        #         visual_scale * torch.randn((visual_width, output_dim), dtype=self.dtype).to(self.device)
        #         for _ in range(self.num_of_visual_encoder_layers)
        #     ])).requires_grad_(True)
        # self.textual_projection = nn.Parameter( # [12, 512, 512]
        #     torch.stack([
        #         textual_scale * torch.randn((textual_width, output_dim), dtype=self.dtype).to(self.device)
        #         for _ in range(self.num_of_textual_encoder_layers)
        #     ])).requires_grad_(True)
        
        self.visual_network = networks.MLP(self.EMBEDDING_DIM, 12, hparams).to(device=self.device, dtype=self.clip_model.dtype)
        self.textual_network = networks.MLP(self.EMBEDDING_DIM, 12, hparams).to(device=self.device, dtype=self.clip_model.dtype)
        
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
        
        self.visual_network.apply(init_weights)
        self.textual_network.apply(init_weights)
        
        classnames = [name.replace('_', ' ') for name in hparams['class_names']]
        print("LOG:",classnames)    # ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
        self.prompt = torch.cat([clip.tokenize(f'a photo of a {ppt}') for ppt in classnames]).to(self.device)

        print("="*50)
        for name, p in self.named_parameters():
            if p.requires_grad:
                print(f"{name} will be updated.")
        print("="*50)

        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams["lr"],
            momentum=self.hparams["momentum"]
        )

    def update(self, minibatches, unlabeled=None):
        # minibatches = [[domain_1], [domain_2], [domain_3]]
        all_x = [data[0].cuda().float() for data in minibatches]
        all_y = torch.cat([data[1].cuda().long() for data in minibatches])
        all_paths = [list(data[2]) for data in minibatches]
        # NOTE: caption part is deprecated.
        # if self.hparams['use_caption']:
        #     all_paths = [list(data[2]) for data in minibatches] # 3 * [32]

        # NOTE: DPLCLIP의 개념을 도입, domain 정보에 따라 레이어의 가중치를 결정
        # image_features :          3 * [32, self.EMBEDDING_DIM]
        image_features = [self.clip_model.encode_image(x) for x in all_x]
        # text_features = [self.clip_model.encode_text(self.prompt) for x in all_x]
        # domain_features :         3 * [32, 12]    # NOTE: 이름 수정해야,
        image_weights = [self.visual_network(feature) for feature in image_features]
        text_weights = [self.textual_network(feature) for feature in image_features]
        # text_weights = [self.textual_network(feature) for feature in text_features]

        # mean_domain_weights :     3 * [ 1, 12]
        mean_image_weights = [weights.mean(dim=0, keepdim=True) for weights in image_weights]
        mean_text_weights = [weights.mean(dim=0, keepdim=True) for weights in text_weights]

        # envs = ['A', 'C', 'P', "S"]
        # with open('/data4/kchanwo/clipall/domain_weights.txt', 'a') as f:
        #     for i in range(3):
        #         iweights = [f"{x:.2f}" for x in mean_image_weights[i].tolist()[0]]
        #         tweights = [f"{x:.2f}" for x in mean_text_weights[i].tolist()[0]]
        #         f.write(f'{envs[i+1]}_image_weights:\t{iweights}\t')
        #         f.write(f'{envs[i+1]}_text_weights:\t{tweights}\n')

        # NOTE: DPLCLIP
        # [12,  7, 512]
        text_features = self.encode_text(self.prompt)
        # 3 * [ 7, 512]
        text_features = [torch.einsum('da,abc->bc', weights, text_features) for weights in mean_text_weights]
        
        # NOTE: CLIPALL
        # 3 * [12, 32, 512]
        image_features = [self.encode_image(x) for x in all_x]
        # 3 * [32, 512]
        image_features = [torch.einsum('da,abc->bc', mean_image_weights[i], image_features[i]) for i in range(3)]

        image_features = [image_features[i] / image_features[i].norm(dim=-1, keepdim=True) for i in range(3)] # 3 * [32, 512]
        text_features = [text_features[i] / text_features[i].norm(dim=-1, keepdim=True) for i in range(3)]    # 3 * [ 7, 512]

        # [96, 7]
        score = torch.cat([image_features[i] @ text_features[i].t() for i in range(3)])

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * score

        loss = F.cross_entropy(logits, all_y) # [96, 21] and [96]

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x, paths):
        # image_features :          [64, self.EMBEDDING_DIM]
        image_feature = self.clip_model.encode_image(x)
        # domain_features :         [64, 12]
        image_weight = self.visual_network(image_feature)
        text_weight = self.textual_network(image_feature)

        # mean_domain_weights :     [ 1, 12]
        mean_image_weight = image_weight.mean(dim=0, keepdim=True)
        mean_text_weight = text_weight.mean(dim=0, keepdim=True)

        # NOTE: DPLCLIP
        # [12, 7, 512]
        text_feature = self.encode_text(self.prompt)
        # 3 * [ 7, 512]
        # TODO: 여기 'da,abc->bc'가 아니라 'da,abc->dbc'로 하고 squeeze(0)을 해주어야 할까??
        text_feature = torch.einsum('da,abc->bc', mean_text_weight, text_feature)
        
        # NOTE: CLIPALL
        # 3 * [64, 512]
        # TODO: 여기 'da,abc->bc'가 아니라 'da,abc->dbc'로 하고 squeeze(0)을 해주어야 할까??
        image_feature = torch.einsum('da,abc->bc', mean_image_weight, self.encode_image(x))

        image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True) # [64, 512]
        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)    # [ 7, 512]

        # [64, 7]
        score = image_feature @ text_feature.t()

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * score

        return logits


class DPLCLIP(CLIP):
    def __init__(self, input_shape, num_classes, num_domains, hparams, sentence_prompt=False):
        super(DPLCLIP, self).__init__(input_shape, num_classes, num_domains, hparams)

        #  initial prompt.
        prompt_prefix = ' '.join(['X'] * hparams['num_domain_tokens'])
        
        classnames = [f"a photo of a {name.replace('_', ' ')}" for name in hparams['class_names']]
        prompts = [prompt_prefix + ' ' + name + '.' for name in classnames]
        # prompts:  ['X X X X X X X X dog.', 'X X X X X X X X elephant.' ...]
        
        #  to get default token_prefix and token_suffix.
        self.tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
        # tokenized_prompts[0] = tensor([49406,   343,   343,   343,   343,   343,   343,   343,   343,  1929, 269, 49407, 0, 0, ...])
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(self.tokenized_prompts).type(self.clip_model.dtype)
        
        self.register_buffer('token_prefix', embedding[:, :1, :])  # SOS
        #  torch.Size([7, 1, 512])
        #  [-0.0001,  0.0002, -0.0046,  ...,  0.0010,  0.0025,  0.0049]
        
        self.register_buffer('token_suffix', embedding[:, hparams['num_domain_tokens'] + 1:, :])  # CLS, EOS
        # torch.Size([7, 68, self.EMBEDDING_DIM]), 68 := 77 - num_domain_tokens_tokens - 2.
        # [ 0.0013,  0.0046, -0.0115,  ...,  0.0112,  0.0147,  0.0040],...,.
        
        self.network = networks.MLP(self.EMBEDDING_DIM, self.EMBEDDING_DIM * hparams['num_domain_tokens'], hparams).to(device=self.device, dtype=self.clip_model.dtype)
        
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
        
        self.network.apply(init_weights)
        
        self.optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=self.hparams["lr"],
            momentum=self.hparams["momentum"]
        )
            
    def update(self, minibatches, unlabeled=None):
        # minibatches = [[domain_1], [domain_2], [domain_3]]
        all_x = [data[0].cuda().float() for data in minibatches]
        all_y = torch.cat([data[1].cuda().long() for data in minibatches])

        #  encode image for each domain.
        image_features = [self.clip_model.encode_image(x) for x in all_x]
        
        #  extract domain_feature for each domain. [32, self.EMBEDDING_DIM] -> [32, self.EMBEDDING_DIM * num_domain_tokens] -> [self.EMBEDDING_DIM * num_domain_tokens].
        domain_features = [self.network(feature) for feature in image_features]
        image_features = torch.cat(image_features)
        #  reshape [self.batch_size, self.EMBEDDING_DIM.]:  -> [1, self.EMBEDDING_DIM.]
        mean_domain_features = [feature.mean(dim=0, keepdim=True) for feature in domain_features]

        #  reshape [1, self.EMBEDDING_DIM.]:  -> [7, self.EMBEDDING_DIM.]
        _mean_domain_features = [feature.repeat_interleave(len(self.hparams['class_names']), dim=0) for feature in mean_domain_features]
        
        #  generate text_feature from domain_feature. text_features.size = [3, 7, 512]
        # text_features = [self._get_text_features(feature) for feature in _mean_domain_features]
        text_features = torch.cat([self._get_text_features(feature) for feature in _mean_domain_features])
            
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits_per_image = self.clip_model.logit_scale.exp() * image_features @ text_features.t()
        loss = F.cross_entropy(logits_per_image, all_y)
            
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def _get_text_features(self, domain_feature, coop=False):
        #  reshape domain_feature: [7, 16 * self.EMBEDDING_DIM] -> [7, 16, self.EMBEDDING_DIM]
        domain_feature = domain_feature.reshape(-1, self.hparams['num_domain_tokens'], self.EMBEDDING_DIM)
        
        #  reshape domain_feature: [7, 16, self.EMBEDDING_DIM] -> [7, 77, self.EMBEDDING_DIM]
        domain_feature = torch.cat([self.token_prefix, domain_feature, self.token_suffix], dim=1)
        
        #  refer CoOp: CoOP github. https://github.com/KaiyangZhou/CoOp/blob/b0a058869cef00a4e4ea5256d40fd7681119c099/trainers/coop.py#L46
        x = domain_feature + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        x = x.permute(1, 0, 2)
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.clip_model.ln_final(x).type(self.clip_model.dtype)
        
        #  mapping domain_features to text_features.
        text_features = x[torch.arange(x.shape[0]), self.tokenized_prompts.argmax(dim=-1)] @ self.clip_model.text_projection      
        return text_features

    def predict(self, x, paths):
        image_feature = self.clip_model.encode_image(x)
        
        domain_feature = self.network(image_feature)
        mean_domain_feature = torch.mean(domain_feature, dim=0, keepdim=True).repeat_interleave(len(self.hparams['class_names']), dim=0)
        text_feature = self._get_text_features(mean_domain_feature)
        
        image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
        return self.clip_model.logit_scale.exp() * image_feature @ text_feature.t()


### NOTE: CLIPALL + DPLCLIP
class DPLCLIPALL(CLIP):
    def __init__(self, input_shape, num_classes, num_domains, hparams): 
        super(DPLCLIPALL, self).__init__(input_shape, num_classes, num_domains, hparams)

        # NOTE: CLIPALL
        visual_width = 768
        textual_width = 512
        visual_scale = visual_width ** -0.5
        textual_scale = textual_width ** -0.5
        output_dim = self.EMBEDDING_DIM

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
            
        
        classnames = [name.replace('_', ' ') for name in hparams['class_names']]
        print("LOG:",classnames)    # ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']

        # NOTE: DPLCLIP
        prompt_prefix = ' '.join(['X'] * hparams['num_domain_tokens'])
        classnames = [f"a photo of a {name.replace('_', ' ')}" for name in hparams['class_names']]
        prompts = [prompt_prefix + ' ' + name + '.' for name in classnames]

        #  to get default token_prefix and token_suffix.
        self.tokenized_prompts = torch.cat([clip.tokenize(p, truncate=True) for p in prompts]).to(self.device)
        # tokenized_prompts[0] = tensor([49406,   343,   343,   343,   343,   343,   343,   343,   343,  1929, 269, 49407, 0, 0, ...])
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(self.tokenized_prompts).type(self.clip_model.dtype)
        
        self.register_buffer('token_prefix', embedding[:, :1, :])  # SOS
        #  torch.Size([7, 1, 512])
        #  [-0.0001,  0.0002, -0.0046,  ...,  0.0010,  0.0025,  0.0049]
        
        self.register_buffer('token_suffix', embedding[:, hparams['num_domain_tokens'] + 1:, :])  # CLS, EOS
        # torch.Size([7, 68, self.EMBEDDING_DIM]), 68 := 77 - num_domain_tokens_tokens - 2.
        # [ 0.0013,  0.0046, -0.0115,  ...,  0.0112,  0.0147,  0.0040],...,.
        
        self.network = networks.MLP(self.EMBEDDING_DIM, self.EMBEDDING_DIM * hparams['num_domain_tokens'], hparams).to(device=self.device, dtype=self.clip_model.dtype)
        self.visual_network = networks.MLP(self.EMBEDDING_DIM, 12, hparams).to(device=self.device, dtype=self.clip_model.dtype)
        self.textual_network = networks.MLP(self.EMBEDDING_DIM, 12, hparams).to(device=self.device, dtype=self.clip_model.dtype)
        
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
        
        self.visual_network.apply(init_weights)
        self.textual_network.apply(init_weights)
        self.network.apply(init_weights)

        print("="*50)
        for name, p in self.named_parameters():
            if p.requires_grad:
                print(f"{name} will be updated.")
        print("="*50)

        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams["lr"],
            momentum=self.hparams["momentum"]
        )

    def encode_image(self, image):          # [12, 96, self.EMBEDDING_DIM]
        # if not self.hparams['use_caption']:
        num_image_layer = self.clip_model.visual.transformer.layers
        image = image.to(self.device)

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

        image_features = torch.stack(out_list)                          # [12, 96, 768]
        image_features = self.ln_post(image_features)                   # [12, 96, 768]
        # image_features = image_features @ self.visual_projection      # [12, 96, self.EMBEDDING_DIM]
        image_features = torch.einsum('abc,acd->abd',
                            image_features,                             # [12, 96, 768]
                            self.visual_projection)                     # [12, 768, self.EMBEDDING_DIM]
        return image_features

    def encode_text(self, domain_feature,   # [12,  7, self.EMBEDDING_DIM]
                        coop=False):
        # [7, 16, self.EMBEDDING_DIM]
        domain_feature = domain_feature.reshape(-1, self.hparams['num_domain_tokens'], self.EMBEDDING_DIM)
        # [7, 77, self.EMBEDDING_DIM]
        domain_feature = torch.cat([self.token_prefix, domain_feature, self.token_suffix], dim=1)

        # NOTE: encoding
        out_list = []
        x = domain_feature + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        x = x.permute(1, 0, 2)                  # NLD -> LND
        for i in range(self.clip_model.transformer.layers):
            x = self.clip_model.transformer.resblocks[i](x)
            tmp = x.permute(1, 0, 2)            # LND -> NLD
            out_list.append(tmp)
        text_features = torch.stack(out_list)
        text_features = self.ln_final(text_features)    # [12,  7, 77, 512]

        # NOTE: projection
        text_features = torch.einsum('abc,acd->abd',
                            text_features[:,
                                torch.arange(text_features.shape[1]),   # [ 7]
                                self.tokenized_prompts.argmax(-1)            # [ 7, 77] -> [ 7]
                            ],                                          # [12,  7, self.EMBEDDING_DIM]
                            self.textual_projection)                    # [12, self.EMBEDDING_DIM, self.EMBEDDING_DIM]
        return text_features

    def update(self, minibatches, unlabeled=None):
        # minibatches = [[domain_1], [domain_2], [domain_3]]
        all_x = [data[0].cuda().float() for data in minibatches]
        all_y = torch.cat([data[1].cuda().long() for data in minibatches])

        # NOTE: DPLCLIP
        # image_features :          3 * [32, self.EMBEDDING_DIM]
        image_features = [self.clip_model.encode_image(x) for x in all_x]
        # domain_features :         3 * [32, self.EMBEDDING_DIM * num_domain_tokens]
        domain_features = [self.network(feature) for feature in image_features]
        # mean_domain_features :    3 * [ 1, self.EMBEDDING_DIM * num_domain_tokens]
        mean_domain_features = [feature.mean(dim=0, keepdim=True) for feature in domain_features]
        # _mean_domain_features :   3 * [ 7, self.EMBEDDING_DIM * num_domain_tokens]
        _mean_domain_features = [feature.repeat_interleave(len(self.hparams['class_names']), dim=0) for feature in mean_domain_features]
        # domain_weights  :         3 * [32, 12]    # NOTE: 이름 수정해야,
        image_weights = [self.visual_network(feature) for feature in image_features]
        text_weights = [self.textual_network(feature) for feature in image_features]
        # mean_domain_weights :     3 * [ 1, 12]
        mean_image_weights = [weights.mean(dim=0, keepdim=True) for weights in image_weights]
        mean_text_weights = [weights.mean(dim=0, keepdim=True) for weights in text_weights]
        
        # NOTE: DPLCLIP
        # [12, 21, 512]
        text_features = torch.cat([self.encode_text(feature) for feature in _mean_domain_features], dim=1)
        # 3 * [21, 512]
        text_features = [torch.einsum('da,abc->bc', weights, text_features) for weights in mean_text_weights]

        # NOTE: CLIPALL
        # 3 * [12, 32, 512]
        image_features = [self.encode_image(x) for x in all_x]
        # 3 * [32, 512]
        image_features = [torch.einsum('da,abc->bc', mean_image_weights[i], image_features[i]) for i in range(3)]

        image_features = [image_features[i] / image_features[i].norm(dim=-1, keepdim=True) for i in range(3)] # 3 * [32, 512]
        text_features = [text_features[i] / text_features[i].norm(dim=-1, keepdim=True) for i in range(3)]    # 3 * [21, 512]

        score = torch.cat([image_features[i] @ text_features[i].t() for i in range(3)])

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * score

        loss = F.cross_entropy(logits, all_y) # [96, 21] and [96]

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}
            
    def predict(self, x, paths):
        # NOTE: DPLCLIP
        image_feature = self.clip_model.encode_image(x)                 # [64, 512]
        domain_feature = self.network(image_feature)                    # [64, 16*512]
        mean_domain_feature = torch.mean(domain_feature, dim=0,         # [ 7, 16*512] 
            keepdim=True).repeat_interleave(len(self.hparams['class_names']), dim=0)

        image_weight = self.visual_network(image_feature)               # [64, 12]
        text_weight = self.textual_network(image_feature)               # [64, 12]
        mean_image_weight = image_weight.mean(dim=0, keepdim=True)      # [ 1, 12]
        mean_text_weight = text_weight.mean(dim=0, keepdim=True)        # [ 1, 12]

        # NOTE: DPLCLIP
        # [12,  7, 512]
        text_feature = self.encode_text(mean_domain_feature)
        text_feature = torch.einsum('da,abc->bc', mean_text_weight, text_feature)

        # NOTE: CLIPALL
        # [12, 64, 512]
        image_feature = self.encode_image(x)
        image_feature = torch.einsum('da,abc->bc', mean_image_weight, image_feature)

        image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True) # [64, 512]
        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)    # [ 7, 512]

        score = image_feature @ text_feature.t()    # [64, 7]

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * score
        return logits