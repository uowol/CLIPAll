import os
import random
import argparse
import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms

from datasets import build_dataset
from datasets.utils import build_data_loader
import clip
from utils import *

import os.path as osp

from torch.cuda.amp import GradScaler, autocast

CUSTOM_DATASETS = {
    "oxford_pets": "oxford_pets", 
    "oxford_flowers": "oxford_flowers",
    "fgvc": "fgvc_aircraft",
    "dtd": "dtd",
    "eurosat": "eurosat", 
    "stanford_cars": "stanford_cars",
    "food101": "food101",
    "sun397": "sun397",
    "caltech101": "caltech101",
    "ucf101": "ucf101",
    "imagenet": "imagenet",
}

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
    
    
class TextEncoder(nn.Module):
    def __init__(self, clip_model, width, output_dim):
        super().__init__()    
        scale = width ** -0.5
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.dtype = clip_model.dtype

        self.text_projection = clip_model.text_projection.cuda()
        self.textual_projection = nn.Parameter( # [12, 512, 512]
            torch.stack([
                scale * torch.randn((width, output_dim), dtype=self.dtype).cuda()
                for _ in range(self.transformer.layers - 1)
            ]+[self.text_projection])).requires_grad_(True)
        
    def forward(self, prompt):
        out_list = []
        x = prompt + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i in range(self.transformer.layers):
            x = self.transformer.resblocks[i](x)
            tmp = x.permute(1, 0, 2)            # LND -> NLD
            out_list.append(tmp)
        text_features = torch.stack(out_list)
        text_features = self.ln_final(text_features).type(self.dtype)
        return text_features
    
    def proj(self, text_features, weights):
        x = text_features.permute(1, 0, 2)  # n_prompt, n_layer, d_model
        # x = self.ln(x).type(self.dtype)
        x = torch.einsum('abc,bcd->abd', x.type(self.dtype), 
                         self.textual_projection.type(self.dtype))  # (8, 12, 768), (12, 768, 512) -> (8, 12, 512)
        x = torch.einsum('bc,acd->abd', 
                         weights.type(self.dtype), x)               # (1, 12), (8, 12, 512)    -> (8, 1, 512)
        x = x.squeeze(1)    # (32, 512)
        
        return x

class ImageEncoder(nn.Module):
    def __init__(self, clip_model, width, output_dim):
        super().__init__()
        scale = width ** -0.5
        self.conv1 = clip_model.visual.conv1
        self.transformer = clip_model.visual.transformer
        self.class_embedding = clip_model.visual.class_embedding
        self.positional_embedding = clip_model.visual.positional_embedding
        self.ln_pre = clip_model.visual.ln_pre
        self.ln_post = clip_model.visual.ln_post
        self.dtype = clip_model.dtype
        
        self.image_projection = clip_model.visual.proj.cuda()
        self.visual_projection = nn.Parameter(  # [12, 768, 512]
            torch.stack([
                scale * torch.randn((width, output_dim), dtype=self.dtype).cuda()
                for _ in range(self.transformer.layers - 1)
            ]+[self.image_projection])).requires_grad_(True)

        self.frozen_image_projection = clip_model.visual.proj.clone().detach().cuda().type(torch.float32)
        self.mlp1 = MLP(output_dim, self.transformer.layers)
        self.mlp2 = MLP(output_dim, self.transformer.layers)
        # self.ln = LayerNorm(width)

        self.softmax = nn.Softmax()
        
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

        self.mlp1.apply(init_weights)
        self.mlp2.apply(init_weights)

    def forward(self, image):
        out_list = []
        
        image = image.type(torch.float16).cuda()

        x = self.conv1(image)
        x = x.reshape(x.shape[0], x.shape[1], -1)   # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)                      # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + 
                    torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)          # NLD -> LND

        for i in range(self.transformer.layers):
            x = self.transformer.resblocks[i](x)
            tmp = x.permute(1, 0, 2)    # LND -> NLD
            tmp = tmp[:, 0, :]
            out_list.append(tmp)

        image_features = torch.stack(out_list)               # [12, 32, 768]
        image_features = self.ln_post(image_features)        # [12, 32, 768]
        
        return image_features
    
    def proj(self, image_features, weights):
        x = image_features.permute(1, 0, 2).type(self.dtype) # batch_size, n_layer, d_model
        # x = self.ln(x).type(self.dtype)
        x = torch.einsum('abc,bcd->abd', x.type(self.dtype), 
                         self.visual_projection.type(self.dtype))   # (32, 12, 768), (12, 768, 512) -> (32, 12, 512)
        x = torch.einsum('bc,acd->abd', 
                         weights.type(self.dtype), x)               # (1, 12), (32, 12, 512)    -> (32, 1, 512)
        x = x.squeeze(1)    # (32, 512)

        return x
    
    def generate_weights(self, image_feature):
        image_feature = image_feature.type(torch.float32) @ self.frozen_image_projection.type(torch.float32)
        w1 = self.mlp1(image_feature).mean(dim=0, keepdim=True)     # 1, 12
        w2 = self.mlp2(image_feature).mean(dim=0, keepdim=True)     # 1, 12   
        # w = self.softmax(w)
        return w1, w2


def get_clip_weights(clipall_model, textual_weights):
    # with torch.no_grad():
    clip_weights = []
    # n_class, 12, n_prompt, 512
    for text_features_ in clipall_model.text_features:    # class 개수만큼 반복
        # 12, n_prompt, 512
        class_embeddings = clipall_model.text_encoder.proj(text_features_, textual_weights) # 77, 512
        class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)  # 512
        class_embedding = class_embedding / class_embedding.norm()
        clip_weights.append(class_embedding)
    clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights


class CLIPALL(nn.Module):
    def __init__(self, cfg, classnames, template, clip_model):
        super().__init__()
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype   # float16
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print("="*50)
        print('Set self.clip_model.parameters.reguires_grad = False!')
        for name, param in clip_model.named_parameters():
            param.requires_grad = False
        print("="*50)
        
        text_width = 512
        image_width = 768
        output_dim = 512
        
        text_encoder = TextEncoder(clip_model, text_width, output_dim)
        if self.dtype == torch.float16:
            text_encoder = text_encoder.cuda()

        image_encoder = ImageEncoder(clip_model, image_width, output_dim)
        if self.dtype == torch.float16:
            image_encoder = image_encoder.cuda()

        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        
        with torch.no_grad():
            self.text_features = []

            for classname in classnames:
                # Tokenize the prompts
                classname = classname.replace('_', ' ')
                tokens = [t.format(classname) for t in template]
                tokens = clip.tokenize(tokens).cuda()
                embeddings = clip_model.token_embedding(tokens).type(self.dtype)
                # prompt ensemble for ImageNet
                text_features = text_encoder(embeddings.cuda()) # 12, n_prompt, 77, 512
                text_features = text_features[:,
                    torch.arange(text_features.shape[1]),       # [batch_size]
                    tokens.argmax(-1)                           # [batch_size, 77] -> [batch_size]
                ]   # n_layer, n_prompt, d_model
                self.text_features.append(text_features)        # [12-layers, n_prompt, 512-emb]
        self.text_features = torch.stack(self.text_features).type(torch.float32)   # n_class, 12, n_prompt, 512

        # embedding dim for image and text encoder.
        self.EMBEDDING_DIM = 512
        
        # self.softmax = nn.LogSoftmax(dim=1)

        print("="*50)
        for name, p in self.named_parameters():
            if p.requires_grad:
                print(f"{name} will be updated.")
        print("="*50)
        
        self.clip_model = clip_model
        
    def encode_image(self, images):
        x = self.image_encoder(images)
        visual_weights, textual_weights = self.image_encoder.generate_weights(x[-1])
        image_features = self.image_encoder.proj(x, visual_weights)
        return image_features, textual_weights
        

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings of Tip-Adapter in yaml format')
    parser.add_argument('--shots', type=int, default=16, help='# of shots')
    args = parser.parse_args()

    return args

def load_model(model, directory):
    if not directory:
        print(
            "Note that load_model() is skipped as no pretrained "
            "model is given (ignore this if it's done on purpose)"
        )
        return

    map_location = None if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(directory, map_location=map_location)
    model.load_state_dict(checkpoint["state_dict"], strict=False)

def load_pretrained_model(cfg, model):
    if cfg['dataset'] == 'imagenet':
        load_model(model,
            f"/data4/kchanwo/clipall/clipall/output/imagenet/CLIPALL/mom_lr2e-3_B256_ep40_16shots/seed1/weighted_projection/model.pth.tar-40") # NOTE!!!
    else:
        load_model(model,
            f"/data4/kchanwo/clipall/clipall/output/{CUSTOM_DATASETS[cfg['dataset']]}/CLIPALL/mom_lr2e-3_B32_ep100_16shots/seed1/weighted_projection/model.pth.tar-100") # NOTE!!!
    print('='*20+"Loaded Model!")
    return model

def run_tip_adapter_F(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_model, clipall_model, train_loader_cache, train_loader_F):
    # Enable the cached keys to be learnable
    adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(clip_model.dtype).cuda()
    adapter.weight = nn.Parameter(cache_keys.t())
    
    params = list(adapter.parameters()) + list(clipall_model.parameters())

    # optimizer = torch.optim.AdamW(params, lr=cfg['lr'], eps=1e-4)
    optimizer = torch.optim.SGD(params, lr=1e-3, momentum=0.1)
    # optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg['lr'], eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F))
    
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    best_acc, best_epoch = 0.0, 0
    
    for train_idx in range(cfg['train_epoch']):
        # Train
        adapter.train()
        clipall_model.train()

        correct_samples, all_samples = 0, 0
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))
        
        for i, (images, target) in enumerate(tqdm(train_loader_F)):
            cache_keys, cache_values = build_cache_model(cfg, clipall_model, train_loader_cache)
            
            images, target = images.cuda(), target.cuda()
            
            # with torch.no_grad():
            image_features, textual_weights = clipall_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            affinity = adapter(image_features)
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values

            clip_weights = get_clip_weights(clipall_model, textual_weights)

            clip_logits = 100. * image_features @ clip_weights
            tip_logits = clip_logits + cache_logits * alpha

            loss = F.cross_entropy(tip_logits, target)

            acc = cls_acc(tip_logits, target)
            correct_samples += acc / 100 * len(tip_logits)
            all_samples += len(tip_logits)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))

        # Eval
        adapter.eval()
        clipall_model.eval()

        affinity = adapter(test_features)
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        clip_logits = 100. * test_features @ clip_weights
        tip_logits = clip_logits + cache_logits * alpha
        acc = cls_acc(tip_logits, test_labels)

        print("**** Tip-Adapter-F's test accuracy: {:.2f}. ****\n".format(acc))
        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx
            torch.save(adapter.weight, cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt-all")
    
    adapter.weight = torch.load(cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt-all")
    print(f"**** After fine-tuning, Tip-Adapter-F's best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")

    print("\n-------- Searching hyperparameters on the val set. --------")

    # Search Hyperparameters
    best_beta, best_alpha = search_hp(cfg, cache_keys, cache_values, val_features, val_labels, clip_weights, adapter=adapter)

    print("\n-------- Evaluating on the test set. --------")
   
    affinity = adapter(test_features)
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values
    
    tip_logits = clip_logits + cache_logits * best_alpha
    acc = cls_acc(tip_logits, test_labels)
    print("**** Tip-Adapter-F's test accuracy: {:.2f}. ****\n".format(max(best_acc, acc)))


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

    # Prepare dataset
    random.seed(1)
    torch.manual_seed(1)
    
    print("Preparing dataset.")
    dataset = build_dataset(cfg['dataset'], cfg['root_path'], args.shots)

    # CLIP
    clip_model, preprocess = clip.load(cfg['backbone'])
    clip_model.eval()
    
    # CLIPALL
    clipall_model = CLIPALL(cfg, dataset.classnames, dataset.template, clip_model)
    load_pretrained_model(cfg, clipall_model)
    clipall_model.eval()
    
    # NOTE: CLIPALL Added
    cfg['load_cache'] = False
    cfg['augment_epoch'] = 1

    val_loader = build_data_loader(data_source=dataset.val, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)
    test_loader = build_data_loader(data_source=dataset.test, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)

    train_tranform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    train_loader_cache = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=train_tranform, is_train=True, shuffle=False)
    train_loader_F = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=train_tranform, is_train=True, shuffle=True)

    # Textual features
    # print("\nGetting textual features as CLIP's classifier.")
    # clip_weights = clip_classifier(dataset.classnames, dataset.template, clip_model)

    # Construct the cache model by few-shot training set
    print("\nConstructing cache model by few-shot visual features and labels.")
    cache_keys, cache_values = build_cache_model(cfg, clipall_model, train_loader_cache)

    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = pre_load_features(cfg, "val", clipall_model, val_loader)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(cfg, "test", clipall_model, test_loader)

    # ------------------------------------------ Tip-Adapter-F ------------------------------------------
    run_tip_adapter_F(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_model, clipall_model, train_loader_cache, train_loader_F)
           

if __name__ == '__main__':
    main()