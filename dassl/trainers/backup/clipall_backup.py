import time
import os
import numpy as np
import os.path as osp
import datetime
from collections import OrderedDict
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from my_dassl.data import DataManager
from my_dassl.optim import build_optimizer, build_lr_scheduler
from my_dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, mkdir_if_missing, resume_from_checkpoint,
    load_pretrained_weights
)
from my_dassl.modeling import build_head, build_backbone
from my_dassl.evaluation import build_evaluator

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.model import Transformer, LayerNorm
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from trainers.imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT


CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.", 
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.", 
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}

_tokenizer = _Tokenizer()

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'CLIPall',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


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
        
        x = self.conv1(image.cuda().float())
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
        x = image_features.permute(1, 0, 2) # batch_size, n_layer, d_model
        # x = self.ln(x).type(self.dtype)
        x = torch.einsum('abc,bcd->abd', x.type(self.dtype), 
                         self.visual_projection.type(self.dtype))   # (32, 12, 768), (12, 768, 512) -> (32, 12, 512)
        x = torch.einsum('bc,acd->abd', 
                         weights.type(self.dtype), x)               # (1, 12), (32, 12, 512)    -> (32, 1, 512)
        x = x.squeeze(1)    # (32, 512)

        return x
    
    def generate_weights(self, image_feature):
        image_feature = image_feature @ self.frozen_image_projection.type(torch.float32)
        w1 = self.mlp1(image_feature).mean(dim=0, keepdim=True)     # 1, 12
        w2 = self.mlp2(image_feature).mean(dim=0, keepdim=True)     # 1, 12   
        # w = self.softmax(w)
        return w1, w2


def _proj_text_features(text_features, text_encoder, textual_weights, device):
    text_embeddings = []
    # n_class, 12, n_prompt, 512
    for text_features_ in text_features:    # class 개수만큼 반복
        # 12, n_prompt, 512
        text_embedding = text_encoder.proj(text_features_, textual_weights)
        text_embeddings.append(text_embedding)
    text_embeddings = torch.stack(text_embeddings).mean(1)
    text_encoder = text_encoder.to(device)
    return text_embeddings.to(device)


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype   # float16
        self.cfg = cfg
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        dataset = cfg.DATASET.NAME

        if dataset == "ImageNet":
            TEMPLATES = IMAGENET_TEMPLATES_SELECT
        else:
            TEMPLATES = []
        TEMPLATES += [CUSTOM_TEMPLATES[dataset]]
        
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
        
        self.text_features = []
        with torch.no_grad():
            for text in classnames:
                tokens = clip.tokenize([template.format(text) for template in TEMPLATES])  # tokenized prompts are indices
                embeddings = clip_model.token_embedding(tokens).type(self.dtype)
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

    def forward(self, image):
        try:
            x = self.image_encoder(image)
            visual_weights, textual_weights = self.image_encoder.generate_weights(x[-1])
            image_features = self.image_encoder.proj(x, visual_weights)
        except:
            x = self.image_encoder(image.float())
            visual_weights, textual_weights = self.image_encoder.generate_weights(x[-1])
            image_features = self.image_encoder.proj(x, visual_weights)

        # text_features = self.text_encoder.proj(self.text_features, textual_weights)
        text_features = _proj_text_features(
            self.text_features, 
            self.text_encoder,
            textual_weights, 
            self.device
        )
        
        # os.makedirs(f"/data4/kchanwo/clipall/clipall/analysis/weights/{self.cfg.DATASET.NAME}", exist_ok=True)
        # with open(f"/data4/kchanwo/clipall/clipall/analysis/weights/{self.cfg.DATASET.NAME}/weights.txt", 'a') as f:
        #     f.write(f"{visual_weights.clone().detach().cpu().numpy().tolist()[0]}/{textual_weights.clone().detach().cpu().numpy().tolist()[0]}\n")

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


@TRAINER_REGISTRY.register()
class CLIPALL(TrainerX):
    """CLIPall"""

    def check_cfg(self, cfg):
        assert cfg.TRAINER.CLIPALL.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.CLIPALL.PREC == "fp32" or cfg.TRAINER.CLIPALL.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()
        
        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)
        self.model.to(self.device)
        self.model = self.model.float()
        
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("weighted_projection", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.CLIPALL.PREC == "amp" else None
        
        # for name, p in self.model.named_parameters():
        #     if p.requires_grad:
        #         print(name, p.size())
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"# of Learnable Parameters: {n_params}")
        
        # print("===After build model===")
        # print(torch.cuda.memory_allocated())
        # print(torch.cuda.max_memory_allocated())
        # print(torch.cuda.memory_reserved())
        # print(torch.cuda.max_memory_reserved())

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.CLIPALL.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label
    
    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()
            
        # print("===After run one epoch===")
        # print(torch.cuda.memory_allocated())
        # print(torch.cuda.max_memory_allocated())
        # print(torch.cuda.memory_reserved())
        # print(torch.cuda.max_memory_reserved())
        # exit()

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)