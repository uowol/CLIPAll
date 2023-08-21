import time
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
from dassl.utils import Registry, check_availability
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.model import Transformer, LayerNorm
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from trainers.imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT

import trainers.maple

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

CUSTOM_DATASETS = {
    "OxfordPets": "oxford_pets", 
    "OxfordFlowers": "oxford_flowers",
    "FGVCAircraft": "fgvc_aircraft",
    "DescribableTextures": "dtd",
    "EuroSAT": "eurosat", 
    "StanfordCars": "stanford_cars",
    "Food101": "food101",
    "SUN397": "sun397",
    "Caltech101": "caltech101",
    "UCF101": "ucf101",
    "ImageNet": "imagenet",
    "ImageNetSketch": "imagenet_sketch",
    "ImageNetV2": "imagenetv2",
    "ImageNetA": "imagenet_a",
    "ImageNetR": "imagenet_r",
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
    design_details = {"trainer": 'MaPLe',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0,
                      "maple_length": cfg.TRAINER.MAPLE.N_CTX}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model

def load_pretrained_model(cfg, model_name='CoOp'):
    avai_trainers = TRAINER_REGISTRY.registered_names()
    check_availability(model_name, avai_trainers)
    model = TRAINER_REGISTRY.get(model_name)(cfg)
    _dataset = CUSTOM_DATASETS[cfg.DATASET.NAME]
    if 'imagenet' in _dataset:
        model.load_model(
            f"/data4/kchanwo/clipall/clipall/output/imagenet/MaPLe/vit_b16_c2_ep5_batch4_2ctx_cross_datasets_16shots/seed1",
            epoch=2) # NOTE!!!
    else:
        model.load_model(
            f"/data4/kchanwo/clipall/clipall/output/{_dataset}/MaPLe/vit_b16_c2_ep5_batch4_2ctx_16shots/seed1",
            epoch=5) # NOTE!!!
    # print(model, model.model)
    # model.load_model(cfg.MODEL_DIR, epoch=cfg.LOAD_EPOCH)
    print('='*20+"Loaded Model!")
    return model.model

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection.requires_grad_(True)
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [x, compound_prompts_deeper_text, 0]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)].type(torch.float32) @ self.text_projection.type(torch.float32)

        return x

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, pretrained_model):
        super().__init__()
        # NOTE: MaPLe
        self.prompt_learner = pretrained_model.prompt_learner
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        # NOTE: CLIPALL
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
                
        text_width = 512
        image_width = 768
        output_dim = 512
        
        self.clip_model = clip_model
        for name, param in self.clip_model.named_parameters():
            if "visual.proj" in name or "text_projection" in name:
                param.requires_grad_(True)
            else: 
                param.requires_grad_(False)
        for name, param in pretrained_model.named_parameters():
            param.requires_grad_(False)

        print("="*50)
        for name, p in self.named_parameters():
            if p.requires_grad:
                print(f"{name} will be updated.")
        print("="*50)

    def forward(self, image):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
        image_features = self.image_encoder(image.type(torch.float32), shared_ctx, deep_compound_prompts_vision)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()
        
        return logits

@TRAINER_REGISTRY.register()
class MaPLeALL(TrainerX):
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
            
        print(f"Loading Pretrained-model")
        pretrained_model = load_pretrained_model(cfg, 'MaPLe')
        
        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model, pretrained_model)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)
        self.model.to(self.device)
        self.model = self.model.float()
        
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("weighted_projection", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.CLIPALL.PREC == "amp" else None
        
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"# of Learnable Parameters: {n_params}")
        
    # def run_epoch(self):
    #     self.set_model_mode("train")
    #     losses = MetricMeter()
    #     batch_time = AverageMeter()
    #     data_time = AverageMeter()
    #     self.num_batches = len(self.train_loader_x)

    #     end = time.time()
    #     for self.batch_idx, batch in enumerate(self.train_loader_x):
    #         data_time.update(time.time() - end)
    #         loss_summary = self.forward_backward(batch)
    #         batch_time.update(time.time() - end)
    #         losses.update(loss_summary)

    #         meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
    #         only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
    #         if meet_freq or only_few_batches:
    #             nb_remain = 0
    #             nb_remain += self.num_batches - self.batch_idx - 1
    #             nb_remain += (
    #                 self.max_epoch - self.epoch - 1
    #             ) * self.num_batches
    #             eta_seconds = batch_time.avg * nb_remain
    #             eta = str(datetime.timedelta(seconds=int(eta_seconds)))

    #             info = []
    #             info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
    #             info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
    #             info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
    #             info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
    #             info += [f"{losses}"]
    #             info += [f"lr {self.get_current_lr():.4e}"]
    #             info += [f"eta {eta}"]
    #             print(" ".join(info))

    #         n_iter = self.epoch * self.num_batches + self.batch_idx
    #         for name, meter in losses.meters.items():
    #             self.write_scalar("train/" + name, meter.avg, n_iter)
    #         self.write_scalar("train/lr", self.get_current_lr(), n_iter)

    #         end = time.time()
            
    #     print("===After run one epoch===")
    #     print(torch.cuda.max_memory_allocated())
    #     exit()

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
            if 'image_encoder.proj' in name or 'text_encoder.text_projection' in name: continue

            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)