import os.path as osp
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.model import LayerNorm
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
    design_details = {"trainer": 'LFA',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model
    
class TextEncoder(nn.Module):
    def __init__(self, clip_model, width, output_dim):
        super().__init__()    
        scale = width ** -0.5
        self.transformer = clip_model.transformer.float()   # NOTE: half asserts on softmax function
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.dtype = clip_model.dtype
        self.textual_projection = clip_model.text_projection
        self.cross_attention = nn.MultiheadAttention(output_dim, num_heads=1, dropout=.0)
        # self.V.weight = nn.Parameter(clip_model.text_projection.t())
        
    def forward(self, tokenized_text):
        out_list = []
        x = self.token_embedding(tokenized_text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i in range(self.transformer.layers):
            x = self.transformer.resblocks[i](x.float())    # NOTE: half asserts on softmax function
            # print(f"LOG::TYPEs {x.dtype}")
            tmp = x.permute(1, 0, 2)            # LND -> NLD
            out_list.append(tmp)
        text_features = torch.stack(out_list)
        text_features = self.ln_final(text_features).type(self.dtype)
        # text_features = text_features @ self.textual_projection
        text_features = torch.einsum('abcd,de->abce', text_features, self.textual_projection)
        return text_features

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
        self.visual_projection = clip_model.visual.proj
        self.self_attention = nn.MultiheadAttention(output_dim, num_heads=1, dropout=.0)

    def forward(self, x: torch.Tensor):
        out_list = []
        
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)   # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)                      # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + 
                    torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)          # NLD -> LND
        for i in range(self.transformer.layers):
            x = self.transformer.resblocks[i](x.float())
            tmp = x.permute(1, 0, 2)    # LND -> NLD
            tmp = tmp[:, 0, :]
            out_list.append(self.ln_post(tmp))

        image_features = torch.stack(out_list)  # 12, 32, 768
        # image_features = image_features @ self.visual_projection
        image_features = torch.einsum('abc,cd->abd', image_features, self.visual_projection)
        return image_features


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
        
        TEXT_WIDTH = 512
        IMAGE_WIDTH = 768
        OUTPUT_DIM = 512
        
        text_encoder = TextEncoder(clip_model, TEXT_WIDTH, OUTPUT_DIM)
        image_encoder = ImageEncoder(clip_model, IMAGE_WIDTH, OUTPUT_DIM)

        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        
        self.text_features = []
        with torch.no_grad():
            for text in classnames:
                tokenized_text = clip.tokenize([template.format(text) for template in TEMPLATES])
                text_features = text_encoder(tokenized_text)    # 12, n_prompt, 77, 512
                text_features = text_features[:,                # 12, n_prompt, 512
                    torch.arange(text_features.shape[1]),       
                    tokenized_text.argmax(-1)                           
                ]   # n_layer, n_prompt, d_model
                self.text_features.append(text_features)
        self.text_features = torch.stack(self.text_features)   # n_class, 12, n_prompt, 512

        # self.softmax = nn.LogSoftmax(dim=1)

        print("="*50)
        for name, p in self.named_parameters():
            if p.requires_grad:
                print(f"{name} will be updated.")
        print("="*50)
        
        
    def forward(self, image):
        # print(f"LOG::TYPEs: {image.dtype, self.text_features.dtype}")
        
        # encoder_image
        x = self.image_encoder(image)                           # 12, batch_size, 512
        x, _ = self.image_encoder.self_attention(x,x,x)   # batch_size, 512
        image_features = x[-1]
        print(f"LOG::image_features.shape: {image_features.shape}")
        
        # encoder_text
        text_features = []
        for t in self.text_features:                            # n_clas5s, 12, n_prompt, 512
            t = t.type(torch.float32).to(self.device)           # 12, n_prompt, 512
            for i in range(x.shape[1]):
                text_features_ = []
                for j in range(t.shape[1]):
                    text_features_.append(
                        self.text_encoder.cross_attention(x[:,i,:], t[:,j,:], t[:,j,:]) # 12, 1, 512
                    )
                text_features.append(torch.stack(text_features, dim=1).mean(1))
        text_features = torch.stack(text_features).mean(2)      # n_class, batch_size, 512
        text_features = text_features.permute(1, 2, 0)          # batch_size, 512, n_class
        # print(f"LOG::text_features.shape: {text_features.shape}")

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * torch.einsum('ab,abc->ac', image_features, text_features)
        # print(f"LOG::logits.shape: {logits.shape}")
        # print(f"LOG::logits: {logits}")

        return logits
    

@TRAINER_REGISTRY.register()
class LFA(TrainerX):

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.LFA.PREC == "fp32" or cfg.TRAINER.LFA.PREC == "amp":
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
        self.register_model("layer_aggregation", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.LFA.PREC == "amp" else None

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.LFA.PREC
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