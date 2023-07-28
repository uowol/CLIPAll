import os.path as osp

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
    "OxfordPets": "a photo of a {}, a type of pet.",            # 
    "OxfordFlowers": "a photo of a {}, a type of flower.",      #
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",     #
    "DescribableTextures": "{} texture.",                       #
    "EuroSAT": "a centered satellite photo of {}.",             # 
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",                #
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",                  #
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}

_tokenizer = _Tokenizer()

# def load_clip_to_cpu(cfg):
#     backbone_name = cfg.MODEL.BACKBONE.NAME
#     url = clip._MODELS[backbone_name]
#     model_path = clip._download(url)

#     try:
#         # loading JIT archive
#         model = torch.jit.load(model_path, map_location="cpu").eval()
#         state_dict = None

#     except RuntimeError:
#         state_dict = torch.load(model_path, map_location="cpu")
#     design_details = {"trainer": 'CLIPall',
#                       "vision_depth": 0,
#                       "language_depth": 0, "vision_ctx": 0,
#                       "language_ctx": 0}
#     model = clip.build_model(state_dict or model.state_dict(), design_details)

#     return model

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

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model, width, layers, heads, output_dim):
        super().__init__()    
        scale = width ** -0.5
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.dtype = clip_model.dtype

        self.layer_positional_embedding = nn.Parameter(scale * torch.randn(self.transformer.layers, width))
        self.ln_layer = LayerNorm(width)
        self.layer_transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads
        )
        self.text_projection = clip_model.text_projection.cuda().requires_grad_(True)
        
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
    
    def proj(self, text_features, tokenized_prompts):
        text_features = text_features[:,
                            torch.arange(text_features.shape[1]),       # [batch_size]
                            tokenized_prompts.argmax(-1)                # [batch_size, 77] -> [batch_size]
                        ]   # n_layer, batch_size, d_model
        x = text_features.permute(1, 0, 2)  # batch_size, n_layer, d_model
        x = x + self.layer_positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # n_layer, batch_size, d_model
        x = self.layer_transformer(x)
        x = x.permute(1, 0, 2)  # batch_size, n_layer, d_model
        x = self.ln_layer(x).type(self.dtype)
        
        x = x[torch.arange(x.shape[0]), -1] @ self.text_projection
        
        return x
    

class ImageEncoder(nn.Module):
    def __init__(self, clip_model, width, layers, heads, output_dim):
        super().__init__()
        scale = width ** -0.5
        self.conv1 = clip_model.visual.conv1
        self.transformer = clip_model.visual.transformer
        self.class_embedding = clip_model.visual.class_embedding
        self.positional_embedding = clip_model.visual.positional_embedding
        self.ln_pre = clip_model.visual.ln_pre
        self.ln_post = clip_model.visual.ln_post
        self.dtype = clip_model.dtype
        
        self.layer_positional_embedding = nn.Parameter(scale * torch.randn(self.transformer.layers, width))
        self.ln_layer = LayerNorm(width)
        self.layer_transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads
        )
        self.image_projection = clip_model.visual.proj.cuda().requires_grad_(True)

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
    
    def proj(self, image_features):
        x = image_features.permute(1, 0, 2) # batch_size, n_layer, d_model
        x = x + self.layer_positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # n_layer, batch_size, d_model
        x = self.layer_transformer(x)
        x = x.permute(1, 0, 2)  # batch_size, n_layer, d_model
        x = self.ln_layer(x).type(self.dtype)
        
        x = x[torch.arange(x.shape[0]), -1] @ self.image_projection

        return x


def _proj_base_text_features(base_text_features, tokens, clip_model, text_encoder, device):    
    text_embeddings = []
    for tokens_, base_text_features_ in zip(tokens, base_text_features):
        if clip_model.dtype == torch.float16:
            # _, text_embedding = text_encoder.proj(
            text_embedding = text_encoder.proj(
                base_text_features_.type(clip_model.dtype), tokens_.cuda())
            text_embeddings.append(text_embedding)  # not support float16 on cpu
        else:
            # _, text_embedding = text_encoder.proj(
            text_embedding = text_encoder.proj(
                base_text_features_.type(clip_model.dtype), tokens_.cuda())
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
        layers = 2
        heads = 16
        output_dim = 512
        
        text_encoder = TextEncoder(clip_model, text_width, layers, heads, output_dim)
        if self.dtype == torch.float16:
            text_encoder = text_encoder.cuda()

        image_encoder = ImageEncoder(clip_model, image_width, layers, heads, output_dim)
        if self.dtype == torch.float16:
            image_encoder = image_encoder.cuda()

        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        
        self.tokens = []
        self.base_text_features = []
        with torch.no_grad():
            for text in classnames:
                tokens = clip.tokenize([template.format(text) for template in TEMPLATES])  # tokenized prompts are indices
                embeddings = clip_model.token_embedding(tokens).type(self.dtype)
                self.base_text_features.append(text_encoder(embeddings.cuda())) # [12-layers, 512-emb]
                self.tokens.append(tokens)
        
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
        with torch.no_grad():
            clip_image_feature = self.clip_model.encode_image(image)    # CLIP에서 사전학습된 좋은 image의 properties를 추출한다.

        try:
            x = self.image_encoder(image)
            image_features = self.image_encoder.proj(x)
        except:
            x = self.image_encoder(image.float())
            image_features = self.image_encoder.proj(x)

        text_features = _proj_base_text_features(
            self.base_text_features, 
            self.tokens, 
            self.clip_model, 
            self.text_encoder, 
            self.device)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)    # [batch_size, self.EMBEDDING_DIM]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)       # [batch_size, self.EMBEDDING_DIM]

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


@TRAINER_REGISTRY.register()
class CLIPall(TrainerX):
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