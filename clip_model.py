"""
Code from: https://github.com/AnanyaKumar/transfer_learning/blob/main/unlabeled_extrapolation/models/clip_model.py
"""

import clip
from collections import OrderedDict
import torchvision.models as models
from torchvision.models import resnet50
import torch
from torch import nn
import os

from torchvision.transforms import Normalize


MODELS = {'RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16',
          'ViT-L/14', 'ViT-L/14@336px'}

normalize_transform = Normalize(
    mean=(0.48145466, 0.4578275, 0.40821073),
    std=(0.26862954, 0.26130258, 0.27577711))

def set_linear_layer_func(layer, coef, intercept):
    coef_tensor = torch.tensor(coef, dtype=layer.weight.dtype).cuda()
    bias_tensor = torch.tensor(intercept, dtype=layer.bias.dtype).cuda()
    coef_param = torch.nn.parameter.Parameter(coef_tensor)
    bias_param = torch.nn.parameter.Parameter(bias_tensor)
    layer.weight = coef_param
    layer.bias = bias_param


def build_model_scratch(model_name, device):
    model_path = clip.clip._download(clip.clip._MODELS[model_name], os.path.expanduser("~/.cache/clip"))
    model = torch.jit.load(model_path, map_location="cpu").eval()
    state_dict = model.state_dict()

    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = clip.model.CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    clip.model.convert_weights(model)
    return model.eval().to(device)


def set_requires_grad(component, val):
    for param in component.parameters():
        param.requires_grad = val


class ClipModel(nn.Module):

    def __init__(self, model_name, scratch=False):
        # If scratch is True, then we randomly initialize weights.
        super().__init__()
        if model_name not in MODELS:
            raise ValueError(f'model_name must be in {MODELS} but was {model_name}')
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        # Note that model has both a language and vision part.
        if scratch:
            model = build_model_scratch(model_name, device=self._device)
        else:
            model, _ = clip.load(model_name, device=self._device)
        self._model_name = model_name
        self._model = model
        self._model.visual.float()
        self.fc = None

    def forward(self, x):
        features = self.get_features(x)
        if self.fc is None:
            return features
        return self.fc(features)

    def set_requires_grad(self, val):
        for param in self._model.parameters():
            param.requires_grad = val
        if self.fc is not None:
            for param in self.fc.parameters():
                param.requires_grad = val

    def freeze_bottom_k(self, k):
        if self._model_name in {'ViT-B/32', 'ViT-B/16',
                                'ViT-L/14', 'ViT-L/14@336px'}:
            if k > 0:
                set_requires_grad(self._model.visual.conv1, False)
            if k > 1:
                set_requires_grad(self._model.visual.ln_pre, False)
            if k > 2:
                resblocks = self._model.visual.transformer.resblocks
                n_freeze_transformers = min(k-2, len(resblocks))
                for i in range(n_freeze_transformers):
                    set_requires_grad(resblocks[i], False)
                if k-2 > len(resblocks):
                    set_requires_grad(self._model.visual.ln_post, False)
        elif self._model_name in {'RN50'}:
            visual = self._model.visual
            if k > 0:
                set_requires_grad(visual.conv1, False)
                set_requires_grad(visual.conv2, False)
                set_requires_grad(visual.conv3, False)
                set_requires_grad(visual.bn1, False)
                set_requires_grad(visual.bn2, False)
                set_requires_grad(visual.bn3, False)
            layers = [visual.layer1, visual.layer2, visual.layer3, visual.layer3,
                      visual.attnpool]
            if k > 1:
                n_freeze_upper = min(k-1, len(layers))
                for i in range(n_freeze_upper):
                    set_requires_grad(layers[i], False)
        else:
            raise NotImplementedError

    def reset_classifier(self, num_classes):
        num_in_features = self._model.visual.output_dim
        self.fc = nn.Linear(num_in_features, num_classes)
        self.fc.to(self._device)

    def add_probe(self, probe):
        self.fc = probe

    def get_classifier(self):
        return self.fc

    def set_last_layer(self, coef, intercept):
        set_linear_layer_func(self.fc, coef, intercept)

    def get_feature_extractor(self):
        raise NotImplementedError('Be careful, we need to normalize image first before encoding it.')
    
    # def get_features(self, x):
    #     return self._model.encode_image(normalize_transform(x))
    def get_features(self, x):
        return self._model.encode_image(x)

def main():
    net = ClipModel(model_name='RN50',scratch=False)
    pdb.set_trace()

