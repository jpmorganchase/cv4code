# SPDX-License-Identifier: Apache-2.0
from timm.models.registry import model_entrypoint
import torch
import timm
import numpy as np
from torch._C import Value
import torch.nn as nn
from torch.autograd import Function
from torchvision import models
from collections.abc import Iterable
import torch.nn.functional as F
from functools import partial
from vit_pytorch import ViT
from vit_pytorch.vit_for_small_dataset import ViT as ViT_fsd
from vit_pytorch.vit import Transformer
import math
from einops import repeat
from .layers import ConvViT

_vit_map = {
    'ViT' : ViT,
    'ViT_fsd': ViT_fsd,
    'ConvViT': ConvViT
}

import logging
from enum import Enum

from .layers import *

logger = logging.getLogger('model')


_CUSTOM_IMPLEMENTED = ['restnet-autoencoder']

embedding = {}
def _get_hook(name):
    def hook(model, input, output):
        embedding[name] = output.detach()
    return hook

def get_bottleneck(name=None):
    if name is None:
        name = 'bottleneck'
    if name in embedding:
        return embedding[name]
    return None

def _build_token_transformer(model_cfg):
    backbone = TokenTransformer(
        sequence_lens=model_cfg['sequence_length'],
        dim=model_cfg['embed_dim'],
        depth=model_cfg['depth'],
        mlp_dim=model_cfg['mlp_dim'],
        heads=model_cfg['heads'],
        emb_dropout=0.0 if 'embed_dropout' not in model_cfg else model_cfg['embed_dropout'],
        dropout=0.0 if 'dropout' not in model_cfg else model_cfg['dropout']
    )
    return backbone, model_cfg['embed_dim']

def _build_custom_resnet(model_cfg, dim=2):
    res_init_maxpool = True
    res_init_strides = 2
    res_init_fc = False
    res_init_kernel = 7
    input_dim = model_cfg['input_embedding_dim'][0]
    global_pool = ['max', [1]]
    final_fc_layers = None
    if 'res_global_pool' in model_cfg:
        global_pool = model_cfg['res_global_pool']
    if 'res_init_maxpool' in model_cfg:
        res_init_maxpool = model_cfg['res_init_maxpool']
    if 'res_init_strides' in model_cfg:
        res_init_strides = model_cfg['res_init_strides']
    if 'res_init_fc' in model_cfg:
        res_init_fc = model_cfg['res_init_fc']
    if len(model_cfg['input_embedding_dim']) == 2:
        input_dim = model_cfg['input_embedding_dim'][1]
    if 'res_final_fc' in model_cfg:
        final_fc_layers = model_cfg['res_final_fc']
    if 'res_init_kernel' in model_cfg:
        res_init_kernel = model_cfg['res_init_kernel']
    
    backbone = CustomResNet(
        input_dim,
        model_cfg['res_init_out_channels'], 
        model_cfg['res_blocks'], 
        final_pool=global_pool,
        init_fc=res_init_fc,
        init_strides=res_init_strides,
        init_max_pool=res_init_maxpool,
        final_fc_layers=final_fc_layers,
        init_kernel=res_init_kernel,
        input_dim=dim
        )
    return backbone, backbone.output_size

def _build_custom(model_cfg):
    # input dimension must be defined
    # layers to be defined like ('Linear', (12, 32))
    layers = []
    out_dim = None
    for l_type, conf in model_cfg['layers']:
        if isinstance(conf, Iterable): 
            layers.append(
                getattr(nn, l_type)(*conf)
            )
        elif conf is None:
            layers.append(
                getattr(nn, l_type)()
            )
        else:
            layers.append(
                getattr(nn, l_type)(conf)
            )
        if 'Linear' in l_type:
            out_dim = layers[-1].out_features
        elif 'Conv' in l_type:
            out_dim = conf[1]
    backbone = nn.Sequential(*layers)
    return backbone, out_dim

def _build_timm_resnetv2(name, use_pretrained):
    backbone= timm.create_model(name, pretrained=use_pretrained)
    out_dim = backbone.head.fc.in_channels
    backbone.head.fc = nn.Identity()
    return backbone, out_dim

def _build_resnet(name, use_pretrained):
    model = getattr(models, name)(pretrained=use_pretrained)
    children = list(model.children())
    out_dim = children[-1].in_features
    backbone = nn.Sequential(*children[:-1])
    return backbone, out_dim

def _build_mobilenet(name, use_pretrained):
    model = getattr(models, name)(pretrained=use_pretrained)
    children = list(model.children())
    out_dim = list(children[-1].children())[0].in_features
    backbone = nn.Sequential(*children[:-1])
    return backbone, out_dim

def _build_custom_vit_model(config):
    vit_variant = config['variant']
    model_initialiser = _vit_map[vit_variant]
    model = None
    if vit_variant in {'ViT', 'ViT_fsd'}:
        model = model_initialiser(
            image_size=tuple(config['input_size']), patch_size=config['patch_size'],
            channels=config['input_embedding_dim'][-1], num_classes=1, dim=config['embed_dim'],
            depth=config['depth'], heads=config['heads'], mlp_dim=config['mlp_dim'],
            dropout=config['dropout'] if 'dropout' in config else 0.0,
            emb_dropout=config['embed_dropout'] if 'embed_dropout' in config else 0.0,
            )
        model.mlp_head = nn.LayerNorm(config['embed_dim'])
    elif vit_variant == 'ConvViT':
        if config['input_size'][0] != config['input_size'][1]:
            raise ValueError(f'ConvViT transformer requires input image to be square')
        model = model_initialiser(
            img_size=config['input_size'][0], 
            n_conv_layers=config['n_conv_layers'], kernel_size=config['kernel_size'],
            stride=config['stride'], padding=config['kernel_size']//2, 
            pooling_kernel_size=config['pooling_kernel_size'], pooling_stride=config['pooling_stride'], pooling_padding=config['pooling_kernel_size']//2,
            n_input_channels=config['input_embedding_dim'][-1], num_classes=1, embedding_dim=config['embed_dim'],
            num_layers=config['depth'], num_heads=config['heads'], mlp_ratio=config['mlp_dim']//config['embed_dim'],
            dropout=config['dropout'] if 'dropout' in config else 0.1,
            emb_dropout=config['embed_dropout'] if 'embed_dropout' in config else 0.0,
            attention_dropout=config['attention_dropout'] if 'attention_dropout' in config else 0.1,
            positional_embedding=config['positional_embedding'],
            token_padding=config['token_padding'] if 'token_padding' in config else -1,
            seq_pool=config['sequence_pooling'] if 'sequence_pooling' in config else False
            )
        seq_len = model.tokenizer.sequence_length(
            n_channels=config['input_embedding_dim'][-1], 
            height=config['input_size'][0],
            width=config['input_size'][1]
            )
        logger.info(f'ConvViT sequence length {seq_len}')
        model.classifier.n_channels = config['input_embedding_dim'][-1]
        model.classifier.fc = nn.Identity()
    
    if model is None:
        raise ValueError('vit initialisation failed')
    if 'cos_pos_embedding_init' in config and config['cos_pos_embedding_init']:
        _, max_len, depth = model.pos_embedding.shape
        pos = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, depth, 2) * (-np.log(1e4) / depth))
        with torch.no_grad():
            model.pos_embed[0, :, 0::2] = torch.sin(pos * div_term)
            model.pos_embed[0, :, 1::2] = torch.cos(pos * div_term)
    children = list(model.children())
    out_dim = config['embed_dim']
    backbone = model
    return backbone, out_dim

def _build_vit_model(name, rand_pe_init, use_pretrained, input_dim=3):
    model = timm.create_model(name, pretrained=use_pretrained)
    if not use_pretrained and not rand_pe_init:
        _, max_len, depth = model.pos_embed.shape
        pos = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, depth, 2) * (-np.log(1e4) / depth))
        with torch.no_grad():
            model.pos_embed[0, :, 0::2] = torch.sin(pos * div_term)
            model.pos_embed[0, :, 1::2] = torch.cos(pos * div_term)
    if input_dim != 3:
        kernel_size = model.patch_embed.proj.kernel_size
        stride = model.patch_embed.proj.stride
        out_dim = model.patch_embed.proj.out_channels
        model.patch_embed.proj = nn.Conv2d(input_dim, out_dim, kernel_size, stride)
    children = list(model.children())
    out_dim = children[-1].in_features
    model.reset_classifier(0)
    backbone = model
    return backbone, out_dim

class CoreNetwork(nn.Module):
    def __init__(self, model_conf):
        super(CoreNetwork, self).__init__()
        self.backbone = None
        self.bottleneck = None
        self.projection = None
        out_dim = None
        # for the backbone initialization
        backbone = model_conf['backbone']
        self._embed_transpose = True
        use_pretrained = False
        if 'use_pretrained' in model_conf and model_conf['use_pretrained']:
            use_pretrained = True
        if isinstance(backbone, str):
            if hasattr(models, backbone) or 'torch' in backbone:
                backbone = backbone.split('_')[1]
                logger.info('Construting model using torchvision')
                logger.info(f'backbone: use preconfigured model {backbone}')
                logger.info(f"backbone: use pretrained weights={model_conf['use_pretrained']}")
                if 'resnet' in backbone:
                    self.backbone, out_dim = _build_resnet(
                        backbone, use_pretrained
                    )
                elif 'mobilenet' in backbone:
                    self.backbone, out_dim = _build_mobilenet(
                        backbone, use_pretrained
                    )
            elif backbone == 'CUSTOM_RESNET':
                self.backbone, out_dim = _build_custom_resnet(model_conf, dim=2)
            elif backbone == 'CUSTOM_VIT':
                self.backbone, out_dim = _build_custom_vit_model(model_conf)
            elif backbone == 'TOKEN_TRANSFORMER':
                self.backbone, out_dim = _build_token_transformer(model_conf)
                self._embed_transpose = False
        if self.backbone is None:
            raise NotImplementedError(f'backbone {backbone} not implemented yet')
    
        finetune = True 
        if 'fintune' in model_conf:
            fintune = model_conf['finetune'] or not model_conf['use_pretrained']
        logger.info(f'backbone: requires_gradients={finetune}')
        for layer in self.backbone.children():
            for param in layer.parameters():
                param.requires_grad = finetune

        # for the bottleneck part
        if 'bottleneck' in model_conf and \
             model_conf['bottleneck'] is not None:
            dim = None
            act = nn.ReLU()
            if isinstance(model_conf['bottleneck'], Iterable):
                dim = model_conf['bottleneck'][0]
                act = Swish() if model_conf['bottleneck'][1].lower() == 'swish' else act
            else:
                dim = model_conf['bottleneck']
            bottleneck_activation = True
            if 'linear_bottleneck' in model_conf:
                bottleneck_activation = not model_conf['linear_bottleneck']
            
            bottlneck = [
                nn.Linear(
                    out_dim, dim
                )
            ]
            if bottleneck_activation:
                bottlneck.append(act)
            self.bottleneck = nn.Sequential(*bottlneck)
            out_dim = dim
        else:
            self.bottleneck = nn.Identity()
        
        if 'projection' in model_conf and \
             model_conf['projection'] is not None:
            projection = []
            projection_layers = [model_conf['projection']] if isinstance(model_conf['projection'], int) \
                else model_conf['projection']
            for n_units in projection_layers :
                projection.append(nn.Linear(out_dim, n_units))
                projection.append(nn.ReLU())
                out_dim = n_units
            self.projection = nn.Sequential(*projection)
        
        self.out_dim = out_dim
    
    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        if self.bottleneck is not None:
            x = self.bottleneck(x)
        if self.projection is not None:
            x = self.projection(x)
        return x
    
class CV4code(nn.Module):
    def __init__(self, hparams, task2num_classes=None):
        super(CV4code, self).__init__()
        self.hp = hparams
        self.core = CoreNetwork(self.hp.model)
        self.head_names = []
        self.task2num_classes = task2num_classes
        self.input_embedding_type = None
        for head, task in self.hp.tasks.items():
            branch_layers = []
            out_dim = self.core.out_dim
            if 'grad_reverse_scale' in task and task['grad_reverse_scale'] > 0:
                branch_layers.append(GradientReverseLayer(task['grad_reverse_scale']))
            if 'branch_weights' in task and task['branch_weights'] is not None:
                for n_unit in task['branch_weights']:
                    branch_layers.extend(
                        [nn.Linear(out_dim, n_unit), nn.ReLU()]
                    )
                    if 'norm' in task:
                        branch_layers.append(getattr(nn, task['norm'])(n_unit))
                    out_dim = n_unit
            if 'classes' in task:
                if task2num_classes is None:
                    logger.warn(f'skipping {head} as task2num_classes is None, ok if extracting embedding')
                    continue
                num_classes = len(task['classes'])
                if head in task2num_classes:
                    num_classes = task2num_classes[head]
                setattr(
                    self, f'head_{head}', 
                    nn.Sequential(
                        *branch_layers,
                        nn.Linear(out_dim, num_classes) if 'AngularMargin' not in task['loss'] \
                            else AngularLogit(out_dim, num_classes)
                    )
                )
            elif 'label_range' in task:
                output_layer = [nn.Linear(out_dim, 1)]
                if 'activation' in task:
                    output_layer.append(
                        getattr(nn, task['activation'])()
                    )
                setattr(
                    self, f'head_{head}', 
                    nn.Sequential(
                        *branch_layers,
                        *output_layer
                    )
                )
            self.head_names.append(head)
        
        if 'input_embedding_dim' in self.hp.model \
            and self.hp.model['input_embedding_dim'] is not None:
            input_embed_size = self.hp.model['input_embedding_dim']
            if len(input_embed_size) == 1:
                self.input_embedding = partial(F.one_hot, num_classes=input_embed_size[0])
                self.input_embedding_type = 'one_hot'
            else:
                self.input_embedding = nn.Embedding(
                    input_embed_size[0],
                    input_embed_size[1]
                )
                self.input_embedding_type = 'dense'
        
    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        if hasattr(self, 'input_embedding'):
            if x.dim() == 4:
                x = self.input_embedding(x[:, 0, :, :].long())
                x = x.transpose_(1, 3).transpose_(2, 3)
            elif x.dim() == 2:
                x = self.input_embedding(x[:, :].long())
                if self.core._embed_transpose:
                    x = x.transpose_(1, 2)
        if x.dtype != torch.float32:
            x = x.float()
        feat = self.core(x)
        output = [
            getattr(self, f'head_{head}')(feat) \
                for head in self.head_names
        ]
        return tuple(output)

    def register_hook(self):
        name = 'bottleneck'
        if not isinstance(self.core.bottleneck, nn.Identity):
            # this takes pre-activation bottleneck
            self._hook = self.core.bottleneck[0].register_forward_hook(_get_hook(name))
        else:
            self._hook = self.core.bottleneck.register_forward_hook(_get_hook(name))
        return self
    
    def delete_hook(self):
        if hasattr(self, '_hook'):
            self._hook.remove()

class SmoothingCE(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(SmoothingCE, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

class AngularLogit(nn.Module):
    def __init__(self, in_features, n_classes):
        super(AngularLogit, self).__init__()
        self.fc = nn.Linear(in_features, n_classes, bias=False)
        self.weight = nn.Parameter(
            torch.FloatTensor(n_classes, in_features)
        )
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        '''
        input shape (N, in_features)
        '''
        return F.linear(F.normalize(x, dim=1), F.normalize(self.weight))

class TaskType(Enum):
    CLASSIFICATION = 1
    REGRESSION = 2

class MultiTaskLoss(nn.Module):
    def __init__(self, tasks, task2num_classes=None, learnable_multitask_weights=False):
        super(MultiTaskLoss, self).__init__()
        self.criteria = []
        self.weights = []
        self.names = []
        self.task_type = []
        self.loss_func = []
        self.task2num_classes = task2num_classes
        self._learnable_weights = learnable_multitask_weights
        for key, task in tasks.items():
            self.criteria.append(task['loss'])
            if not self._learnable_weights or len(tasks) == 1:
                self.weights.append(task['weight'] if 'weight' in task else 1.0)
            else:
                # logvar
                setattr(self, 'w{}'.format(key), nn.Parameter(torch.FloatTensor(1).uniform_(0, 1)))
                self.weights.append(getattr(self, 'w{}'.format(key)))
            self.names.append(key)
            self.task_type.append(
                TaskType.CLASSIFICATION if 'classes' in task else \
                    TaskType.REGRESSION)
            if task['loss'] == 'AngularMargin' or task['loss'] == 'AdditiveAngularMargin':
                s = task['loss_config']['scale']
                m = task['loss_config']['margin']
                self.loss_func.append(
                    LogSoftmaxWrapper(AdditiveAngularMargin(
                        scale=s,
                        margin=m
                        ) if task['loss'] == 'AdditiveAngularMargin' else 
                        AngularMargin(scale=s, margin=m))
                )
            elif task['loss'] == 'CrossEntropyLoss':
                smoothing = 0.0
                if 'smoothing' in task:
                    smoothing = task['smoothing']
                if smoothing > 0.0:
                    self.loss_func.append(
                        SmoothingCE(task2num_classes[key], smoothing))
                else:
                    self.loss_func.append(nn.CrossEntropyLoss())
            else:
                self.loss_func.append(
                    getattr(nn, task['loss'])())
        if not self._learnable_weights:
            self.weights = torch.tensor(self.weights) / sum(self.weights)

        
    def is_classification(self, idx_or_name):
        idx = idx_or_name
        if isinstance(idx_or_name, str) and idx_or_name in self.names:
            idx = self.names.index(idx_or_name)
        return self.task_type[idx] == TaskType.CLASSIFICATION

            

    def forward(self, pred, label):
        assert len(pred) == len(label), \
            'number of heads mismatchs between data and model'
        raw_loss = [loss_func(x, y) \
            for loss_func, x, y in zip(self.loss_func, pred, label)]
        loss = None
        if not self._learnable_weights or len(self.loss_func) == 1:
            loss = [l * w \
                for l, w in zip(raw_loss, self.weights)]
        else:
            loss = [l * torch.exp(-w) + w \
                for l, w in zip(raw_loss, self.weights)]

        return (loss, raw_loss)

class AngularMargin(nn.Module):
    def __init__(self, margin=0.0, scale=1.0):
        super(AngularMargin, self).__init__()
        self.m = margin
        self.s = scale

    def forward(self, outputs, targets):
        outputs = outputs - self.m * targets
        return self.s * outputs

class AdditiveAngularMargin(AngularMargin):
    """
    Additive Angular Margin (AAM)
    """

    def __init__(self, margin=0.0, scale=1.0):
        super(AdditiveAngularMargin, self).__init__(margin, scale)

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m
    
    def forward(self, outputs, targets):
        cosine = outputs.float()
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        outputs = (targets * phi) + ((1.0 - targets) * cosine)
        return self.s * outputs

class LogSoftmaxWrapper(nn.Module):
    def __init__(self, loss_fn):
        super(LogSoftmaxWrapper, self).__init__()
        self.loss_fn = loss_fn
        self.criterion = torch.nn.KLDivLoss(reduction="sum")

    def forward(self, outputs, targets):
        outputs = outputs
        targets = targets
        targets = F.one_hot(targets.long(), outputs.shape[1]).float()
        try:
            predictions = self.loss_fn(outputs, targets)
        except TypeError:
            predictions = self.loss_fn(outputs)

        predictions = F.log_softmax(predictions, dim=1)
        loss = self.criterion(predictions, targets) / targets.sum()
        return loss

class TokenTransformer(nn.Module):
    def __init__(self, *, sequence_lens, dim, depth, heads, mlp_dim, pool = 'cls', dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        num_patches = sequence_lens

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim)
        )

    def forward(self, x):
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        return self.mlp_head(x)