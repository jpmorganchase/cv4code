# SPDX-License-Identifier: Apache-2.0
import os
import csv
import torch
import logging
from enum import Enum
import numpy as np
from collections import defaultdict
from torchtext.vocab.vocab_factory import vocab
from torchvision import io, transforms
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import gc
from torch.utils.data._utils.collate import default_collate
from .tokenizer import tokenize
from torchtext.vocab import vocab

logger = logging.getLogger('dataset')

_g_vocab = None

_TASK_GARBAGE_COLLECTOR_KEY = 'OTHERS'

_TASK_FROM_FILE = 'FROM_FILE'

# INCLUDING NULL 96
_PRINTABLE_ASCII_RANGE_DECIMAL = [32, 126]

INTERPOLATION_MODE='BICUBIC'

_CUSTOM_TRANSFORMS = [
    'HalveSize', 
    'TopLeftCrop', 
    'AspectRatioPreservedResize', 
    'TopLeftCropOrResize',
    'CharFrequencyASCII',
    'CharSequenceASCII',
    'CharSequenceASCIIReduceDup',
    'CharSequenceASCIINoDup',
    'PrintableASCIIOnly',
    'DimDrop',
    'RandomDrop',
    'TokenSequence',
    'RemoveEmptyLines',
    ]
# HalveSize
def halve_size(image):
    _, h, w = image.shape
    return transforms.functional.resize(
        image, 
        (int(h/2), int(w/2)), 
        interpolation=getattr(transforms.InterpolationMode, INTERPOLATION_MODE)
    )

# InterleavePad
def interleave_crop(image, size):
    h, w = size
    c, img_h, img_w = image.shape
    pad_h = max(0, h-img_h)
    pad_w = max(0, w-img_w)
    if pad_h > 0:
        if pad_h > img_h:
            # when padding more than the original image
            n_times = pad_h / img_h
            zeros = torch.zeros([c, img_h, img_w])
            image = torch.stack((image, *[zeros] * int(n_times)), dim=2)
            image = image.view(c, img_h * (int(n_times) + 1), img_w)
            pad_h = h - image.shape[1]
            if pad_h > 0:
                image = transforms.functional.pad(image, [0, 0, 0, pad_h], padding_mode='constant')
        elif pad_h < img_h:
            # when padding less than the original image
            zeros = torch.zeros([c, pad_h, img_w])
            padded_segment =  image[:, :pad_h, :]
            padded_segment = torch.stack(
                (image[:, :pad_h, :], zeros), dim=2).view(c, pad_h*2, img_w)
            image = torch.cat((padded_segment, image[:, pad_h:, :]), dim=1)
        else:
            # when the original height is exactly half
            zeros = torch.zeros([c, pad_h, img_w])
            image = torch.stack((image, zeros), dim=2).view(c, img_h + pad_h, img_w)
    if pad_w > 0:
        image = transforms.functional.pad(image, [0, 0, pad_w, 0], padding_mode='constant')
    return transforms.functional.crop(image, 0, 0, h, w)

# TopLeftCrop
def top_left_crop(image, size, mode='edge'):
    if mode == 'interleave':
        return interleave_crop(image, size)
    h, w = size
    _, img_h, img_w = image.shape
    pad_h = max(0, h-img_h)
    pad_w = max(0, w-img_w)
    if pad_h > 0 or pad_w > 0:
        image = transforms.functional.pad(image, [0, 0, pad_w, pad_h], padding_mode=mode)
    return transforms.functional.crop(image, 0, 0, h, w)

# AspectRatioPreservedResize - decrecated
def aspect_ratio_preserved_resize(image, size):
    h, w = size
    target_ar = h / w
    _, img_h, img_w = image.shape
    src_ar = img_h/img_w
    if target_ar > src_ar:
        t_img_h = int(target_ar * img_w)
        image = transforms.functional.pad(
            image, [0, 0, 0, abs(t_img_h-img_h)], 
            padding_mode='edge')
    elif target_ar < src_ar:
        t_img_w = int(img_h / target_ar)
        image = transforms.functional.pad(
            image, [0, 0, abs(t_img_w-img_w), 0], 
            padding_mode='edge')
    return transforms.functional.resize(
        image, 
        (h, w), 
        interpolation=getattr(transforms.InterpolationMode, INTERPOLATION_MODE)
    )

# AspectRatioPreservedResize
def efficient_aspect_ratio_preserved_resize(image, size):
    h, w = size
    _, img_h, img_w = image.shape

    target_ar = h / w
    src_ar = img_h / img_w
        
    if target_ar > src_ar:
        ratio = w / img_w
        image = transforms.functional.resize(
            image, 
            (int(img_h * ratio), w), 
            interpolation=transforms.InterpolationMode.BICUBIC)
        image = transforms.functional.pad(
            image, 
            [0, 0, 0, abs(h-int(img_h*ratio))], 
            padding_mode='edge')
    elif target_ar < src_ar:
        ratio = h / img_h
        image = transforms.functional.resize(
            image, 
            (h, int(img_w * ratio)), 
            interpolation=transforms.InterpolationMode.BICUBIC)
        image = transforms.functional.pad(
            image, 
            [0, 0, abs(w-int(img_w*ratio)), 0], 
            padding_mode='edge')
    else:
        if img_h != h or img_w != w:
            image = transforms.functional.resize(
                image, 
                (h, w), 
                interpolation=transforms.InterpolationMode.BICUBIC)

    return image

# TopLeftCropOrResize
def top_left_crop_or_resize(image, size, mode='edge'):
    h, w = size
    _, img_h, img_w = image.shape
    if img_h > h or img_w > w:
        image = top_left_crop(image, size, mode)
    else:
        image = aspect_ratio_preserved_resize(image, size)
    return image

def ascii_char_frequency(image, minval=1, maxval=129, norm_type='l2'):
    feat = torch.histc(image.float(), bins=maxval-minval+1, min=minval, max=maxval)
    if norm_type == 'l2':
        feat = feat / feat.norm(2)
    elif norm_type == 'l1':
        feat = feat / feat.norm(1)
    elif norm_type == '':
        pass
    feat[torch.isnan(feat)] = 0
    return feat

def token_sequence(image):
    assert _g_vocab is not None, 'vocabulary is None, token_sequence requires precomputed vocabulary'
    seq = ascii_char_sequence(image, note_newline=10) #note it newline control
    return torch.LongTensor(_g_vocab(tokenize(''.join(chr(codepoint) for codepoint in seq))))

def ascii_printable_char(image, rebase=False):
    image = image.long()
    base = 0 if not rebase else _PRINTABLE_ASCII_RANGE_DECIMAL[0] - 1
    # replace tab with a space
    image[image==9] = 32
    image = torch.where(
            (image >= _PRINTABLE_ASCII_RANGE_DECIMAL[0]) & (image <= _PRINTABLE_ASCII_RANGE_DECIMAL[1]),
            image, base
    ) - base
    return image

def ascii_remove_empty_lines(image, rebased=False):
    minval = 1 if rebased else 32
    return image[(image > minval).sum(dim=-1) > 0, :].unsqueeze_(0)

def ascii_dim_drop(image, prob=0.5, dim=1):
    image = image.long()
    other_dim = (dim + 1) % 3
    tile_dim = list(image.shape)
    tile_dim[dim] = 1
    drop_mask = torch.rand(image.shape[dim]).unsqueeze(0).unsqueeze(other_dim).tile((tile_dim))
    return torch.where(drop_mask > prob, image, 0)

def ascii_random_drop(image, prob=0.5):
    drop_mask = torch.rand(image.shape)
    return torch.where(drop_mask > prob, image, 0)

def ascii_char_sequence_reduced_dup(image, to_remove=[0]):
    in_tensor = image
    if in_tensor.dim() > 1:
        in_tensor = in_tensor.reshape(-1)
    out_tensor = [in_tensor[0]]
    for idx in range(1, len(in_tensor)):
        if in_tensor[idx-1] != in_tensor[idx] or in_tensor[idx] not in to_remove:
            out_tensor.append(in_tensor[idx])
    return torch.LongTensor(out_tensor)

def ascii_char_sequence_no_dup(image):
    out_tensor = image
    if out_tensor.dim() > 1:
        out_tensor = out_tensor.reshape(-1)
    return torch.unique_consecutive(out_tensor[torch.nonzero(out_tensor)], dim=-1).squeeze_()

def ascii_char_sequence(image, note_newline=-1):
    if note_newline > 0:
        new_line = torch.ones([image.shape[0], image.shape[1], 1], dtype=torch.long) * note_newline
        image = torch.cat([image, new_line], dim=2)
    out_tensor = image.reshape(-1)
    return out_tensor[torch.nonzero(out_tensor)].squeeze_()

def generate_data_transforms(hparams, is_training):
    transform_ops = hparams.train_transform_ops if is_training else \
        hparams.val_transform_ops
    transform_list = []
    for x, y in transform_ops:
        if x in _CUSTOM_TRANSFORMS:
            if x == 'TopLeftCrop':
                tl_crop_arg = y
                transform_list.append(
                    transforms.Lambda(
                        lambda x : top_left_crop(x, *tl_crop_arg if isinstance(tl_crop_arg, list) else tl_crop_arg)
                        )
                )
            elif x == 'AspectRatioPreservedResize':
                size=y
                transform_list.append(
                    transforms.Lambda(
                        lambda x : efficient_aspect_ratio_preserved_resize(x, size)
                        )
                )
            elif x == 'HalveSize':
                transform_list.append(
                    transforms.Lambda(
                        lambda x : halve_size(x)
                        )
                )
            elif x == 'TopLeftCropOrResize':
                tlr_crop_arg = y
                transform_list.append(
                    transforms.Lambda(
                        lambda x : top_left_crop_or_resize(
                            x, 
                            *tlr_crop_arg if isinstance(tlr_crop_arg , list) else tlr_crop_arg )
                        )
                )
            elif x == 'CharFrequencyASCII':
                char_freq_arg = y
                transform_list.append(
                    transforms.Lambda(
                        lambda x :  ascii_char_frequency(x, *char_freq_arg)
                        )
                )
            elif x == 'CharSequenceASCII':
                char_seq_arg = y
                transform_list.append(
                    transforms.Lambda(
                        lambda x :  ascii_char_sequence(x, char_seq_arg)
                        )
                )
            elif x == 'CharSequenceASCIINoDup':
                transform_list.append(
                    transforms.Lambda(
                        lambda x :  ascii_char_sequence_no_dup(x)
                        )
                )
            elif x == 'CharSequenceASCIIReduceDup':
                char_seq_rdup_arg = y if y is not None else None
                transform_list.append(
                    transforms.Lambda(
                        lambda x :  ascii_char_sequence_reduced_dup(x, char_seq_rdup_arg) 
                        )
                )
            elif x == 'PrintableASCIIOnly':
                rebase = y if y is not None else False
                transform_list.append(
                    transforms.Lambda(
                        lambda x : ascii_printable_char(x, rebase)
                        )
                )
            elif x == 'RemoveEmptyLines':
                rebased = y
                transform_list.append(
                    transforms.Lambda(
                        lambda x : ascii_remove_empty_lines(x, rebased=rebased)
                        )
                )
            elif x == 'TokenSequence':
                transform_list.append(
                    transforms.Lambda(
                        lambda x : token_sequence(x)
                        )
                )
            elif x == 'DimDrop':
                arg = y
                transform_list.append(
                    transforms.Lambda(
                        lambda x : ascii_dim_drop(x, *arg)
                        )
                )
            elif x == 'RandomDrop':
                prob = y
                transform_list.append(
                    transforms.Lambda(
                        lambda x : ascii_random_drop(x, prob)
                        )
                )
        else:
            if x == 'ConvertImageDtype':
                y = getattr(torch, y)
            if isinstance(y, list):
                transform_list.append(getattr(transforms, x)(*y))
            elif y is None:
                transform_list.append(getattr(transforms, x)())
            else:
                transform_list.append(getattr(transforms, x)(y))
    transforms_composed = transforms.Compose(transform_list)

    if len(transform_ops) == 0:
        transforms_composed = None
    return transforms_composed


def generate_target_transforms(task_descriptor, categories=None):
    """
    task_descriptor: dict
    """
    # classification task
    if 'classes' in task_descriptor:
        if categories is None:
            categories = task_descriptor['classes']
        else:
            assert task_descriptor['classes'] == _TASK_FROM_FILE, f'invalid task descriptor {task_descriptor}'
        transform_map = dict()
        if isinstance(task_descriptor['classes'][0], list):
            for idx, group in enumerate(task_descriptor['classes']):
                for cls_name in group:
                    transform_map[cls_name] = torch.tensor(idx)
        else:
            transform_map = {x: torch.tensor(idx) for idx, x in enumerate(categories)}
        def target_transform(category):
            if category not in categories and _TASK_GARBAGE_COLLECTOR_KEY in transform_map:
                category = _TASK_GARBAGE_COLLECTOR_KEY
            return transform_map[category]
        return target_transform
    elif 'label_range' in task_descriptor:
        scale = 1.0
        if 'label_scale' in task_descriptor:
            scale = task_descriptor['label_scale']
        if 'label_divide_scale' in task_descriptor:
            scale = 1.0 / task_descriptor['label_divide_scale']
        lower, upper = task_descriptor['label_range']
        def target_transform(value):
            v = torch.tensor(float(value), dtype=torch.float32) * scale
            return torch.clamp(v, min=lower, max=upper).view([1])
        return target_transform
    else:
        raise NotImplementedError(f'task descriptor not recognised - {task_descriptor}')
    return None
    
def process_codenet_csv(csv_filepath, tasks=None, transforms=None, langauges=None):
    dataset = defaultdict(dict)
    with open(csv_filepath, 'r') as fd:
        reader = csv.DictReader(fd)
        if tasks is None:
            tasks = [x for x in reader.fieldnames if x != 'submission_id']
        elif isinstance(tasks, dict):
            tasks = list(tasks.keys())
        for row in reader:
            if langauges is not None and row['language'] not in langauges:
                continue
            for task in tasks:
                if transforms is not None:
                    dataset[row['submission_id']][task] = transforms[task](row[task])
                else:
                    dataset[row['submission_id']][task] = row[task]
    logger.info(f'parsed {len(dataset)} data points from {csv_filepath}')
    return dataset, tasks

class DatasetMode(Enum):
    INVALID = 0
    TRAIN = 1
    VAL = 2
    TEST = 3

class CodeImageDataset(Dataset):
    def __init__(self, hparams, mode='train', raw_label=False, image_dir='', **kargs):
        super(CodeImageDataset, self).__init__()
        self._mode = DatasetMode.INVALID
        self._csv = ''
        if mode == 'train':
            self._mode = DatasetMode.TRAIN
            self._csv = 'train.csv'
        elif mode == 'validation':
            self._mode = DatasetMode.VAL
            self._csv = 'val.csv'
        elif mode == 'test':
            self._mode = DatasetMode.TEST
            self._csv = 'test.csv'
        else:
            logger.error('invalid mode received, options : train, validation and test')
        self._is_training = True if self._mode == DatasetMode.TRAIN else False
        hp = hparams
        self.image_dir = image_dir
        self.data_transforms = generate_data_transforms(hp, self._is_training)
        self.heads = None if raw_label else np.array([target for target in hp.tasks.keys()])
        self.raw_label = raw_label
        self.task2num_classes = None
        dataset_name = hp.dataset_name
        if hasattr(self, 'process_' + dataset_name):
            self.task2num_classes = getattr(self, 'process_' + dataset_name)(**kargs, hp=hp)
        else:
            raise NotImplementedError(f'CodeImageDataset does not have handler for {dataset_name}')
        logger.info(f'Dataset={dataset_name}, mode={self._mode}, {len(self.datapoints)}')
    
    def get_task2num_classes(self):
        task2num_classes = self.task2num_classes
        delattr(self, 'task2num_classes')
        return task2num_classes

    def process_codenet(self, dataset_dir, hp):
        """
        a dataset handler for CodeNet
        """
        if os.path.isfile(os.path.join(dataset_dir, 'vocab')):
            logger.info(f'loading vocabulary')
            self.load_g_vocab(os.path.join(dataset_dir, 'vocab'))
        task2num_classes = dict()
        csvname = self._csv
        csvpath = os.path.join(dataset_dir, csvname)
        if self.raw_label and os.path.isfile(os.path.join(dataset_dir, 'test.csv')):
            csvpath = os.path.join(dataset_dir, 'test.csv')
        logger.info(f'Loading labels from {csvpath}')
        label_lookup = dict()
        if hasattr(hp, 'tasks'):
            for task_name in hp.tasks.keys():
                path = os.path.join(dataset_dir, f'{task_name}.lbl')
                if os.path.isfile(path):
                    logger.info(f'Loading class target {task_name} from {path}')
                    with open(path, 'r') as fd:
                        categories = [clas.strip() for clas in fd]
                        label_lookup[task_name] = categories
                        task2num_classes[task_name] = len(categories)
                else:
                    label_lookup[task_name] = None
                    if 'classes' in hp.tasks[task_name]:
                        task2num_classes[task_name] = len(hp.tasks[task_name]['classes'])

        init_target_transforms = None
        if not self.raw_label:
            init_target_transforms = {
                task_name : generate_target_transforms(task, label_lookup[task_name]) \
                    for task_name, task in hp.tasks.items()
            }

        logger.info(f'Loading and transforming label from {csvpath}')
        metadata, heads = process_codenet_csv(
            csvpath, 
            hp.tasks if not self.raw_label else None, 
            init_target_transforms, 
            hp.codenet_keep_languages if hasattr(hp, 'codenet_keep_languages') else None
            )

        if not os.path.isdir(dataset_dir):
            raise FileNotFoundError(f'{dataset_dir} does not exist')
        datapoints = list()
        with open(os.path.join(dataset_dir, 'image2id'), 'r') as fd:
            for line in fd:
                fields = line.strip().split()
                path = fields[0]
                dataid = fields[1]
                if len(fields) > 2:
                    path = ' '.join(fields[:-1])
                    dataid = fields[-1]
                path = path[1:] if path.startswith('/') and self.image_dir != '' else path
                path = os.path.join(self.image_dir, path)
                if dataid in metadata and os.path.exists(path):
                    datapoints.append(
                        (path, *[metadata[dataid][head] for head in heads])
                    )
        # memory issue as reported https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        self.datapoints = np.array(datapoints, dtype=object)
        self.heads = np.array(heads)
        gc.collect()
        return task2num_classes
    
    @classmethod
    def load_g_vocab(cls, vocab_filepath):
        global _g_vocab
        with open(vocab_filepath, 'r') as fd:
            # newline char at the end, strip it but keep if the token itself is a newline
            vocab_dict = {word[:-1] : idx for idx, word in enumerate(fd.readlines())}
            _g_vocab = vocab(vocab_dict, min_freq=0)
            _g_vocab.set_default_index(_g_vocab['<unk>'])
            logger.info(f'Vocabulary size: {len(_g_vocab)}')
    
    def __len__(self):
        return len(self.datapoints)
    
    def __getitem__(self, idx):
        data = self.datapoints[idx]
        path = data[0]
        
        transform = self.data_transforms

        image = None        
        if path.split('.')[-1] == 'npy':
            image = transform(torch.from_numpy(np.load(path))) \
             if transform is not None else np.load(path)
        else:
            image = transform(io.read_image(path)) \
                if transform is not None else io.read_image(path)
        if image is None:
            raise RuntimeError('no image is loaded')
        return image, data[1:].tolist()

def pad_collate(pad_val=0, max_length=-1):
    if isinstance(pad_val, str):
        assert _g_vocab is not None, 'padding using vocab, vocab cannot be empty'
        pad_val = _g_vocab([pad_val])[0]
    def collate_func(batch):
        x, y = zip(*batch)
        x_limit = [] if max_length > 0 else x
        if max_length > 0:
            for x_ in x:
                try:
                    if x_.shape[0] < max_length:
                        x_limit.append(x_)
                    else:
                        x_limit.append(x_[:max_length])
                except IndexError:
                    x_limit.append((torch.ones(max_length) * pad_val).long())
        x_pad = pad_sequence(x_limit, batch_first=True, padding_value=pad_val)
        return x_pad, default_collate(y)
    return collate_func

def image_pad_collate(
        min_size=(24, 12), 
        max_size=(224, 224), 
        percentile=100, 
        square=False, 
        mode='constant', 
        infer=False):
    assert percentile <= 100 and percentile > 0
    def collate_func(batch):
        if infer:
            x = batch
        else:
            x, y = zip(*batch)
        max_h, max_w = max_size
        min_h, min_w = min_size
        batch_h, batch_w = [], []
        for x_ in x:
            _, img_h, img_w = x_.shape
            batch_h.append(img_h)
            batch_w.append(img_w)
        if percentile == 100:
            batch_h = max(batch_h)
            batch_w = max(batch_w)
        else:
            batch_h = int(np.percentile(batch_h, percentile))
            batch_w = int(np.percentile(batch_w, percentile))
        batch_h = min(max(batch_h, min_h), max_h)
        batch_w = min(max(batch_w, min_w), max_w)
        if square:
            max_hw = max(batch_h, batch_w)
            batch_h = batch_w = max_hw 
        padded_x = [top_left_crop(x_, (batch_h, batch_w), mode) for x_ in x]
        if infer:
            return torch.stack(padded_x, 0)
        return torch.stack(padded_x, 0), default_collate(y)
    return collate_func