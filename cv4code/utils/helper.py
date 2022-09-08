# SPDX-License-Identifier: Apache-2.0
from genericpath import exists
import os
import sys
import json
import math
import torch
import shutil
import logging

logger = logging.getLogger('helper')

_MODEL_PREFIX = 'model-'
_MODEL_SUFFIX= '.pt'

def clean_checkpoints(ckpt_dir, max_to_save, skip_step=None):
    model_indices = [x.split('-')[1].split('.')[0] for x in os.listdir(ckpt_dir)]
    models = []
    for idx in model_indices:
        try:
            models.append(int(idx))
        except ValueError:
            pass
    models = sorted(models)

    if len(models) < max_to_save:
        return

    for idx in range(len(models) - max_to_save):
        if skip_step and models[idx] == skip_step:
            continue
        os.remove(os.path.join(ckpt_dir, f'{_MODEL_PREFIX}{models[idx]}{_MODEL_SUFFIX}'))

def logging_handle(logdir=None, exclude=[]):
    all_loggers = [logging.getLogger(x) for x in logging.root.manager.loggerDict if x not in exclude]
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh = None
    if logdir is not None:
        fh = logging.FileHandler(os.path.join(logdir, 'train.log'))
    ch = logging.StreamHandler(sys.stdout)
    for logger in all_loggers:
        logger.propagate = False
        logger.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)
        if fh is not None:
            fh.setFormatter(formatter)
            fh.setLevel(logging.DEBUG)
            logger.addHandler(fh)

def save_model(model, ckpt_dir, step, log_dir=None, epoch=None, name=None):
    os.makedirs(ckpt_dir, exist_ok=True)
    name = _MODEL_PREFIX if not name else name
    mdl_path= os.path.join(ckpt_dir, f'{name}{step}{_MODEL_SUFFIX}')
    torch.save(model.state_dict(), mdl_path)
    logger.debug(f'Saved checkpoint at step {step} to {mdl_path}')
    
    if log_dir is not None and epoch is not None:
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, 'progress.json'), 'w') as fd:
            json.dump({'epoch': epoch, 'step': step}, fd, indent=4)
    logger.debug(f'Updated progress log')

def load_model(model, ckpt_dir, step=None, skip=None):
    if step is None:
        # load the last checkpoint
        steps = [mdl.split('-')[1].split('.')[0] for mdl in os.listdir(ckpt_dir)]
        if 'optimal' in steps:
            step = 'optimal'
        else:
            step = max(steps)
    kargs = dict()
    if not torch.cuda.is_available():
        kargs['map_location'] = torch.device('cpu')
    ckpt_state_dict = torch.load(
            os.path.join(ckpt_dir, f'{_MODEL_PREFIX}{step}{_MODEL_SUFFIX}'),
            **kargs
            )
    model_state_dict = model.state_dict()
    for key, tensor in ckpt_state_dict.copy().items():
        if key not in model_state_dict:
            ckpt_state_dict.pop(key)
            logger.warning(f'checkpointed {key} do not exist in model, skipped')
            continue
        if tensor.shape != model_state_dict[key].shape:
            ckpt_state_dict.pop(key)
            logger.warning(f'checkpointed {key} mismatch in shape with that in the model, skipped')
            continue
        if skip:
            for node in skip:
                if key.startswith(node):
                    ckpt_state_dict.pop(key)
                    logger.warning(f'checkpointed {key} skipped from reloading')
    model.load_state_dict(ckpt_state_dict, strict=False)
    logger.info(f'Loading model checkpoint from step {step}')
    return step

def track_best_model(log_dir, metric_key, min_max='min', new_entry=None):
    """
    new entry: 
        dict: {'loss': float, epoch: int, step: int}
    """
    
    logfile = os.path.join(log_dir, 'best_model.json') 
    info = {'progress': [], 'best_model': dict()}
    try:
        if os.path.isfile(logfile):
            with open(logfile, 'r') as fd:
                info = json.load(fd)
    except json.JSONDecodeError:
        logger.warning('corrupted progress json found, skip')
    
    best_loss = math.inf if min_max == 'min' else -math.inf
    best_step = 0
    best_epoch = 0
    steps_since_last_impr = 0

    try:
        best_model = info['best_model']
        best_loss = best_model[metric_key]
        best_step = best_model['step']
        steps_since_last_impr = best_model['steps_since_last_improvement']
        best_epoch = best_model['epoch']
    except KeyError:
        logger.warning('Best model tracking has no historical data (ok if this is the first ckpt)')
    
    if new_entry is not None:
        new_entry[metric_key] = float(new_entry[metric_key])
        info['progress'].append(
            new_entry
        )
        loss = new_entry[metric_key]
        if (min_max == 'min' and loss <= best_loss) or \
            (min_max == 'max' and loss >= best_loss):
            best_step = new_entry['step']
            best_epoch = new_entry['epoch']
            best_loss = loss
        steps_since_last_impr = new_entry['step'] - best_step
    
    with open(logfile, 'w') as fd:
        info['best_model'] = {
            metric_key: float(best_loss),
            'step': best_step,
            'epoch': best_epoch,
            'steps_since_last_improvement': steps_since_last_impr
        }
        json.dump(info, fd)
        
    return steps_since_last_impr, info['best_model']

def keep_best_model(ckpt_dir, step):
    src_ckpt = os.path.join(ckpt_dir, f'{_MODEL_PREFIX}{step}{_MODEL_SUFFIX}')
    trgt_ckpt = os.path.join(ckpt_dir, f'{_MODEL_PREFIX}optimal{_MODEL_SUFFIX}')
    if os.path.exists(src_ckpt):
        shutil.copy2(src_ckpt, trgt_ckpt)
        
def tensorboard_loggable_dict(hparams):
    filtered_dict = {}
    allowed_types = [int, float, str, bool]
    for key, value in hparams.__dict__.items():
        for t in allowed_types:
            if isinstance(value, t):
                filtered_dict[key] = value
            else:
                filtered_dict[key] = f'{value}'
    return filtered_dict