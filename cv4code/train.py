# SPDX-License-Identifier: Apache-2.0
import os
import sys
import json
import random
import logging
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from time import time
from torch.utils.data import DataLoader
from modules.model import CV4code, MultiTaskLoss, get_bottleneck
from modules.hparams import YHparams
from modules.dataset import CodeImageDataset, pad_collate, image_pad_collate
from modules.lr_scheduler import *
import torch.utils.tensorboard as tensorboard
import torchmetrics
from utils.helper import *

logger = logging.getLogger('train')

# frequency of console logging
_LOGGING_INTERVAL = int(os.getenv('LOGGING_INTERVAL', '100'))
# maximum number of checkpoints to keep 
# (to prevent from running out of disk space)
_MAX_NUM_CKPTS = int(os.getenv('MAX_NUM_CKPTS', '50'))

# the number of data points stored for projection
_MAX_NUM_PROJECTOR = int(os.getenv('MAX_NUM_PROJECTOR', '1000'))

    
def validation(tasks, model, dataloader, criterion, device,
    metric_computers=None, do_auroc=False, extra_metric_computers=None):
    # test_metric_computeres: [(task_name, computer), ...]
    model.eval()
    metrics = dict()
    running_losses = []
    embedding_list = []
    embedding_label_list = []

    auroc = dict()
    if metric_computers:
        for tsk_name, computer in metric_computers.items():
            computer.reset()
            if do_auroc and 'classes' in tasks[tsk_name]:
                auroc[tsk_name] = torchmetrics.AveragePrecision(
                    num_classes=model.task2num_classes[tsk_name],
                    compute_on_step=False)
    if extra_metric_computers:
        for tsk_name, computer in extra_metric_computers:
            computer.reset()
        
    model.register_hook()
    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device)
            labels = [label.to(device) for label in labels]
            output = model(data)
            _, losses = criterion(output, labels)
            running_losses.append([x.detach().item() for x in losses])
            label_strings = [''] * data.shape[0]
            if metric_computers:
                for idx, (pred, label) in enumerate(zip(output, labels)):
                    pred, label = pred.cpu(), label.cpu()
                    metric_computers[criterion.names[idx]](pred, label)
                    if extra_metric_computers:
                        for tsk_name, computer in extra_metric_computers:
                            if tsk_name == criterion.names[idx]:
                                computer(pred, label)
                    if do_auroc and criterion.names[idx] in auroc:
                        auroc[criterion.names[idx]](pred, label)
                    if len(embedding_list) < _MAX_NUM_PROJECTOR:
                        for point_idx, point_lbl in enumerate(label):
                            label_strings[point_idx] = label_strings[point_idx]  + \
                                criterion.names[idx] + '_' + str(point_lbl.item()) + ','

            if len(embedding_list) < _MAX_NUM_PROJECTOR:
                bottleneck = get_bottleneck()
                if bottleneck is None:
                    continue
                embedding_list.extend([x for x in get_bottleneck().cpu().numpy()])
                embedding_label_list.extend(label_strings)


    for tsk_id, tsk_name in enumerate(criterion.names):
        key = f'loss_{tsk_name}'
        loss = np.mean([loss[tsk_id] for loss in running_losses])
        metrics[key] = loss
    
    weighted_loss = np.mean([sum(losses) for losses in running_losses])
    metrics['loss'] = weighted_loss

    if metric_computers:
        for tsk_name, computer in metric_computers.items():
            tp = type(computer).__name__
            if tp == 'Accuracy' and computer.top_k:
                tp += f'_top{computer.top_k}'
            metrics[f'{tp}_{tsk_name}'] = computer.compute()
            computer.reset()
            if do_auroc and tsk_name in auroc:
                metrics[f'auroc_{tsk_name}'] = auroc[tsk_name].compute()
    if extra_metric_computers:
        for tsk_name, computer in extra_metric_computers:
            tp = type(computer).__name__
            if tp == 'Accuracy' and computer.top_k:
                tp += f'_top{computer.top_k}'
            metrics[f'{tp}_{tsk_name}'] = computer.compute()
            computer.reset()

    model.delete_hook()
    if len(embedding_list) == 0:
        return metrics, None
    return metrics, \
        {
            'embed': np.array(embedding_list), 
            'label': embedding_label_list
        }

def train(args):
    hp = YHparams(args.hparams, args.tag)
    torch.manual_seed(hp.seed)
    np.random.seed(hp.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(hp.seed)
    random.seed(hp.seed)
    exp_dir = os.path.join('exp', args.tag)
    log_dir = os.path.join(exp_dir, 'logs')
    ckpt_dir = os.path.join(exp_dir, 'checkpoints')
    event_dir = os.path.join(exp_dir, 'events')
    for x in [exp_dir, log_dir, ckpt_dir, event_dir]:
        os.makedirs(x, exist_ok=True)

    logging_handle(log_dir)
    tsb_writer = tensorboard.SummaryWriter(event_dir)
    tsb_writer.add_hparams(
        tensorboard_loggable_dict(hp), {})

    train_set = CodeImageDataset(
        hp, 
        mode='train', 
        image_dir=args.imagedir,
        dataset_dir=args.dataset_dir,
    )
    task2num_classes = train_set.get_task2num_classes()

    val_set = CodeImageDataset(
        hp, 
        mode='validation', 
        image_dir=args.imagedir,
        dataset_dir=args.dataset_dir
    )

    model = CV4code(hp, task2num_classes=task2num_classes)
    model.train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        logger.info(f'DataParallel on {torch.cuda.device_count()} GPUs')
    model.to(device)

    macs = 'unknown'
    try:
        input_size = None
        if 'input_embedding_dim' in hp.model:
            # sequence model
            if hp.image_size: 
                input_size = (hp.model['input_embedding_dim'][-1], hp.image_size[1][0], hp.image_size[1][1])
            else:
                input_size = (hp.model['input_embedding_dim'][-1], max(hp.sequence_max_length, 1))
                if hp.sequence_max_length < 1:
                    logger.warning(f'sequence_max_length is set to < 1, MACs assumed with length 1')
        else:
            # dense model
            input_size = (hp.model['layers'][0][1][0],)
    except Exception:
        logger.warning(f'MACs computation failed')
        pass
    logger.info(f'Total #params (mil):  \t {sum(x.numel() for x in model.parameters())/1e6}')
    logger.info(f'Train #params (mil):  \t {sum(x.numel() for x in model.parameters() if x.requires_grad)/1e6}')
    logger.info(f'Compute device:       \t {device}')
    logger.info(f'Logging interval:     \t {_LOGGING_INTERVAL}')
    logger.info(f'Checkpoint interval:  \t {args.n_ckpt_steps}')
    logger.info(f'#Checkpoints:         \t {_MAX_NUM_CKPTS}')

    if hp.sequence_padding and hp.image_size:
        print('sequence_padding and image_size can not be specified at the same time', file=sys.stderr)

    collate_func = pad_collate(
        hp.sequence_padding_value,
        hp.sequence_max_length) if hp.sequence_padding else None
    if hp.image_size is not None:
        collate_func = image_pad_collate(
            hp.image_size[0],
            hp.image_size[1],
            percentile=hp.image_size[2] if len(hp.image_size) > 2 else 100,
            square=hp.image_size[3] if len(hp.image_size) > 3 else False,
            mode=hp.image_size[4] if len(hp.image_size) > 4 else 'constant',
            )
    train_loader = DataLoader(
        train_set, 
        hp.batch_size, 
        shuffle=True, 
        num_workers=args.n_loader_jobs,
        collate_fn=collate_func,
        pin_memory=True if torch.cuda.is_available() else False
    )

    test_collate_func = collate_func
    if hasattr(hp, 'test_image_size'):
        test_collate_func = image_pad_collate(
            hp.test_image_size[0],
            hp.test_image_size[1],
            percentile=hp.test_image_size[2] if len(hp.test_image_size) > 2 else 100,
            square=hp.test_image_size[3] if len(hp.test_image_size) > 3 else False,
            mode=hp.test_image_size[4] if len(hp.test_image_size) > 4 else 'constant',
            )

    val_loader = DataLoader(
        val_set, 
        hp.batch_size if not hasattr(hp, 'test_batch_size') else hp.test_batch_size, 
        shuffle=False, 
        num_workers=args.n_loader_jobs,
        collate_fn=test_collate_func,
        pin_memory=True if torch.cuda.is_available() else False
    )

    criterion = MultiTaskLoss(
        hp.tasks, 
        task2num_classes, 
        learnable_multitask_weights=hp.learnable_multitask_weights
        )

    metric_computers = {}
    for idx, name in enumerate(criterion.names):
        metric_computers[name] = torchmetrics.Accuracy() \
            if criterion.is_classification(idx) else torchmetrics.MeanAbsoluteError()
    
    extra_test_metrics = None
    if hasattr(hp, 'extra_test_metrics'):
        extra_test_metrics = list()
        for metric in hp.extra_test_metrics:
            extra_test_metrics.append(
                (metric[0], getattr(torchmetrics, f'{metric[1]}')(**metric[2]))
            )

    metric_selector = 'loss' if not hasattr(hp, 'metric_selection') \
        else hp.metric_selection['name']
    metric_selector_min_max = 'min' if not hasattr(hp, 'metric_selection') \
        else hp.metric_selection['min_max']
    
    kargs = {'momentum' : hp.momentum} \
        if hasattr(hp, 'momentum') and hp.optimiser == 'SGD' else {}
    params = list(model.parameters())
    if hp.learnable_multitask_weights:
        criterion.to(device)
        params += list(criterion.parameters())

    optimiser = getattr(
        optim, 
        hp.optimiser)(
            params,
            lr=hp.lr_scheduler['config']['base_lr'],
            weight_decay=hp.lr_scheduler['weight_decay'],
            **kargs
        )


    early_stopping_tolerance_step = hp.lr_scheduler['epoch_stopping_tolerance']
    lr_scheduler = None
    steps_per_epoch = np.ceil(len(train_set) / hp.batch_size)
    if hp.lr_scheduler['strategy'] == 'StepLRScheduler':
        lr_scheduler = StepLRScheduler(
            optim=optimiser, 
            steps_per_epoch=steps_per_epoch if hp.lr_scheduler['step_size_in_epoch_base'] else None,
            **hp.lr_scheduler['config'])
    elif hp.lr_scheduler['strategy'] == 'CyclicLRScheduler':
        lr_scheduler = CyclicLRScheduler(
            optim=optimiser, 
            steps_per_epoch=steps_per_epoch if hp.lr_scheduler['step_size_in_epoch_base'] else None,
            **hp.lr_scheduler['config'])
    elif hp.lr_scheduler['strategy'] == 'CosineAnnealingLRScheduler':
        lr_scheduler = CosineAnnealingLRScheduler(
            optim=optimiser, 
            total_epochs=hp.epochs,
            steps_per_epoch=steps_per_epoch,
            **hp.lr_scheduler['config'])
    else:
        logger.error(f'lr_scehduler.strategy {hp.lr_scheduler["strategy"]} is unknown')
        exit(1)
    running_losses = None
    epoch = 0
    step = 0

    if os.path.exists(os.path.join(exp_dir, 'progress.json')):
        with open(os.path.join(exp_dir, 'progress.json'), 'r') as fd:
            progress = json.load(fd)
            epoch = progress['epoch']
            step = progress['step']
        load_model(model, ckpt_dir, step, hp.skip_from_checkpoints)
        # in case we're recovering from a previous run, recover the learning schedule
    if lr_scheduler and args.recover_lr:
        for _ in range(step):
            lr_scheduler.step()
    logger.info('Start training loops')
    graph_added = False
    optimiser.zero_grad()
    optimiser.step()
    early_stopped = False
    time_1 = time()
    torch.autograd.set_detect_anomaly(True)
    d_dur, m_dur = 0.0, 0.0
    while epoch < hp.epochs and not early_stopped:
        for batch_index, (data, labels) in enumerate(train_loader):
            if not model.training:
                model.train()
            data = data.to(device)
            labels = [label.to(device) for label in labels]
            optimiser.zero_grad()
            time_2 = time()
            # forward pass
            output = model(data)
            # compute loss
            weighted_losses, losses = criterion(output, labels)
            loss = sum(weighted_losses)
            # backward pass
            loss.backward()
            time_3 = time()
            # update weights
            if hp.gradient_max_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=hp.gradient_max_norm)
            optimiser.step()
            step += 1

            if not graph_added:
                tsb_writer.add_graph(model, data)
                graph_added = True

            with torch.no_grad():
                running_losses = 0.9 * running_losses + 0.1 * np.array([x.detach().item() for x in losses]) \
                    if running_losses is not None else np.array([x.detach().item() for x in losses])
                for idx, (pred, label) in enumerate(zip(output, labels)):
                    metric_computers[criterion.names[idx]](pred.cpu(), label.cpu())

            d_dur += time_2 - time_1
            m_dur += time_3 - time_2
            if step % _LOGGING_INTERVAL == 0 or batch_index == len(train_loader) - 1:
                d_dur /= _LOGGING_INTERVAL
                m_dur /= _LOGGING_INTERVAL
                logstr = (f'[epoch {epoch}, step {step}, lr {lr_scheduler.get_last_lr():.1e}] '
                          f'({d_dur:.1f}s/{m_dur:.1f}s) '
                          f'TRN loss : {sum(running_losses):.5f}')
                if hp.learnable_multitask_weights and len(criterion.weights) > 1:
                    logstr += ', ' + \
                         '|'.join(([
                             f'{x}:{np.exp(-torch.squeeze(y).detach().item()):.2f}' for x, y in zip(criterion.names, criterion.weights)]))
                d_dur, m_dur = 0.0, 0.0
                tsb_writer.add_scalars(
                    'loss', {'train': sum(running_losses)}, step)
                for tsk_loss, name in zip(running_losses, criterion.names):
                    tsb_writer.add_scalars(
                        f'loss_{name}', {'train': tsk_loss}, step)
                    # if not criterion.is_classification(name):
                    #     logstr += f', {name} loss : {tsk_loss:5f}'
                for tsk_name, computer in metric_computers.items():
                    tp = type(computer).__name__
                    metric = computer.compute()
                    logstr += f', {tsk_name} {tp} : {metric:.5f}'
                    tsb_writer.add_scalars(
                        f'{tp}_{tsk_name}', {'train': metric}, step)
                    computer.reset()
                logger.info(logstr)
            
            if (args.n_ckpt_steps > 0 and step % args.n_ckpt_steps == 0) or batch_index == len(train_loader) - 1:
                save_model(model, ckpt_dir, step, exp_dir, epoch)
                val_metrics, embed = validation(
                    hp.tasks, model, val_loader, criterion, 
                    device, metric_computers)
                metric_str = ', '.join([f'{x} : {y:5f}' for x, y in val_metrics.items()])
                logger.info(f'[epoch {epoch}, step {step}] VAL {metric_str}')
                steps_ellapsed, model_info = track_best_model(
                    log_dir, 
                    metric_key=metric_selector,
                    min_max=metric_selector_min_max,
                    new_entry={
                        metric_selector: val_metrics[metric_selector], 
                        'step': step, 
                        'epoch': epoch
                    }
                )

                keep_best_model(ckpt_dir, model_info['step'])
                clean_checkpoints(ckpt_dir, _MAX_NUM_CKPTS, skip_step=model_info['step'])
                if embed is not None:
                    tsb_writer.add_embedding(
                        embed['embed'], 
                        embed['label'], 
                        global_step=step,
                        tag=args.tag)
                for key, metric in val_metrics.items():
                    tsb_writer.add_scalars(
                        key, {'val': metric}, step)
                tsb_writer.add_scalar(
                    'learning_rate', lr_scheduler.get_last_lr(), step)
            time_1 = time()
            if lr_scheduler:
                lr_scheduler.step()
        epoch += 1
        save_model(model, ckpt_dir, step, exp_dir, epoch)
        if early_stopping_tolerance_step > 0 and steps_ellapsed >= early_stopping_tolerance_step * steps_per_epoch:
            logger.info(f'[step {step}, epoch {epoch}] {early_stopping_tolerance_step} early stopping criteria has been met, stopping')
            early_stopped = True
            break
    tsb_writer.close()

    logger.info('Loading test set')
    test_set = CodeImageDataset(
        hp, 
        mode='test', 
        image_dir=args.imagedir,
        dataset_dir=args.dataset_dir
    )

    test_loader = DataLoader(
        test_set, 
        hp.batch_size if not hasattr(hp, 'test_batch_size') else hp.test_batch_size, 
        shuffle=False, 
        num_workers=args.n_loader_jobs,
        collate_fn=test_collate_func,
        pin_memory=True if torch.cuda.is_available() else False
    )

    _, best_model_info = track_best_model(
        log_dir, 
        metric_key=metric_selector,
        min_max=metric_selector_min_max,
    )
    best_step, best_epoch = best_model_info['step'], best_model_info['epoch']
    logger.info(f'Loading best model from step {best_step} of epoch {best_epoch}')
    load_model(model, ckpt_dir, best_step)

    logger.info('Running test')
    test_metrics, _ = validation(
        hp.tasks, model, test_loader, criterion, 
        device, metric_computers, extra_metric_computers=extra_test_metrics)
    metric_str = ', '.join([f'{x} : {y:5f}' for x, y in test_metrics.items()])
    logger.info(f'Test results: {metric_str}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tag', type=str, help='tag identifier from hparams.yml')
    parser.add_argument('dataset_dir', type=str, help='dataset directory')
    parser.add_argument('--hparams', type=str, default=os.path.join('hparams', 'default.yml'), help='hyperparameter file path, default hparams/default.yml')
    parser.add_argument('--n-loader-jobs', type=int, default=0, help='number of workers for data loading')
    parser.add_argument('--n-ckpt-steps', type=int, default=-1, help='checkpointing/validation interval (in steps), -1 for per epoch')
    parser.add_argument('--imagedir', type=str, default='', help='source directory of the image data')
    parser.add_argument('--recover_lr', action='store_true', default=False, help='compute the decayed learning rate from progress')
    args = parser.parse_args()
    torch.set_num_threads(os.cpu_count())
    train(args)
