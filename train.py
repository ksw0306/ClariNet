import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from data import LJspeechDataset, collate_fn
from modules import ExponentialMovingAverage, GaussianLoss
from wavenet import Wavenet
from torch.distributions.normal import Normal
import numpy as np
import librosa
import os
import argparse
import json
import time
import gc

torch.backends.cudnn.benchmark = True
np.set_printoptions(precision=4)

parser = argparse.ArgumentParser(description='Train WaveNet of LJSpeech',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', type=str, default='./DATASETS/ljspeech/', help='Dataset Path')
parser.add_argument('--sample_path', type=str, default='./samples', help='Sample Path')
parser.add_argument('--save', '-s', type=str, default='./params', help='Folder to save checkpoints.')
parser.add_argument('--load', '-l', type=str, default='./params', help='Checkpoint path to resume / test.')
parser.add_argument('--loss', type=str, default='./loss', help='Folder to save loss')
parser.add_argument('--log', type=str, default='./log', help='Log folder.')

parser.add_argument('--model_name', type=str, default='wavenet_gaussian_01', help='Model Name')
parser.add_argument('--load_step', type=int, default=0, help='Model Load Step')

parser.add_argument('--epochs', '-e', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--batch_size', '-b', type=int, default=8, help='Batch size.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.001, help='The Learning Rate.')
parser.add_argument('--ema_decay', type=float, default=0.9999, help='Exponential Moving Average Decay')

parser.add_argument('--num_blocks', type=int, default=2, help='Number of blocks')
parser.add_argument('--num_layers', type=int, default=10, help='Number of layers')
parser.add_argument('--residual_channels', type=int, default=128, help='Residual Channels')
parser.add_argument('--gate_channels', type=int, default=256, help='Gate Channels')
parser.add_argument('--skip_channels', type=int, default=128, help='Skip Channels')
parser.add_argument('--kernel_size', type=int, default=2, help='Kernel Size')
parser.add_argument('--cin_channels', type=int, default=80, help='Cin Channels')
parser.add_argument('--num_workers', type=int, default=2, help='Number of workers')

args = parser.parse_args()

# Init logger
if not os.path.isdir(args.log):
    os.makedirs(args.log)

# Checkpoint dir
if not os.path.isdir(args.save):
    os.makedirs(args.save)
if not os.path.isdir(args.loss):
    os.makedirs(args.loss)
if not os.path.isdir(args.sample_path):
    os.makedirs(args.sample_path)
if not os.path.isdir(os.path.join(args.save, args.model_name)):
    os.makedirs(os.path.join(args.save, args.model_name))

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# LOAD DATASETS
train_dataset = LJspeechDataset(args.data_path, True, 0.1)
test_dataset = LJspeechDataset(args.data_path, False, 0.1)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
                          num_workers=args.num_workers, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn,
                         num_workers=args.num_workers, pin_memory=True)


def build_model():
    model = Wavenet(out_channels=2,
                    num_blocks=args.num_blocks,
                    num_layers=args.num_layers,
                    residual_channels=args.residual_channels,
                    gate_channels=args.gate_channels,
                    skip_channels=args.skip_channels,
                    kernel_size=args.kernel_size,
                    cin_channels=args.cin_channels,
                    upsample_scales=[16, 16])
    return model


def clone_as_averaged_model(model, ema):
    assert ema is not None
    averaged_model = build_model()
    averaged_model.to(device)
    averaged_model.load_state_dict(model.state_dict())

    for name, param in averaged_model.named_parameters():
        if name in ema.shadow:
            param = ema.shadow[name].clone()
    return averaged_model


def train(epoch, model, optimizer, ema):
    global global_step
    epoch_loss = 0.
    running_loss = 0.
    model.train()
    start_time = time.time()
    display_step = 100
    for batch_idx, (x, y, c, _) in enumerate(train_loader):
        global_step += 1
        if global_step == 200000:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
                state['learning_rate'] = param_group['lr']
        if global_step == 400000:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
                state['learning_rate'] = param_group['lr']
        if global_step == 600000:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
                state['learning_rate'] = param_group['lr']
        x, y, c = x.to(device), y.to(device), c.to(device)

        optimizer.zero_grad()
        y_hat = model(x, c)
        loss = criterion(y_hat[:, :, :-1], y[:, 1:, :], size_average=True)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 10.)
        optimizer.step()
        if ema is not None:
            for name, param in model.named_parameters():
                if name in ema.shadow:
                    ema.update(name, param.data)

        running_loss += loss.item() / display_step
        epoch_loss += loss.item()
        if (batch_idx + 1) % 100 == 0:
            end_time = time.time()
            print('Global Step : {}, [{}, {}] loss : {:.4f}'.format(global_step, epoch, batch_idx + 1, running_loss))
            print('100 Step Time : {}'.format(end_time - start_time))
            start_time = time.time()
            running_loss = 0.
        del y_hat, x, y, c, loss
    gc.collect()
    print('{} Epoch Training Loss : {:.4f}'.format(epoch, epoch_loss / (len(train_loader))))
    del running_loss
    return epoch_loss / len(train_loader)


def evaluate(model, ema=None):
    if ema is not None:
        model_ema = clone_as_averaged_model(model, ema)
    model_ema.eval()
    running_loss = 0.
    epoch_loss = 0.
    display_step = 100
    for batch_idx, (x, y, c, _) in enumerate(test_loader):
        x, y, c = x.to(device), y.to(device), c.to(device)

        y_hat = model_ema(x, c)

        loss = criterion(y_hat[:, :, :-1], y[:, 1:, :], size_average=True)

        running_loss += loss.item() / display_step
        epoch_loss += loss.item()

        if (batch_idx + 1) % display_step == 0:
            print('{} Loss : {:.4f}'.format(batch_idx + 1, running_loss))
            running_loss = 0.
        del y_hat, x, y, c, loss
    del model_ema
    epoch_loss /= len(test_loader)
    print('Evaluation Loss : {:.4f}'.format(epoch_loss))
    return epoch_loss


def save_checkpoint(model, optimizer, global_step, global_epoch, ema=None):
    checkpoint_path = os.path.join(args.save, args.model_name, "checkpoint_step{:09d}.pth".format(global_step))
    optimizer_state = optimizer.state_dict()
    torch.save({"state_dict": model.state_dict(),
                "optimizer": optimizer_state,
                "global_step": global_step,
                "global_epoch": global_epoch}, checkpoint_path)
    if ema is not None:
        averaged_model = clone_as_averaged_model(model, ema)
        checkpoint_path = os.path.join(args.save, args.model_name, "checkpoint_step{:09d}_ema.pth".format(global_step))
        torch.save({"state_dict": averaged_model.state_dict(),
                    "optimizer": optimizer_state,
                    "global_step": global_step,
                    "global_epoch": global_epoch}, checkpoint_path)


def load_checkpoint(step, model, optimizer, ema=None):
    global global_step
    global global_epoch

    checkpoint_path = os.path.join(args.save, args.model_name, "checkpoint_step{:09d}.pth".format(step))
    print("Load checkpoint from: {}".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]

    if ema is not None:
        checkpoint_path = os.path.join(args.save, args.model_name, "checkpoint_step{:09d}_ema.pth".format(step))
        checkpoint = torch.load(checkpoint_path)
        averaged_model = build_model()
        averaged_model.to(device)
        averaged_model.load_state_dict(checkpoint["state_dict"])
        for name, param in averaged_model.named_parameters():
            if param.requires_grad:
                ema.register(name, param.data)
    return model, optimizer, ema


model = build_model()
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
criterion = GaussianLoss()

ema = ExponentialMovingAverage(args.ema_decay)
for name, param in model.named_parameters():
    if param.requires_grad:
        ema.register(name, param.data)

global_step, global_epoch = 0, 0
load_step = args.load_step

log = open(os.path.join(args.log, '{}.txt'.format(args.model_name)), 'w')
state = {k: v for k, v in args._get_kwargs()}

if load_step == 0:
    list_train_loss, list_loss = [], []
    log.write(json.dumps(state) + '\n')
    test_loss = 100.0
else:
    model, optimizer, ema = load_checkpoint(load_step, model, optimizer, ema)
    list_train_loss = np.load('{}/{}_train.npy'.format(args.loss, args.model_name)).tolist()
    list_loss = np.load('{}/{}.npy'.format(args.loss, args.model_name)).tolist()
    list_train_loss = list_train_loss[:global_epoch]
    list_loss = list_loss[:global_epoch]
    test_loss = np.min(list_loss)

for epoch in range(global_epoch + 1, args.epochs + 1):
    training_epoch_loss = train(epoch, model, optimizer, ema)
    with torch.no_grad():
        test_epoch_loss = evaluate(model, ema)
    
    state['training_loss'] = training_epoch_loss
    state['eval_loss'] = test_epoch_loss
    state['epoch'] = epoch
    list_train_loss.append(training_epoch_loss)
    list_loss.append(test_epoch_loss)

    if test_loss > test_epoch_loss:
        test_loss = test_epoch_loss
        save_checkpoint(model, optimizer, global_step, epoch, ema)
        print('Epoch {} Model Saved! Loss : {:.4f}'.format(epoch, test_loss))
    np.save('{}/{}_train.npy'.format(args.loss, args.model_name), list_train_loss)
    np.save('{}/{}.npy'.format(args.loss, args.model_name), list_loss)

    log.write('%s\n' % json.dumps(state))
    log.flush()
    print(state)
    gc.collect()

log.close()
