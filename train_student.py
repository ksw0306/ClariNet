import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal
from data import LJspeechDataset, collate_fn, collate_fn_synthesize
from modules import ExponentialMovingAverage, KL_Loss, stft
from wavenet import Wavenet
from wavenet_iaf import Wavenet_Student
import numpy as np
import librosa
import os
import argparse
import json
import time

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

parser.add_argument('--teacher_name', type=str, default='wavenet_gaussian_01', help='Model Name')
parser.add_argument('--model_name', type=str, default='clarinet_01', help='Model Name')
parser.add_argument('--teacher_load_step', type=int, default=0, help='Teacher Load Step')
parser.add_argument('--load_step', type=int, default=0, help='Student Load Step')

parser.add_argument('--KL_type', type=str, default='qp', help='KL_pq vs KL_qp')
parser.add_argument('--epochs', '-e', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--batch_size', '-b', type=int, default=8, help='Batch size.')
parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3, help='The Learning Rate.')
parser.add_argument('--ema_decay', type=float, default=0.9999, help='Exponential Moving Average Decay')
parser.add_argument('--num_blocks_t', type=int, default=2, help='Number of blocks (Teacher)')
parser.add_argument('--num_layers_t', type=int, default=10, help='Number of layers (Teacher)')
parser.add_argument('--num_layers_s', type=int, default=10, help='Number of layers (Student)')
parser.add_argument('--residual_channels', type=int, default=128, help='Residual Channels')
parser.add_argument('--gate_channels', type=int, default=256, help='Gate Channels')
parser.add_argument('--skip_channels', type=int, default=128, help='Skip Channels')
parser.add_argument('--kernel_size', type=int, default=2, help='Kernel Size')
parser.add_argument('--cin_channels', type=int, default=80, help='Cin Channels')
parser.add_argument('--num_workers', type=int, default=3, help='Number of workers')
parser.add_argument('--num_gpu', type=int, default=1, help='Number of GPUs to use. >1 uses DataParallel')

args = parser.parse_args()

# Init logger
if not os.path.isdir(args.log):
    os.makedirs(args.log)

# Checkpoint dir
if not os.path.isdir(args.save):
    os.makedirs(args.save)
if not os.path.isdir(args.loss):
    os.makedirs(args.loss)
if not os.path.isdir(os.path.join(args.save, args.teacher_name)):
    os.makedirs(os.path.join(args.save, args.teacher_name))
if not os.path.isdir(os.path.join(args.save, args.teacher_name, args.model_name)):
    os.makedirs(os.path.join(args.save, args.teacher_name, args.model_name))
if not os.path.isdir(os.path.join(args.sample_path, args.teacher_name)):
    os.makedirs(os.path.join(args.sample_path, args.teacher_name))
if not os.path.isdir(os.path.join(args.sample_path, args.teacher_name, args.model_name)):
    os.makedirs(os.path.join(args.sample_path, args.teacher_name, args.model_name))

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# LOAD DATASETS
train_dataset = LJspeechDataset(args.data_path, True, 0.1)
test_dataset = LJspeechDataset(args.data_path, False, 0.1)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
                          num_workers=args.num_workers, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn,
                         num_workers=args.num_workers)
synth_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn_synthesize,
                          num_workers=args.num_workers, pin_memory=True)


def build_model():
    model_t = Wavenet(out_channels=2,
                      num_blocks=args.num_blocks_t,
                      num_layers=args.num_layers_t,
                      residual_channels=args.residual_channels,
                      gate_channels=args.gate_channels,
                      skip_channels=args.skip_channels,
                      kernel_size=args.kernel_size,
                      cin_channels=args.cin_channels,
                      upsample_scales=[16, 16])
    return model_t


def build_student():
    model_s = Wavenet_Student(num_blocks_student=[1, 1, 1, 1, 1, 1],
                              num_layers=args.num_layers_s)
    return model_s


def clone_as_averaged_model(model_s, ema):
    assert ema is not None
    averaged_model = build_student()
    averaged_model.to(device)
    if args.num_gpu > 1:
        averaged_model = torch.nn.DataParallel(averaged_model)
    averaged_model.load_state_dict(model_s.state_dict())
    for name, _ in averaged_model.named_parameters():
        if name in ema.shadow:
            averaged_model.named_parameters()[name] = ema.shadow[name].clone()
    return averaged_model


def train(epoch, model_t, model_s, optimizer, ema):
    global global_step
    epoch_loss = 0.0
    running_loss = [0.0, 0.0, 0.0, 0.0]
    model_t.eval()
    model_s.train()
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

        q_0 = Normal(x.new_zeros(x.size()), x.new_ones(x.size()))
        z = q_0.sample()

        optimizer.zero_grad()
        c_up = model_t.upsample(c)
        x_student, mu_s, logs_s = model_s(z, c_up)  # q_T ~ N(mu_tot, logs_tot.exp_())

        mu_logs_t = model_t(x_student, c)

        if args.KL_type == 'pq':
            loss_t, loss_KL, loss_reg = criterion_t(mu_logs_t[:, 0:1, :-1], mu_logs_t[:, 1:, :-1], mu_s, logs_s)
        elif args.KL_type == 'qp':
            loss_t, loss_KL, loss_reg = criterion_t(mu_s, logs_s, mu_logs_t[:, 0:1, :-1], mu_logs_t[:, 1:, :-1])

        stft_student = stft(x_student[:, 0, 1:], scale='linear')
        stft_truth = stft(x[:, 0, 1:], scale='linear')
        loss_frame = criterion_frame(stft_student, stft_truth)
        loss_tot = loss_t + loss_frame
        loss_tot.backward()

        nn.utils.clip_grad_norm_(model_s.parameters(), 10.)
        optimizer.step()
        if ema is not None:
            for name, param in model_s.named_parameters():
                if name in ema.shadow:
                    ema.update(name, param.data)

        running_loss[0] += loss_tot.item() / display_step
        running_loss[1] += loss_KL.item() / display_step
        running_loss[2] += loss_reg.item() / display_step
        running_loss[3] += loss_frame.item() / display_step
        epoch_loss += loss_tot.item()
        if (batch_idx + 1) % display_step == 0:
            end_time = time.time()
            print('Global Step : {}, [{}, {}] [Total Loss, KL Loss, Reg Loss, Frame Loss] : {}'
                  .format(global_step, epoch, batch_idx + 1, np.array(running_loss)))
            print('{} Step Time : {}'.format(display_step, end_time - start_time))
            start_time = time.time()
            running_loss = [0.0, 0.0, 0.0, 0.0]
        del loss_tot, loss_frame, loss_KL, loss_reg, loss_t, x, y, c, c_up, stft_student, stft_truth, q_0, z
        del x_student, mu_s, logs_s, mu_logs_t
    print('{} Epoch Training Loss : {:.4f}'.format(epoch, epoch_loss / (len(train_loader))))
    return epoch_loss / len(train_loader)


def evaluate(model_t, model_s, ema=None):
    if ema is not None:
        model_s_ema = clone_as_averaged_model(model_s, ema)
    model_t.eval()
    model_s_ema.eval()
    running_loss = [0., 0., 0., 0.]
    epoch_loss = 0.

    display_step = 100
    for batch_idx, (x, y, c, _) in enumerate(test_loader):
        x, y, c = x.to(device), y.to(device), c.to(device)

        q_0 = Normal(x.new_zeros(x.size()), x.new_ones(x.size()))
        z = q_0.sample()
        c_up = model_t.upsample(c)

        x_student, mu_s, logs_s = model_s_ema(z, c_up)

        mu_logs_t = model_t(x_student, c)

        if args.KL_type == 'pq':
            loss_t, loss_KL, loss_reg = criterion_t(mu_logs_t[:, 0:1, :-1], mu_logs_t[:, 1:, :-1], mu_s, logs_s)
        elif args.KL_type == 'qp':
            loss_t, loss_KL, loss_reg = criterion_t(mu_s, logs_s, mu_logs_t[:, 0:1, :-1], mu_logs_t[:, 1:, :-1])

        stft_student = stft(x_student[:, 0, 1:], scale='linear')
        stft_truth = stft(x[:, 0, 1:], scale='linear')

        loss_frame = criterion_frame(stft_student, stft_truth.detach())

        loss_tot = loss_t + loss_frame

        running_loss[0] += loss_tot.item() / display_step
        running_loss[1] += loss_KL.item() / display_step
        running_loss[2] += loss_reg.item() / display_step
        running_loss[3] += loss_frame.item() / display_step
        epoch_loss += loss_tot.item()

        if (batch_idx + 1) % display_step == 0:
            print('{} [Total, KL, Reg, Frame Loss] : {}'.format(batch_idx + 1, np.array(running_loss)))
            running_loss = [0., 0., 0., 0.]
        del loss_tot, loss_frame, loss_KL, loss_reg, loss_t, x, y, c, c_up, stft_student, stft_truth, q_0, z
        del x_student, mu_s, logs_s, mu_logs_t
    epoch_loss /= len(test_loader)
    print('Evaluation Loss : {:.4f}'.format(epoch_loss))
    del model_s_ema
    return epoch_loss


def synthesize(model_t, model_s, ema=None):
    global global_step
    if ema is not None:
        model_s_ema = clone_as_averaged_model(model_s, ema)
    model_s_ema.eval()
    for batch_idx, (x, y, c, _) in enumerate(synth_loader):
        if batch_idx == 0:
            x, c = x.to(device), c.to(device)

            q_0 = Normal(x.new_zeros(x.size()), x.new_ones(x.size()))
            z = q_0.sample()
            wav_truth_name = '{}/{}/{}/generate_{}_{}_truth.wav'.format(args.sample_path, args.teacher_name,
                                                                        args.model_name, global_step, batch_idx)
            librosa.output.write_wav(wav_truth_name, y.squeeze().numpy(), sr=22050)
            print('{} Saved!'.format(wav_truth_name))

            torch.cuda.synchronize()
            start_time = time.time()
            c_up = model_t.upsample(c)

            with torch.no_grad():
                if args.num_gpu == 1:
                    y_gen = model_s_ema.generate(z, c_up).squeeze()
                else:
                    y_gen = model_s_ema.module.generate(z, c_up).squeeze()
            torch.cuda.synchronize()
            print('{} seconds'.format(time.time() - start_time))
            wav = y_gen.to(torch.device("cpu")).data.numpy()
            wav_name = '{}/{}/{}/generate_{}_{}.wav'.format(args.sample_path, args.teacher_name,
                                                            args.model_name, global_step, batch_idx)
            librosa.output.write_wav(wav_name, wav, sr=22050)
            print('{} Saved!'.format(wav_name))
            del y_gen, wav, x,  y, c, c_up, z, q_0
    del model_s_ema


def save_checkpoint(model, optimizer, global_step, global_epoch, ema=None):
    checkpoint_path = os.path.join(args.save, args.teacher_name, args.model_name, "checkpoint_step{:09d}.pth".format(global_step))
    optimizer_state = optimizer.state_dict()
    torch.save({"state_dict": model.state_dict(),
                "optimizer": optimizer_state,
                "global_step": global_step,
                "global_epoch": global_epoch}, checkpoint_path)
    if ema is not None:
        averaged_model = clone_as_averaged_model(model, ema)
        checkpoint_path = os.path.join(args.save, args.teacher_name, args.model_name, "checkpoint_step{:09d}_ema.pth".format(global_step))
        torch.save({"state_dict": averaged_model.state_dict(),
                    "optimizer": optimizer_state,
                    "global_step": global_step,
                    "global_epoch": global_epoch}, checkpoint_path)


def load_checkpoint(step, model_s, optimizer, ema=None):
    global global_step
    global global_epoch

    checkpoint_path = os.path.join(args.save, args.teacher_name, args.model_name, "checkpoint_step{:09d}.pth".format(step))
    print("Load checkpoint from: {}".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)
    model_s.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]

    if ema is not None:
        checkpoint_path = os.path.join(args.save, args.teacher_name, args.model_name, "checkpoint_step{:09d}_ema.pth".format(step))
        checkpoint = torch.load(checkpoint_path)
        averaged_model = build_student()
        averaged_model.to(device)
        try:
            averaged_model.load_state_dict(checkpoint["state_dict"])
        except RuntimeError:
            print("INFO: this model is trained with DataParallel. Creating new state_dict without module...")
            state_dict = checkpoint["state_dict"]
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            averaged_model.load_state_dict(new_state_dict)
        for name, param in averaged_model.named_parameters():
            if param.requires_grad:
                ema.register(name, param.data)
    return model_s, optimizer, ema


def load_teacher_checkpoint(path, model_t):
    print("Load checkpoint from: {}".format(path))
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    model_t.load_state_dict(checkpoint["state_dict"])
    return model_t


teacher_step = args.teacher_load_step
path = os.path.join(args.load, args.teacher_name, "checkpoint_step{:09d}_ema.pth".format(teacher_step))
model_t = build_model()
model_t = load_teacher_checkpoint(path, model_t)
model_s = build_student()

model_t.to(device)
model_s.to(device)
if args.num_gpu > 1:
    #model_t = torch.nn.DataParallel(model_t)
    model_s = torch.nn.DataParallel(model_s)

optimizer = optim.Adam(model_s.parameters(), lr=args.learning_rate)
criterion_t = KL_Loss()
criterion_frame = nn.MSELoss()

ema = ExponentialMovingAverage(args.ema_decay)
for name, param in model_s.named_parameters():
    if param.requires_grad:
        ema.register(name, param.data)
for name, param in model_t.named_parameters():
    if param.requires_grad:
        param.requires_grad = False

global_step, global_epoch = 0, 0
load_step = args.load_step

log = open(os.path.join(args.log, '{}.txt'.format(args.model_name)), 'w')
state = {k: v for k, v in args._get_kwargs()}

if load_step == 0:
    list_train_loss, list_loss = [], []
    log.write(json.dumps(state) + '\n')
    test_loss = 100.0
else:
    model_s, optimizer, ema = load_checkpoint(load_step, model_s, optimizer, ema)
    list_train_loss = np.load('{}/{}_train.npy'.format(args.loss, args.model_name)).tolist()
    list_loss = np.load('{}/{}.npy'.format(args.loss, args.model_name)).tolist()
    list_train_loss = list_train_loss[:global_epoch]
    list_loss = list_loss[:global_epoch]
    test_loss = np.min(list_loss)

for epoch in range(global_epoch + 1, args.epochs + 1):
    training_epoch_loss = train(epoch, model_t, model_s, optimizer, ema)
    with torch.no_grad():
        test_epoch_loss = evaluate(model_t, model_s, ema)

    state['training_loss'] = training_epoch_loss
    state['eval_loss'] = test_epoch_loss
    state['epoch'] = epoch
    list_train_loss.append(training_epoch_loss)
    list_loss.append(test_epoch_loss)

    if test_loss > test_epoch_loss:
        test_loss = test_epoch_loss
        save_checkpoint(model_s, optimizer, global_step, epoch, ema)
        print('Epoch {} Model Saved! Loss : {:.4f}'.format(epoch, test_loss))
        synthesize(model_t, model_s, ema)
    np.save('{}/{}_train.npy'.format(args.loss, args.model_name), list_train_loss)
    np.save('{}/{}.npy'.format(args.loss, args.model_name), list_loss)

    log.write('%s\n' % json.dumps(state))
    log.flush()
    print(state)

log.close()
