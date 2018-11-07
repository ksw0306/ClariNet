import time
import torch
from torch.utils.data import Dataset, DataLoader
from data import LJspeechDataset, collate_fn, collate_fn_synthesize
from wavenet import Wavenet
import librosa
import os
import argparse


parser = argparse.ArgumentParser(description='Train WaveNet of LJSpeech',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', type=str, default='./DATASETS/ljspeech/', help='Dataset Path')
parser.add_argument('--sample_path', type=str, default='./samples', help='Sample Path')
parser.add_argument('--save', '-s', type=str, default='./params', help='Folder to save checkpoints.')
parser.add_argument('--load', '-l', type=str, default='./params', help='Checkpoint path to resume / test.')
parser.add_argument('--loss', type=str, default='./loss', help='Folder to save loss')
parser.add_argument('--log', type=str, default='./log', help='Log folder.')
parser.add_argument('--model_name', type=str, default='wavenet_gaussian_01', help='Model Name')
parser.add_argument('--load_step', type=int, default=0, help='Load Step')

# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--batch_size', '-b', type=int, default=1, help='Batch size.')
parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3, help='The Learning Rate.')
parser.add_argument('--ema_decay', type=float, default=0.9999, help='Exponential Moving Average Decay')

parser.add_argument('--num_blocks', type=int, default=4, help='Number of blocks')
parser.add_argument('--num_layers', type=int, default=6, help='Number of layers')
parser.add_argument('--residual_channels', type=int, default=128, help='Residual Channels')
parser.add_argument('--gate_channels', type=int, default=256, help='Gate Channels')
parser.add_argument('--skip_channels', type=int, default=128, help='Skip Channels')
parser.add_argument('--kernel_size', type=int, default=3, help='Kernel Size')
parser.add_argument('--cin_channels', type=int, default=80, help='Cin Channels')

parser.add_argument('--num_samples', type=int, default=5, help='Number of Samples')

parser.add_argument('--num_workers', type=int, default=1, help='Number of workers')


args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

if not os.path.isdir(args.sample_path):
    os.makedirs(args.sample_path)
if not os.path.isdir(os.path.join(args.sample_path, args.model_name)):
    os.makedirs(os.path.join(args.sample_path, args.model_name))

# LOAD DATASETS
train_dataset = LJspeechDataset(args.data_path, True, 0.1)
test_dataset = LJspeechDataset(args.data_path, False, 0.1)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
                          num_workers=args.num_workers, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn_synthesize,
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


def load_checkpoint(path, model):
    print("Load checkpoint from: {}".format(path))
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])
    return model


step = args.load_step
path = os.path.join(args.load, args.model_name, "checkpoint_step{:09d}_ema.pth".format(step))
model = build_model()
model = load_checkpoint(path, model)
model.to(device)

model.eval()
for i, (x, y, c, _) in enumerate(test_loader):
    if i < args.num_samples:
        x, c = x.to(device), c.to(device)

        wav_truth_name = '{}/{}/generate_{}_{}_truth.wav'.format(args.sample_path, args.model_name, step, i)
        librosa.output.write_wav(wav_truth_name, y.squeeze().numpy(), sr=22050)
        torch.cuda.synchronize()
        start_time = time.time()

        with torch.no_grad():
            y_gen = model.generate(x.size()[-1], c).squeeze()
        torch.cuda.synchronize()
        print('{} seconds'.format(time.time()-start_time))
        wav = y_gen.numpy()
        wav_name = '{}/{}/generate_{}_{}.wav'.format(args.sample_path, args.model_name, step, i)
        librosa.output.write_wav(wav_name, wav, sr=22050)
        del y_gen

