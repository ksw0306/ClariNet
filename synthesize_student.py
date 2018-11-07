import time
import torch
from torch.utils.data import Dataset, DataLoader
from data import LJspeechDataset, collate_fn_synthesize
from wavenet import Wavenet
from wavenet_iaf import Wavenet_Student
from torch.distributions.normal import Normal
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

parser.add_argument('--teacher_name', type=str, default='wavenet_gaussian_01', help='Teacher Name')
parser.add_argument('--model_name', type=str, default='wavenet_student_gaussian_01', help='Model Name')
parser.add_argument('--teacher_load_step', type=int, default=0, help='Teacher Load Step')
parser.add_argument('--load_step', type=int, default=0, help='Student Load Step')

parser.add_argument('--num_blocks_t', type=int, default=4, help='Number of blocks (Teacher)')
parser.add_argument('--num_layers_t', type=int, default=6, help='Number of layers (Teacher)')
parser.add_argument('--num_layers_s', type=int, default=6, help='Number of layers (Student)')
parser.add_argument('--residual_channels', type=int, default=128, help='Residual Channels')
parser.add_argument('--gate_channels', type=int, default=256, help='Gate Channels')
parser.add_argument('--skip_channels', type=int, default=128, help='Skip Channels')
parser.add_argument('--kernel_size', type=int, default=3, help='Kernel Size')
parser.add_argument('--cin_channels', type=int, default=80, help='Cin Channels')

parser.add_argument('--temp', type=float, default=0.7, help='Temperature')
parser.add_argument('--num_samples', type=int, default=10, help='Number of samples')

parser.add_argument('--num_workers', type=int, default=1, help='Number of workers')


args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

if not os.path.isdir(args.sample_path):
    os.makedirs(args.sample_path)
if not os.path.isdir(os.path.join(args.sample_path, args.teacher_name)):
    os.makedirs(os.path.join(args.sample_path, args.teacher_name))
if not os.path.isdir(os.path.join(args.sample_path, args.teacher_name, args.model_name)):
    os.makedirs(os.path.join(args.sample_path, args.teacher_name, args.model_name))

# LOAD DATASETS
test_dataset = LJspeechDataset(args.data_path, False, 0.1)

test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn_synthesize,
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
    model_s = Wavenet_Student(num_blocks_student=[1, 1, 1, 4],
                              num_layers=args.num_layers_s)
    return model_s


def load_checkpoint(path, model):
    print("Load checkpoint from: {}".format(path))
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])
    return model


step_t = args.teacher_load_step
step_s = args.load_step
path_t = os.path.join(args.load, args.teacher_name, "checkpoint_step{:09d}_ema.pth".format(step_t))
path_s = os.path.join(args.load, args.teacher_name, args.model_name, "checkpoint_step{:09d}_ema.pth".format(step_s))
model_t = build_model()
model_t = load_checkpoint(path_t, model_t)
model_s = build_student()
model_s = load_checkpoint(path_s, model_s)
model_t.to(device)
model_s.to(device)

model_t.eval()
model_s.eval()
for i, (x, y, c, l) in enumerate(test_loader):
    if i < args.num_samples:
        x, y, c = x.to(device), y.to(device), c.to(device)
        c_up = model_t.upsample(c)
        q_0 = Normal(x.new_zeros(x.size()), x.new_ones(x.size()))
        z = q_0.sample() * args.temp

        wav_truth_name = '{}/{}/{}/generate_{}_{}_truth.wav'.format(args.sample_path,
                                                                    args.teacher_name,
                                                                    args.model_name,
                                                                    args.load_step,
                                                                    i)
        librosa.output.write_wav(wav_truth_name, y.squeeze().to(torch.device("cpu")).numpy(), sr=22050)
        torch.cuda.synchronize()
        start_time = time.time()

        with torch.no_grad():
            y_gen = model_s.generate(z, c_up).squeeze()
        torch.cuda.synchronize()
        print('{} seconds'.format(time.time() - start_time))
        wav = y_gen.to(torch.device("cpu")).data.numpy()
        wav_name = '{}/{}/{}/generate_{}_{}_{}.wav'.format(args.sample_path,
                                                           args.teacher_name,
                                                           args.model_name,
                                                           args.load_step,
                                                           i,
                                                           args.temp)
        librosa.output.write_wav(wav_name, wav, sr=22050)
        print('{} Saved!'.format(wav_name))

