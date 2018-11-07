# ClariNet
A Pytorch Implementation of ClariNet (Mel Spectrogram --> Waveform)


# Requirements

PyTorch 0.4.0 & python 3.6 & Librosa

# Examples

#### Step 1. Download Dataset

- LJSpeech : [https://keithito.com/LJ-Speech-Dataset/](https://keithito.com/LJ-Speech-Dataset/)

#### Step 2. Preprocessing (Preparing Mel Spectrogram)

`python preprocessing.py --in_dir ljspeech --out_dir DATASETS/ljspeech`

#### Step 3. Train Gaussian Autoregressive WaveNet (Teacher)

`python train.py --model_name wavenet_gaussian --batch_size 8 --num_blocks 4 --num_layers 6`

#### Step 4. Synthesize (Teacher)

`--load_step CHECKPOINT` : the # of the pre-trained *teacher* model's global training step (also depicted in the trained weight file)

`python synthesize.py --model_name wavenet_gaussian --num_blocks 4 --num_layers 6 --load_step 10000`

#### Step 5. Train Gaussian Inverse Autoregressive Flow (Student)

`--teacher_name YOUR TEACHER MODEL'S NAME`

`--teacher_load_step CHECKPOINT` : : the # of the pre-trained *teacher* model's global training step (also depicted in the trained weight file)

`--KL_type qp` : Reversed KL divegence KL(q||p)  or `--KL_type pq` : Forward KL divergence KL(p||q)

`python train_student.py --model_name wavenet_gaussian_student --teacher_name wavenet_gaussian --teacher_load_step 10000 --batch_size 4 --num_blocks_t 4 --num_layers_t 6 --num_layers_s 6 --KL_type qp`

#### Step 6. Synthesize (Student)

`--model_name YOUR STUDENT MODEL'S NAME`

`--load_step CHECKPOINT` : : the # of the pre-trained *student* model's global training step (also depicted in the trained weight file)

`--teacher_name YOUR TEACHER MODEL'S NAME`

`--teacher_load_step CHECKPOINT` :  the # of the pre-trained *teacher* model's global training step (also depicted in the trained weight file)

`--KL_type qp` : Reversed KL divegence KL(q||p)  or `--KL_type pq` : Forward KL divergence KL(p||q)

`--temp TEMPERATURE` : Temperature (standard deviation) value implemented as z ~ N(0, 1 * TEMPERATURE)

`python synthesize_student.py --model_name wavenet_gaussian_student --load_step 10000 --teacher_name wavenet_gaussian --teacher_load_step 10000 --batch_size 4 --num_blocks_t 4 --num_layers_t 6 --num_layers_s 6 --KL_type qp --num_blocks_t 4 --num_layers_t 6 --num_layers_s 6 --num_samples 5 --temp 0.7`

# References

WaveNet vocoder : [https://github.com/r9y9/wavenet_vocoder](https://github.com/r9y9/wavenet_vocoder)

ClariNet : [https://arxiv.org/abs/1807.07281](https://arxiv.org/abs/1807.07281)
