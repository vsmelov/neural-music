# neural-music
Use neural network to arranging and composing music

Usage:
- (if you want to use GPU) install nvidia GPU drivers
- install nvidia-docker https://github.com/NVIDIA/nvidia-docker (for GPU-mode) or only docker (for CPU-mode)
- build container
nvidia-docker build -t neural-music .
or
docker build -t neural-music .
- run command nvidia-docker -t -i -v ./neural-music:/neural-music
$PWD

cd sms/software/models/utilFunctions_C
python compileModule.py build_ext --inplace

1) convert to wav
2) prepare date
3) train
4) compose
