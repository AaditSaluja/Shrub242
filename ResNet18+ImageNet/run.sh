module load python/3.10.9-fasrc01 cuda/12.0.1-fasrc01 cudnn
# mamba create -n aadit_tori
mamba activate whatover
# mamba install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
# pip3 install numpy, ptflops, torchjpeg, matplotlib
# pip3 install tenserflow
# mamba install tenserflow
pip3 install scipy


python resnet18+imagenet.py
