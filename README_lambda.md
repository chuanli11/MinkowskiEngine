Note
===



__Installation__

Install CUDA 11.1

```
wget https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda_11.1.1_455.32.00_linux.run

sudo sh cuda_11.1.1_455.32.00_linux.run --toolkit --silent --override

sudo reboot
```


Build MinkowskiEngine
```
git clone https://github.com/chuanli11/MinkowskiEngine.git

NAME_NGC=pytorch:20.10-py3

docker run --gpus all --rm --shm-size=128g -v ~/MinkowskiEngine:/workspace -it nvcr.io/nvidia/${NAME_NGC}

export CUDA_HOME=/usr/local/cuda-11.1
python setup.py install --blas=openblas
```


__Run__ 

Download data

```
wget https://lambdalabs-files.s3-us-west-2.amazonaws.com/ddi_samples.zip

unzip ddi_samples.zip
```

Run benchmark

```
# change batch_size for different GPUs
python minkunet_lambda.py 
```


