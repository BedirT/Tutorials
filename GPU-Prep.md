# Setting up NVIDIA and Cuda for Tensorflow and Pytorch

These are the exact steps I have followed after another crash on my tf setup...

### Specs
Ubuntu version:
```bash
$ lsb_release -a
No LSB modules are available.
Distributor ID: Ubuntu
Description:    Ubuntu 20.04.2 LTS
Release:        20.04
Codename:       focal
```

Devices:
```bash
$ sudo lshw -C display
  *-display                 
       description: VGA compatible controller
       product: GP104 [GeForce GTX 1080]
       vendor: NVIDIA Corporation
       physical id: 0
       bus info: pci@0000:01:00.0
       version: a1
       width: 64 bits
       clock: 33MHz
       capabilities: pm msi pciexpress vga_controller bus_master cap_list rom
       configuration: driver=nvidia latency=0
       resources: irq:132 memory:de000000-deffffff memory:c0000000-cfffffff memory:d0000000-d1ffffff ioport:e000(size=128) memory:df000000-df07ffff
```

## Steps

### Remove Everything
- Remove NVIDIA Drivers
#### Remove Cuda
- Go to appropriate cuda file you have in your system (Should be located in ```/usr/local/``` with name i.e. ```cuda-10.0``` for cuda version 10.0), and go to ```bin/``` run ```cuda-uninstaller```
```bash
cd /usr/local/cuda-10.0/bin
chmod +x cuda-uninstaller
sudo ./cuda-uninstaller
```
- If you can't locate or want to do it manually
```bash
sudo apt-get remove nvidia-cuda-toolkit
sudo apt-get remove --auto-remove nvidia-cuda-toolkit
sudo apt-get purge nvidia-cuda-toolkit
sudo apt-get purge --auto-remove nvidia-cuda-toolkit
sudo rm -rf /opt/cuda
sudo rm -rf ~/NVIDIA_GPU_Computing_SDK
```
- Remove the lines 
```shell
export PATH=$PATH:/opt/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/cuda/lib:/opt/cuda/lib64
```
from ```~/.bash_profile``` and/or ```~/.bashrc```

### Install NVIDIA
- Go to [Nvidia website](https://www.nvidia.com/en-us/drivers/unix/) and get the latest *Latest Production Branch Version* link. For me the link is **https://us.download.nvidia.com/XFree86/Linux-x86_64/460.73.01/NVIDIA-Linux-x86_64-460.73.01.run**
```shell
wget -c https://us.download.nvidia.com/XFree86/Linux-x86_64/460.73.01/NVIDIA-Linux-x86_64-460.73.01.run
```
- Unload the kernel (needed if the graphical instance of the system is running somehow) 
```bash
systemctl isolate multi-user.target
modprobe -r nvidia-drm
```
To restart the graphical env. run ```systemctl start graphical.target```
[details](https://unix.stackexchange.com/questions/440840/how-to-unload-kernel-module-nvidia-drm)
- Run the installation file
```bash
sudo sh NVIDIA-Linux-x86_64-460.73.01.run
```
- Reboot 
```bash
sudo reboot
```
### Install CUDA
- Download the latest version, for me it is 11.0.2. [Here](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&=Ubuntu)
```bash
wget http://developer.download.nvidia.com/compute/cuda/11.0.2/local_installers/cuda_11.0.2_450.51.05_linux.run
```
- Run the installation file.
```bash
sudo sh cuda_11.0.2_450.51.05_linux.run
```
- Set the env variables on ```.bashrc```
```bash
export PATH=/usr/local/cuda-11.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH
```
### Install cuDNN
- Download the cuDNN file. Go to the [official cuDNN website](), register and get the link for the file. If the machine you are using have graphical interface then you can just download the file. If not (which is the case for me since I am using a remote server.) then we need to get a public version of the link since it needs auth otherwise and you will get a permission error. Here is the link it generates, we need to edit it according to the script below. 
```bash
wget -c https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.1.1.33/11.2_20210301/cudnn-11.2-linux-x64-v8.1.1.33.tgz
```
Here is the script that we use to edit the link. You can just insert the changes by hand as well. (Don't forget to remove the $'s)
```bash
VERSION_FULL="8.1.0.77"
VERSION="${VERSION_FULL%.*}"
CUDA_VERSION="11.2"
OS_ARCH="linux-x64"
CUDNN_URL="https://developer.download.nvidia.com/compute/redist/cudnn/v${VERSION}/cudnn-${CUDA_VERSION}-${OS_ARCH}-v${VERSION_FULL}.tgz" 
wget -c ${CUDNN_URL}
```
So the link for 8.1.1.33 will be;
```wget -c https://developer.download.nvidia.com/compute/redist/cudnn/v8.1.1/cudnn-11.2-linux-x64-v8.1.1.33.tgz```
- Setting up the cuDNN files.
```bash
tar -xzvf cudnn-11.2-linux-x64-v8.1.1.33.tgz

sudo cp cuda/include/cudnn*.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```
If you have a different system version here is how the link actually got generated;

[For details](https://stackoverflow.com/questions/60849474/how-to-download-the-cudnn-straight-from-nvidia-website-to-my-linux-instance-on-g)

Mostly I have followed [this post](https://gist.github.com/kmhofmann/cee7c0053da8cc09d62d74a6a4c1c5e4) only with minor changes, so check if needed.
