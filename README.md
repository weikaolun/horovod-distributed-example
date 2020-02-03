
# Gradient Horovod Examples

Horovod is a distributed deep learning training framework for TensorFlow, Keras, PyTorch, and Apache MXNet. The goal of Horovod is to make distributed deep learning fast and easy to use.
https://github.com/horovod/horovod



### Install

Prerequisites
To run this project on your local machine, you need:

 - Python 3.7+ 
 - Git 

If you're adventurous and want to test MPI locally, I suggest pulling a pre-built Docker image. For example, docker pull horovod/horovod:0.18.1-tf1.14.0-torch1.2.0-mxnet1.5.0-py3.6

### To install Horovod:

Install Open MPI or another MPI implementation. Learn how to install Open MPI on this page.

Note: Open MPI 3.1.3 has an issue that may cause hangs. The recommended fix is to downgrade to Open MPI 3.1.2 or upgrade to Open MPI 4.0.0.

If you've installed TensorFlow from PyPI, make sure that the g++-4.8.5 or g++-4.9 is installed.

If you've installed PyTorch from PyPI, make sure that the g++-4.9 or above is installed.

If you've installed either package from Conda, make sure that the gxx_linux-64 Conda package is installed.

Install the horovod pip package.
$ pip install horovod

**Usage**
You can run this code either locally or on Gradient platform without any code changes. You can also run this in single instance or distributed mode without any code changes.

**Local**
Start out by cloning this repository onto your local machine.

```bash
git clone https://github.com/Paperspace/horovod-distributed-example.git
```
Then cd into the repo folder and install requirements
```bash
pip install -r requirements.txt
```


Single instance mode is very simple. You just execute main.py:
```bash
python keras/keras_mnist.py
```
For distributed mode, you need to have a working MPI. If you have Docker, you can do this:
```bash
docker run -it -v $(pwd):/code horovod/horovod:0.18.1-tf1.14.0-torch1.2.0-mxnet1.5.0-py3.6 /bin/bash
uber/horovod> cd /code
uber/horovod> mpirun --allow-run-as-root -np 1 --hostfile /generated/hostfile  -bind-to none -map-by slot  -x NCCL_DEBUG=INFO -mca pml ob1 -mca btl ^openib python keras/keras_mnist.py
```
This will start three local processes running synchronous distributed training.

## Run it on Gradient 

CLI Command to launch it on Gradient

$ gradient experiments create multinode \
--name mpi-test \
--experimentType MPI 
--workerContainer horovod/horovod:0.18.1-tf1.14.0-torch1.2.0-mxnet1.5.0-py3.6 \
--workerMachineType p2.xlarge \
--workerCount 2 \
--masterContainer horovod/horovod:0.18.1-tf1.14.0-torch1.2.0-mxnet1.5.0-py3.6 \
--masterMachineType p2.xlarge \
--masterCommand "mpirun --allow-run-as-root -np 1 --hostfile /generated/hostfile  -bind-to none -map-by slot  -x NCCL_DEBUG=INFO -mca pml ob1 -mca btl ^openib python keras/keras_mnist.py"  \
--masterCount 1 \
--workspace https://github.com/horovod/horovod.git \
--apiKey XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX \
--vpc