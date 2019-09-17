
# horovod-distributed-example

### Install

Prerequisites
To run this project on your local machine, you need:

 - Python 3.6 
 - Git 

If you're adventurous and want to test MPI locally, I suggest pulling a pre-built Docker image. For example, docker pull uber/horovod:0.15.0-tf1.11.0-torch0.4.1-py3.5 (be warned, the image is 3GB--see more options here)

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
python main.py
```
For distributed mode, you need to have a working MPI. If you have Docker, you can do this:
```bash
docker run -it -v $(pwd):/code uber/horovod:0.15.0-tf1.11.0-torch0.4.1-py3.5 /bin/bash
uber/horovod> cd /code
uber/horovod> mpirun --allow-run-as-root -np 3 -H localhost:3 -bind-to none -map-by slot -mca pml ob1 -mca btl ^openib python main.py
```
This will start three local processes running synchronous distributed training.

## Run it on Gradient 

[WIP]

Command to launch it on kubernetes

--docker-image horovod-0.15.0-cpu-py36-tf1.11.0 \
--ps-docker-image horovod-0.15.0-cpu-py36-tf1.11.0 \
--command "mpirun --allow-run-as-root -np 3 --hostfile /kube-openmpi/generated/hostfile -bind-to none -map-by slot -mca pml ob1 -mca btl ^openib python openmpi/horovod/main.py" \
