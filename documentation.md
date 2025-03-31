# LUMI container wrapper
The LUMI container wrapper is set of tools which wrap software installation inside an Apptainer/Singularity container to improve startup times, reduce  I/O load and lessen the number of files on large parallel file system.

The LUMI container wrapper is a general purpose installation wrapper that supports wrapping:

1. Existing installations on the filesystem: Mainly to reduce the I/O load and improve startup times, but may also be used to containerize existing installations that cannot be re-installed.
2. Existing Singularity/Apptainer containers: Mainly to hide the need for using the container runtime from the user.
3. Conda installations: Directly wrap a Conda installation based on a Conda environment file.
4. Pip installations: Directly wrap a pip installation based on a requirements.txt file.



# Examples of using the LUMI container wrapper
We will be running the simple PyTorch example to test LUMI container wrapper. Before moving forward we need to make sure we load the required modules.
The tools provided by the LUMI container wrapper are accessible by loading the lumi-container-wrapper module that is available in the LUMI and CrayEnv software stacks. First run `module purge` and the load the modules given below:

````bash
$ module load LUMI
$ module load lumi-container-wrapper
````

Now, the next thing we want to do is wrap a basic Conda installation.To wrap a basic Conda installation, create an installation directory and run the conda-containerize tool given below:

In our case we are creating the directory named tykkyTest here ` /scratch/<your-project-number>/tykkyTest `.

It contains the .yml file that we will be using to install the software required to run our PyTorch script.
Note that, you need to specify the rocm version of the PyTorch in order to run your training script. If you install the PyTroch which is not compatible to the specific machine, you may not get the error while creating the container wrapper, but it will not work.

.yml file
````bash
name: pytorch_rocm
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11.7
  - numpy=1.26.4
  - filelock=3.16.1
  - fsspec=2024.12.0
  - Jinja2=3.1.5
  - MarkupSafe=3.0.2
  - mpmath=1.3.0
  - networkx=3.2.1
  - pillow=11.1.0
  - sympy=1.13.1
  - typing_extensions=4.12.2
  - setuptools
  - pip:
    - --extra-index-url https://download.pytorch.org/whl/
    - torch==2.6.0+rocm6.2.4
    - torchvision==0.21.0+rocm6.2.4
    - torchaudio==2.6.0+rocm6.2.4
````

Once we have the .yml file we can create the new installation directory and then run the conta-contanerized tool using this command given below.

````bash
$ mkdir <install_dir>
$ conda-containerize new --prefix <install_dir> env.yml
````

Aftert the installation is done, we will need to add the bin directory <install_dir>/bin to your PATH.

````bash
$ export PATH="<install_dir>/bin:$PATH"
````

Since, I named the installation directory to tykkyTest, the export command for me look something like this
````bash
export PATH="/scratch/<your-project-number>/tykkyTest/tykkyTest/bin:$PATH"
````

We can confirm the PyTorch version using this command:
`/scratch/<your-project-number>/tykkyTest/tykkyTest/bin/pip show torch`

Moreover, it is also possible to create the CONDA environment in your local machine with all the softwares you need and then transfer it to LUMI where you can then wrap it with the container wrapper. This should work without any issue.




# LUMI- container wrapper for running PyTorch script in single GPU, multi-GPU(single-node), multi-GPU(multiple-node)

## Single GPU

For this example we will focus on very simple model.We will first do the training on the single GPU and migrate it to multiple GPU using DDP.


In the single GPU code, we have the Trainer class that takes the model, the training data and the optimizer along with the device we are running on and how often we should be checking our model checkpoints.The training happens in the _run_epoch method and we call train specifying how many epochs we want to run our training job on. 
We have a few helper functions load_train_objs() which loads all of the objects we will need to for training process,namely the training dataset and model itself and an optimizer.For this specific example for the model , we are using the linear layer and the training dataset is just some random numbers.The optimizers is registered to all of the parameters of the model.Also there is a function, prepare_dataloader() which takes the dataset and wraps DataLoader around it.We can specify the batch size over there.

At first we want to run the script for 10 epochs and save it at every 2nd epoch.We are using the device = 0 which essentially lets Pytorch know to use the first available GPU.

The python codes and job script are given below.

singleGpu.py
````Python
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datautils import MyTrainDataset


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int, 
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)

    def _save_checkpoint(self, epoch):
        ckp = self.model.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch)


def load_train_objs():
    train_set = MyTrainDataset(2048)  # load your dataset
    model = torch.nn.Linear(20, 1)  # load your model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True
    )


def main(device, total_epochs, save_every, batch_size):
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, device, save_every)
    trainer.train(total_epochs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    
    device = 0  # shorthand for cuda:0
    main(device, args.total_epochs, args.save_every, args.batch_size) 
````


datautils.py
````python
import torch
from torch.utils.data import Dataset

class MyTrainDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.data = [(torch.rand(20), torch.rand(1)) for _ in range(size)]

    def __len__(self):
        return self.size
 
    def __getitem__(self, index):
        return self.data[index]
````


### Example of job script using Tykky

````bash
#!/bin/bash
#SBATCH --account=<your-project-number>
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=0:15:00

module purge

# module load rocm/6.0.3

#module use /appl/local/training/modules/AI-20241126/
# module load singularity-userfilesystems singularity-CPEbits

# module list
export PATH="/scratch/<your-project-number>/tykkyTest/tykkyTest/bin:$PATH"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
srun python ./singleGpu.py 10 2
````


### Example of job script using Container

````bash
#!/bin/bash
#SBATCH --account=<your-project-number>
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=0:15:00

module purge
module use /appl/local/training/modules/AI-20241126/
module load singularity-userfilesystems singularity-CPEbits

CONTAINER=/project/<your-project-number>/binod/tykky/pytorch_transformers.sif

srun singularity exec $CONTAINER  python ./singleGpu.py 10 2
````


### Output
In the output below, we will find that the it took 64 steps to run throught the whole training dataset for the batch size of 32.

````bash
Python path: /scratch/<your-project-number>/tykkyTest/tykkyTest/bin/python
Python version: Python 3.11.7
[GPU0] Epoch 0 | Batchsize: 32 | Steps: 64
Epoch 0 | Training checkpoint saved at checkpoint.pt
[GPU0] Epoch 1 | Batchsize: 32 | Steps: 64
[GPU0] Epoch 2 | Batchsize: 32 | Steps: 64
Epoch 2 | Training checkpoint saved at checkpoint.pt
[GPU0] Epoch 3 | Batchsize: 32 | Steps: 64
[GPU0] Epoch 4 | Batchsize: 32 | Steps: 64
Epoch 4 | Training checkpoint saved at checkpoint.pt
[GPU0] Epoch 5 | Batchsize: 32 | Steps: 64
[GPU0] Epoch 6 | Batchsize: 32 | Steps: 64
Epoch 6 | Training checkpoint saved at checkpoint.pt
[GPU0] Epoch 7 | Batchsize: 32 | Steps: 64
[GPU0] Epoch 8 | Batchsize: 32 | Steps: 64
Epoch 8 | Training checkpoint saved at checkpoint.pt
[GPU0] Epoch 9 | Batchsize: 32 | Steps: 64
Time for loading data and training: 29.94 seconds
````

## Multiple-GPU on a single node

We will not be discussing on the aspects of how we can use PyTorch Distributed Data Parallel framework here, but simple demonstrate the job script and output that utilizes it as our goal is to test if LUMI container- wrapper works in all cases.

### Example of job script using Tykky
````bash
#!/bin/bash
#SBATCH --account=<your-project-number>
#SBATCH --partition=standard-g
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=120G
#SBATCH --time=0:15:00


export PATH="/scratch/<your-project-number>/tykkyTest/tykkyTest/bin:$PATH"

srun torchrun --standalone --nnodes=1 --nproc_per_node=${SLURM_GPUS_PER_NODE} ./multigpu.py 50 10
````

### Output
In this distributed setup since we are using 2 GPUs, those 64 steps have halved to 32. i.e. the training load has been divided across 2 GPUs.

````bash
[GPU1] Epoch 1 | Batchsize: 32 | Steps: 32
Epoch 0 | Training snapshot saved at snapshot.pt
[GPU0] Epoch 1 | Batchsize: 32 | Steps: 32
[GPU1] Epoch 2 | Batchsize: 32 | Steps: 32
[GPU0] Epoch 2 | Batchsize: 32 | Steps: 32
[GPU1] Epoch 3 | Batchsize: 32 | Steps: 32
[GPU0] Epoch 3 | Batchsize: 32 | Steps: 32
````

## Multiple-GPU on a multiple node

### Example of job script using Tykky

````bash
cat start.sh 
#!/bin/bash
#SBATCH --account=<your-project-number>
#SBATCH --partition=standard-g
#SBATCH --nodes=2
#SBATCH --gpus-per-node=2
#SBATCH --time=00:10:00

#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G


export PATH="/scratch/<your-project-number>/tykkyTest/tykkyTest/bin:$PATH"
# Tell RCCL to use Slingshot interfaces and GPU RDMA
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500


# Run the distributed training job
srun torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    ./multinode.py 50 10
````

### Output
As you can see from the output that, since we are using the two nodes with two GPU each, the steps are reduced to 16.It means that the training load has been now divided into multiple nodes and multiple GPUs.

````bash
[GPU3] Epoch 0 | Batchsize: 32 | Steps: 16
[GPU2] Epoch 0 | Batchsize: 32 | Steps: 16
[GPU1] Epoch 1 | Batchsize: 32 | Steps: 16
[GPU3] Epoch 1 | Batchsize: 32 | Steps: 16
Epoch 0 | Training snapshot saved at snapshot.pt
````


## Summary
We also did some benchmarking on the time aspects. For all of our setup we compare the time taken to load the data and run the model using two different approach. i.e. Using the containerized approach and the LUMI container wrapper approach. Table below summarize it:

| Configuration                  | Container Wrapper time(secs) | Container time(secs) | Epoch Ran | Batch Size |
|--------------------------------|-------------------|-----------|-----------|------------|
| Single GPU                     | 29                | 8         | 10        | 32         |
| Multi GPU Single Node          | 19.06             | 10.28     | 50        | 32         |
| Multiple GPU, Multiple Node    | 22.39             | 11.52     | 50        | 32         |
