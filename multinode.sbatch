#!/bin/bash
#SBATCH --job-name="elm"
#SBATCH --partition=gpu
#SBATCH --mem-per-cpu=16GB        # Amount of CPU memory
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8      # Crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=6           # Number of cores per tasks
#SBATCH --hint=nomultithread         # We get physical cores not logical
#SBATCH --gres=gpu:8                 # Number of gpus
#SBATCH --output=%x_%j.out   # Set this dir where you want slurm outs to go
#SBATCH --error=%x_%j.out    # Set this dir where you want slurm outs to go
#SBATCH --exclusive      # Turn off node sharing
#SBATCH --comment=elm
module purge
module load openmpi
module load cuda/11.4

mkdir -p /fsx/home-$(whoami)/hostfiles
hostfile=/fsx/home-$(whoami)/hostfiles/hosts_$SLURM_JOBID
rm $hostfile &> /dev/null # for consecutive calls to this script in interactive jobs

for i in `scontrol show hostnames $SLURM_NODELIST`
do
    echo $i slots=8 >>$hostfile
done

export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/nccl/build/lib:/opt/aws-ofi-nccl-install/lib
#export NCCL_PROTO=simple
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/aws-ofi-nccl/lib
#export PATH=$PATH:/opt/amazon/efa/bin:/opt/amazon/openmpi/bin
#export FI_EFA_FORK_SAFE=1
#export FI_LOG_LEVEL=1
#export FI_EFA_USE_DEVICE_RDMA=1 # use for p4dn
#export NCCL_DEBUG=info
#export OMPI_MCA_mtl_base_verbose=1
#export FI_EFA_ENABLE_SHM_TRANSFER=0
#export FI_PROVIDER=efa
#export FI_EFA_TX_MIN_CREDITS=64
#export NCCL_TREE_THRESHOLD=0
#export OMPI_MCA_pml="^cm"
#export OMPI_MCA_btl="tcp,self"
#export OMPI_MCA_btl_tcp_if_exclude="lo,docker1"
#export OMPI_MCA_plm_rsh_no_tree_spawn=1
export LD_LIBRARY_PATH=/opt/aws-ofi-nccl/lib:/opt/amazon/efa/lib64:/usr/local/cuda-11.0/efa/lib:/usr/local/cuda-11.0/lib:/usr/local/cuda-11.0/lib64:/usr/local/cuda-11.0:/opt/nccl/build/lib:/opt/aws-ofi-nccl-install/lib:/opt/aws-ofi-nccl/lib:$LD_LIBRARY_PATH
export PATH=/opt/amazon/efa/bin:/opt/amazon/openmpi/bin:$PATH
export LD_PRELOAD="/opt/nccl/build/lib/libnccl.so"

export NCCL_DEBUG=WARN
export NCCL_TREE_THRESHOLD=0
export NCCL_PROTO=simple
# Network issues without these set; See https://github.com/NVIDIA/nccl/issues/676
# export NCCL_P2P_DISABLE=1
export NCCL_IBEXT_DISABLE=1
export NCCL_SOCKET_IFNAME="eth0"


export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_EFA_USE_DEVICE_RDMA=1 # use for p4dn
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64

export PYTHONFAULTHANDLER=1

# export CUDA_LAUNCH_BLOCKING=1

export OMPI_MCA_mtl_base_verbose=1


export TORCH_EXTENSIONS_DIR=extensions
export XDG_CACHE_HOME=hf_cache
export WANDB_ENTITY=carper-elm
export WANDB_PROJECT=diff-model

source /fsx/home-honglu/miniconda3/bin/activate
#source /fsx/codeSeCodegen/codeSeEnv/bin/activate
conda activate training
#apt-get install libopenmpi-dev
#pip install mpi4py
#pip install -r requirements.txt
#conda install pytorch torchvision torchaudio pytorch-cuda=11.4 -c pytorch -c nvidia -y
#apt-get install -y pdsh
export TORCH_EXTENSIONS_DIR=extensions

deepspeed --launcher openmpi --hostfile $hostfile --master_addr $MASTER_ADDR  run_clm_diff.py --model_name_or_path=codegen-2b --per_device_train_batch_size=2 --ignore_long_samples --train_diff_model --save_final_dataset --concatenate_texts --num_train_epochs 1 --preprocessing_num_workers 25 --save_strategy=epoch --output_dir=diff_2b_full --report_to "wandb" --dataset_name final_dataset --load_from_disk --skip_concat --tokenizer_name codegen-2b --block_size 2048 --gradient_accumulation_steps 2 --do_train --logging_strategy=epoch --fp16 --overwrite_output_dir --adam_beta1=0.9 --adam_beta2=0.95 --weight_decay=2e-02 --learning_rate=1e-05 --warmup_steps=895 --per_device_eval_batch_size=1 --cache_dir="hf_cache_multinode" --gradient_checkpointing=True --deepspeed config_multinode.json 
