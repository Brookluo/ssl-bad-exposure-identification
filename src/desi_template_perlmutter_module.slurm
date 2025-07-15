#!/bin/bash
#SBATCH -A desi
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -J inference
#SBATCH --mail-user=yourname@email.edu
#SBATCH --mail-type=ALL
#SBATCH -o /directory-path/tmp/inference/%j.out
#SBATCH -e /directory-path/tmp/inference/%j.err

# note the --ntasks-per-node and --gpus-per-task can be interchangable
# depending on whether torchrun or slurm manages the multinode tasks
# might need --gpu-bind=none to let all gpus available to all task

# initiate environment
# . /etc/profile
# module load gpu/1.0
module load pytorch/2.0.1
#module load cray-hdf5/1.12.2.3

# make directory if not exist
#[ ! -d $job_dir ] && mkdir $job_dir
#[ ! -d $chkpt_path ] && mkdir $chkpt_path

node_id=node${NODEIDX}
output_dir=/directory-path/dr11/${node_id}
[ ! -d $output_dir ] && mkdir $output_dir

src_dir=/directory-path/ssl-exposure-identification-paper/src
cd $src_dir

run_cmd="python master_eval.py -dir ${output_dir} \
                                --dset-path /directory-path/dr11/proc-data/${node_id}_dr11_sample.csv \
                                --dr dr11 \
                                --cont"

echo $run_cmd
srun -N 1 -c 32 -n 1 -G 4 --gpu-bind=none $run_cmd
