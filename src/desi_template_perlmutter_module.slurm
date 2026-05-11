#!/bin/bash
#SBATCH -A desi
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -J inference
#SBATCH --mail-user=yourname@email.edu
#SBATCH --mail-type=ALL
#SBATCH -o /pscratch/sd/b/brookluo/decam-exposure/revision/tmp/inference/%j.out
#SBATCH -e /pscratch/sd/b/brookluo/decam-exposure/revision/tmp/inference/%j.err

# note the --ntasks-per-node and --gpus-per-task can be interchangable
# depending on whether torchrun or slurm manages the multinode tasks
# might need --gpu-bind=none to let all gpus available to all task

set -eux
# initiate environment
# . /etc/profile
# module load gpu/1.0
#module load cray-hdf5/1.12.2.3

# make directory if not exist
#[ ! -d $job_dir ] && mkdir -p $job_dir
#[ ! -d $chkpt_path ] && mkdir -p $chkpt_path

src_dir=$HOME/ssl-exposure-identification-paper/src
cd $src_dir

srun -N 1 -c 128 -n 1 -G 4 --gpu-bind=none $src_dir/run-dist-job.sh
