#!/bin/bash
#SBATCH -A desi
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -J inference
#SBATCH --mail-user=${USER_EMAIL:-[EMAIL]}
#SBATCH --mail-type=ALL
#SBATCH -o ${OUTPUT_ROOT:-/path/to/output}/tmp/inference/%j.out
#SBATCH -e ${OUTPUT_ROOT:-/path/to/output}/tmp/inference/%j.err

# Required environment variables (set before sbatch):
#   PROJECT_ROOT  - path to this repository
#   OUTPUT_ROOT   - path for output/checkpoint directories
#   DSET_ROOT     - path to dataset CSVs
#   USER_EMAIL    - email for job notifications

# initiate environment
module load pytorch/2.0.1

node_id=node${NODEIDX}
output_dir=${OUTPUT_ROOT:-/path/to/output}/dr11/${node_id}
mkdir -p "$output_dir"

run_cmd="python ${PROJECT_ROOT:-.}/bin/master_eval \
    --output-dir ${output_dir} \
    --dset-path ${DSET_ROOT:-/path/to/data}/dr11/proc-data/${node_id}_dr11_sample.csv \
    --dr dr11 \
    --cont"

echo "$run_cmd"
srun -N 1 -c 32 -n 1 -G 4 --gpu-bind=none $run_cmd
