#!/bin/bash
set -eux
module load pytorch/2.6.0
node_id=node0
output_dir=/pscratch/sd/b/brookluo/decam-exposure/revision/${node_id}
[ ! -d $output_dir ] && mkdir $output_dir
src_dir=$HOME/ssl-exposure-identification-paper/src
export PYTHONPATH=$src_dir:$PYTHONPATH
python -m decam_qa.cli embed \
    --config $src_dir/../configs/embed.yaml \
    --dset /pscratch/sd/b/brookluo/decam-exposure/revision/proc-data/${node_id}_dr10_sample.csv \
    --dr dr10 \
    --cont
