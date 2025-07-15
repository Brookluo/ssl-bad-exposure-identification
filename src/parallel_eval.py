import torch
from torch.utils.data import DataLoader

# import sys
# sys.path.append("../src")
import decam_info
from decam_dataset import DECamImageDataset

from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from torchvision import transforms
import argparse


def get_arguments():
    parser = argparse.ArgumentParser(description="Inference with trained model", add_help=False)
    parser.add_argument("--dset-path", type=Path, default="/path/to/imagenet", required=True,
                        help='Path to the *.csv file')
    parser.add_argument("--exp-dir", type=Path, default="./exp", required=True,
                        help='Path to the experiment folder, where all logs/checkpoints will be stored')
    parser.add_argument("--imdir", type=Path, default="./exp", required=True,
                        help='Path to the image directory')
    
    parser.add_argument("--model-size", choices=["small", "base", "large", "giant"], default="base",
                        help="Size of the model to use")
    parser.add_argument("--batch-size", type=int, default=2, required=True,
                        help='batch size for inferencing')
    parser.add_argument("--crop-size", nargs=2, default=(1540, 1540), type=int,
                       help="crop size for the five crops on the image")
    
    parser.add_argument("--num-workers", type=int, default=2,
                       help="Number of workers to load data")
    parser.add_argument("--gpu-idx", type=int, default=0,
                       help="GPU index to use")
    return parser

def split_dset(dset_path, dst_dir, num_parts, keep_index=True):
    # num_gpu = torch.cuda.device_count()
    tmp_dir = dst_dir #/ 'tmp'
    tmp_dir.mkdir(exist_ok=True, parents=True)
    df = pd.read_csv(dset_path)
    idx_length = len(df) // num_parts
    for i in range(num_parts):
        if i == num_parts - 1:
            # last part
            tmp_df = df.iloc[i*idx_length:]
        else:
            tmp_df = df.iloc[i*idx_length:(i+1)*idx_length]
        if keep_index:
            tmp_df.insert(0, "original_df_idx", tmp_df.index)
        tmp_df.to_csv(tmp_dir / f"{i}_worker_sample.csv", index=False)

def create_model(BACKBONE_SIZE, use_register=True):
    # BACKBONE_SIZE = "base" # in ("small", "base", "large" or "giant")
    backbone_archs = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
    }
    backbone_arch = backbone_archs[BACKBONE_SIZE]
    reg = ""
    if use_register:
        reg = "_reg"
    backbone_name = f"dinov2_{backbone_arch}{reg}"

    model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
    model.eval()
    return model

def gen_embeds(model, exp_dir, imdir, args):
    # data_dir = root_dir / "data"
    ckpt_dir = exp_dir / "checkpoint"
    # test_dir = data_dir / "test"

    # eval_dir = exp_dir / "eval"
    # if not eval_dir.exists():
        # eval_dir.mkdir()

    embeds_dir = exp_dir / "embeds_out"
    if not embeds_dir.exists():
        embeds_dir.mkdir()
    tmp_dir = exp_dir / "tmp"
    # dr10_dir = Path('/global/cfs/cdirs/cosmo/work/legacysurvey/dr10')
    # dr10_imdir = dr10_dir / "images"
    
    if torch.cuda.is_available():
        print("Using GPU")
        gpu = torch.device("cuda", args.gpu_idx)
        model = model.cuda(gpu)

    # (2046 // 14) * 14 = 2044
    # (4096 // 14) * 14 = 4088
    # center_crop = transforms.CenterCrop((4088, 2044))
    # five_crop = transforms.FiveCrop(args.crop_size)
    # resize = transforms.Resize((int(14 * 168), int(14 * 84)), antialias=False)
    resize = transforms.Resize(args.crop_size, antialias=False)
    dataset = DECamImageDataset(
        # tmp_dir / f"{args.gpu_idx}_worker_sample.csv",
        args.dset_path,
        image_dir=imdir, seed=0
    )
    sampler = torch.utils.data.SequentialSampler(dataset)
    loader = DataLoader(dataset, 
                        batch_size=args.batch_size, 
                        num_workers=args.num_workers,
                       sampler=sampler,
                       pin_memory=False)
    h5fname = f"{args.gpu_idx}_worker_embeds.h5"
    with h5py.File(embeds_dir / h5fname, 'a') as h5f:
        if "images" not in h5f.keys():
            h5f.create_group("images")
    with torch.no_grad():
        for step, (img, label) in enumerate(loader):
            img = img.expand(-1, 3, -1, -1)
            im = resize(img)
            # imgs = five_crop(img)
            # Tuple[Tensor] of size 5
            # each Tensor is B, C, H, W
            if gpu:
                im = im.to(gpu)
                # spec = spec.type(torch.cuda.HalfTensor)
            embeds = model.forward(im).numpy(force=True)
            for i, onelab in enumerate(label):
                # five_embeds = [em[i] for em in all_embeds]
                # always stick original dataframe's index to view data clearer
                orig_idx = dataset.df_data.original_df_idx[step*args.batch_size + i]
                name = f'idx_{orig_idx:d}_label_{onelab.item():d}'
                with h5py.File(embeds_dir / h5fname, 'r+') as h5f:
                    h5f['images'].create_dataset(name, data=embeds, dtype='float')


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Generate feature vector with trained models', parents=[get_arguments()])
    args = parser.parse_args()
    print("Generating:")
    print(args)
    model = create_model(args.model_size, use_register=True)
    gen_embeds(model, args.exp_dir, args.imdir, args)
