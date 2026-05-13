"""DINOv2 model loading and embedding generation for DECam images."""
import json
from turtle import back
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import h5py
from pathlib import Path


def create_model(version='v2', size="base", use_register=True):
    """Load a pre-trained DINO Vision Transformer from PyTorch Hub.

    Parameters
    ----------
    version : str
        DINO version: currently only ``"v2"`` is supported.
    size : str
        Model size: ``"small"``, ``"base"``, ``"large"``, or ``"giant"``.
    use_register : bool
        If True, load the register variant (e.g. ``dinov2_vitb14_reg``).

    Returns
    -------
    torch.nn.Module
        DINO model set to evaluation mode.
    """
    assert version in ["v2", "v3"], f"Unsupported DINO version: {version}"
    assert size in ["small", "base", "large", "giant"], f"Unsupported size: {size}"
    backbone_archs = {
        "small": "vits14" if version == "v2" else "vits16plus",
        "base": "vitb14" if version == "v2" else "vitb16",
        "large": "vitl14" if version == "v2" else "vitl16",
        "giant": "vitg14" if version == "v2" else "vith16plus",
    }
    backbone_arch = backbone_archs[size]
    reg = ""
    backbone_name = f"dino{version}_{backbone_arch}"
    if version == "v2":
        reg = "_reg" if use_register else ""
        backbone_name = backbone_name + reg
        model = torch.hub.load(repo_or_dir=f"facebookresearch/dino{version}", model=backbone_name)
    else:
        REPO_DIR = Path("/pscratch/sd/b/brookluo/dino-models")
        dinov3_name = {
            "vits16plus": "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
            "vitb16": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
            "vitl16": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
            "vith16plus": "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth",
        }
        model = torch.hub.load(str(REPO_DIR / "dinov3") , backbone_name, source='local', weights=str(REPO_DIR / dinov3_name[backbone_arch]))
    model.eval()
    return model


def generate_embeddings(dataset, model, device, output_dir,
                        batch_size=2, num_workers=2, gpu_idx=0,
                        crop_size=(1540, 1540)):
    """Generate DINOv2 embeddings for a dataset and write them to HDF5.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        DECamImageDataset instance.
    model : torch.nn.Module
        Pre-loaded DINOv2 model.
    device : str
        Device string (e.g. ``"cpu"`` or ``"cuda"``).
    output_dir : pathlib.Path or str
        Output directory; ``embeds_out/`` subdirectory is created inside it.
    batch_size : int
        DataLoader batch size.
    num_workers : int
        Number of data loading workers.
    gpu_idx : int
        GPU index used for naming output file and CUDA device selection.
    crop_size : tuple of (int, int)
        (height, width) to resize images to before running the model.
    """
    output_dir = Path(output_dir)
    embeds_dir = output_dir / "embeds_out"
    embeds_dir.mkdir(exist_ok=True)

    is_cuda = device != "cpu" and torch.cuda.is_available()
    if is_cuda:
        cuda_device = torch.device("cuda", gpu_idx)
        model = model.cuda(cuda_device)
    else:
        cuda_device = None

    resize = transforms.Resize(crop_size, antialias=False)
    sampler = torch.utils.data.SequentialSampler(dataset)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        sampler=sampler,
                        pin_memory=False,
                        drop_last=False)

    h5fname = f"{gpu_idx}_worker_embeds.h5"
    with h5py.File(embeds_dir / h5fname, 'a') as h5f:
        if "images" not in h5f.keys():
            h5f.create_group("images")

    with torch.no_grad():
        for step, (img, label) in enumerate(loader):
            img = img.expand(-1, 3, -1, -1)
            im = resize(img)
            if is_cuda:
                im = im.to(cuda_device)
            embeds = model.forward(im).numpy(force=True)
            for i, onelab in enumerate(label):
                orig_idx = step * batch_size + i
                name = f'idx_{orig_idx:d}_label_{onelab.item():d}'
                with h5py.File(embeds_dir / h5fname, 'r+') as h5f:
                    h5f['images'].create_dataset(name, data=embeds[i], dtype='float')


def convert_patch_embed_to_single_channel(model):
    """Replace 3-channel DINOv2 patch projection with 1-channel equivalent.

    Sums the existing 3-channel conv weights so a single-channel grayscale
    input produces identical patch embeddings. Copy bias unchanged.

    This is exact only when no per-channel RGB normalization is applied.

    Parameters
    ----------
    model : torch.nn.Module
        DINOv2 model with a ``patch_embed.proj`` Conv2d attribute.
    """
    old_conv = model.patch_embed.proj
    new_conv = torch.nn.Conv2d(
        in_channels=1,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    with torch.no_grad():
        new_conv.weight.copy_(old_conv.weight.sum(dim=1, keepdim=True))
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)
    model.patch_embed.proj = new_conv


def generate_exposure_multiscale_embeddings(
    dataset, model, device, output_dir,
    batch_size=1, num_workers=2, top_k=8,
    crop_size=None, resume=False, overwrite=False,
):
    """Generate multi-scale embeddings for exposure-grouped data.

    For each exposure: embed the global stamp and top-K local views.
    Store one HDF5 group per exposure with global, local_N, and metadata.

    Parameters
    ----------
    dataset : DECamExposureDataset
    model : torch.nn.Module
    device : str
    output_dir : str
    batch_size : int
    num_workers : int
    top_k : int
    crop_size : tuple or None
        If None, resizes local crops to (2352, 1176).
    resume : bool
        Skip exposure groups that are already complete (global + expected number
        of locals + metadata). Partially-written groups are deleted and regenerated.
    overwrite : bool
        If True and resume=False, overwrite existing.
    """
    output_dir = Path(output_dir)
    embeds_dir = output_dir / "embeds_out"
    embeds_dir.mkdir(exist_ok=True)

    is_cuda = device != "cpu" and torch.cuda.is_available()
    if is_cuda:
        model = model.cuda()
    else:
        model = model.cpu()

    if crop_size is None:
        crop_size = (2352, 1176)
    resize = transforms.Resize(crop_size, antialias=False)

    h5path = embeds_dir / "0_worker_embeds.h5"

    with torch.no_grad():
        for i in range(len(dataset)):
            item = dataset[i]
            expnum = item["expnum"]

            if resume:
                if h5path.exists():
                    needs_regeneration = False
                    with h5py.File(h5path, 'r') as f:
                        grp = f.get("exposures", {}).get(f"exp_{expnum}")
                        if grp is not None:
                            has_global = "global" in grp
                            n_locals = sum(1 for k in grp.keys() if k.startswith("local_"))
                            has_metadata = "metadata" in grp
                            n_selected = len(item.get("selected_ccds", []))
                            if has_global and n_locals == n_selected and has_metadata:
                                continue
                            else:
                                needs_regeneration = True
                    if needs_regeneration:
                        with h5py.File(h5path, 'r+') as del_f:
                            grp_path = f"exposures/exp_{expnum}"
                            if grp_path in del_f:
                                del del_f[grp_path]

            global_tensor = torch.from_numpy(item["global_stamp"]).unsqueeze(0)
            if is_cuda:
                global_tensor = global_tensor.cuda()
            global_emb = model.forward(global_tensor).numpy(force=True).flatten()

            local_embs = []
            for sel_img in item.get("selected_images", []):
                if sel_img.ndim == 2:
                    img_t = torch.from_numpy(sel_img).unsqueeze(0).unsqueeze(0).float()
                else:
                    img_t = torch.from_numpy(sel_img).float()
                img_t = resize(img_t)
                if is_cuda:
                    img_t = img_t.cuda()
                emb = model.forward(img_t).numpy(force=True).flatten()
                local_embs.append(emb)

            with h5py.File(h5path, 'a') as f:
                if "exposures" not in f:
                    f.create_group("exposures")
                exp_grp = f["exposures"].create_group(f"exp_{expnum}")
                exp_grp.create_dataset("global", data=global_emb, dtype='float')
                for j, lem in enumerate(local_embs):
                    exp_grp.create_dataset(f"local_{j:03d}", data=lem, dtype='float')
                meta = {
                    "num_selected_views": len(item.get("selected_ccds", [])),
                    "filter": item.get("filter", 0),
                }
                exp_grp.create_dataset("metadata", data=json.dumps(meta))
