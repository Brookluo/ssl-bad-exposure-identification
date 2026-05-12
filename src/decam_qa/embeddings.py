"""DINOv2 model loading and embedding generation for DECam images."""
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import h5py
from pathlib import Path


def create_model(size="base", use_register=True):
    """Load a pre-trained DINOv2 Vision Transformer from PyTorch Hub.

    Parameters
    ----------
    size : str
        Model size: ``"small"``, ``"base"``, ``"large"``, or ``"giant"``.
    use_register : bool
        If True, load the register variant (e.g. ``dinov2_vitb14_reg``).

    Returns
    -------
    torch.nn.Module
        DINOv2 model set to evaluation mode.
    """
    backbone_archs = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
    }
    backbone_arch = backbone_archs[size]
    reg = "_reg" if use_register else ""
    backbone_name = f"dinov2_{backbone_arch}{reg}"
    model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
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
