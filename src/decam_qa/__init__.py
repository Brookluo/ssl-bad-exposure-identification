"""decam_qa — DECam exposure quality assessment toolkit."""
import importlib as _importlib

_LAZY = {
    'info': ['ccdname2num', 'ccdnum2name', 'reason_num_dict', 'reason_li',
             'reason_source_dict', 'decode_reason', 'decode_vi_source',
             'decode_ml_label', 'filter_dict', 'is_miss_2ccd'],
    'io': ['read_embeddings', 'read_fits_image', 'write_embeddings'],
    'dataset': ['DECamImageDataset'],
    'embeddings': ['create_model', 'generate_embeddings'],
    'classifier': ['build_pipeline', 'train', 'predict'],
    'pipeline': ['ParallelEvaluator'],
    'config': ['load_config', 'get_default_config'],
}


def __getattr__(name):
    if name == '_LAZY':
        raise AttributeError(name)
    for submod, names in _LAZY.items():
        if name in names:
            mod = _importlib.import_module(f'decam_qa.{submod}')
            attr = getattr(mod, name)
            globals()[name] = attr
            return attr
    raise AttributeError(f"module 'decam_qa' has no attribute '{name}'")
