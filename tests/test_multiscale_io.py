"""Tests for multi-scale HDF5 I/O."""
import numpy as np
import h5py
import pytest
from decam_qa.io import read_exposure_embeddings


class TestReadExposureEmbeddings:
    def test_reads_global_and_locals(self, tmp_path):
        fpath = tmp_path / "0_worker_embeds.h5"
        rng = np.random.default_rng(42)
        with h5py.File(fpath, 'w') as f:
            grp = f.create_group("exposures")
            for eid in [1, 2]:
                eg = grp.create_group(f"exp_{eid}")
                eg.create_dataset("global", data=rng.normal(0, 1, 768).astype(np.float32))
                for i in range(3):
                    eg.create_dataset(f"local_{i:03d}", data=rng.normal(0, 1, 768).astype(np.float32))
                import json
                eg.create_dataset("metadata", data=json.dumps({"filter": 1}))

        result = read_exposure_embeddings(str(tmp_path))
        assert len(result) == 2
        for expnum, data in result.items():
            assert "global" in data
            assert len(data["locals"]) > 0
            assert "metadata" in data

    def test_merges_across_workers(self, tmp_path):
        """Two worker files with different exposures should be merged."""
        rng = np.random.default_rng(42)
        for w in range(2):
            fpath = tmp_path / f"{w}_worker_embeds.h5"
            with h5py.File(fpath, 'w') as f:
                grp = f.create_group("exposures")
                eid = w + 1
                eg = grp.create_group(f"exp_{eid}")
                eg.create_dataset("global", data=rng.normal(0, 1, 768).astype(np.float32))
                eg.create_dataset("local_000", data=rng.normal(0, 1, 768).astype(np.float32))
                import json
                eg.create_dataset("metadata", data=json.dumps({"filter": 1}))

        result = read_exposure_embeddings(str(tmp_path))
        assert len(result) == 2
        assert 1 in result and 2 in result
