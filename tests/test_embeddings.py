"""Tests for decam_qa.embeddings — DINOv2 model loading and embed generation."""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def mock_torch_hub():
    class MockModel:
        def eval(self):
            return self

        def cuda(self, device):
            return self

        def forward(self, x):
            class MockResult:
                def numpy(self, force=False):
                    return np.zeros((1, 768), dtype=np.float32)

            return MockResult()

    with patch("torch.hub.load", return_value=MockModel()) as mock:
        yield mock


class TestCreateModel:
    def test_create_model_base(self, mock_torch_hub):
        from decam_qa.embeddings import create_model

        create_model(size="base", use_register=True)
        mock_torch_hub.assert_called_once()
        assert "dinov2_vitb14_reg" in str(mock_torch_hub.call_args)

    def test_create_model_small(self, mock_torch_hub):
        from decam_qa.embeddings import create_model

        create_model(size="small", use_register=True)
        assert "vits14" in str(mock_torch_hub.call_args)

    def test_create_model_no_register(self, mock_torch_hub):
        from decam_qa.embeddings import create_model

        create_model(size="base", use_register=False)
        assert "_reg" not in str(mock_torch_hub.call_args)

    def test_create_model_invalid_size(self, mock_torch_hub):
        from decam_qa.embeddings import create_model

        with pytest.raises(KeyError):
            create_model(size="xxxl", use_register=True)


class TestGenerateEmbeddings:
    def test_generate_embeddings_output_count(self, tmp_path):
        import torch
        import h5py
        from decam_qa.embeddings import generate_embeddings

        model = MagicMock()
        result = MagicMock()
        result.numpy.return_value = np.zeros((1, 768), dtype=np.float32)
        model.forward.return_value = result

        dataset = MagicMock()
        dataset.df_data.original_df_idx = [0, 1, 2, 3]
        dataset.__len__.return_value = 4

        img = torch.zeros(1, 3, 224, 224)
        batches = [
            (img, torch.tensor([0])),
            (img, torch.tensor([1])),
            (img, torch.tensor([2])),
            (img, torch.tensor([0])),
        ]

        mock_loader_instance = MagicMock()
        mock_loader_instance.__iter__.return_value = iter(batches)

        with patch("decam_qa.embeddings.DataLoader", return_value=mock_loader_instance):
            generate_embeddings(
                dataset, model, "cpu", str(tmp_path), batch_size=1, num_workers=0
            )

        embeds_dir = tmp_path / "embeds_out"
        assert embeds_dir.exists()
        h5_files = list(embeds_dir.glob("*.h5"))
        assert len(h5_files) == 1
        with h5py.File(h5_files[0], "r") as h5f:
            assert len(h5f["images"]) == 4


class TestSingleChannel:
    def test_convert_preserves_numerics(self, fake_dino_model):
        """Replicated grayscale via channel sum must match single-channel path."""
        import torch
        from decam_qa.embeddings import convert_patch_embed_to_single_channel

        model = fake_dino_model
        original_proj = model.patch_embed.proj
        convert_patch_embed_to_single_channel(model)

        x1 = torch.randn(1, 1, 224, 224)

        out1 = model.patch_embed.proj(x1)

        summed_weight = original_proj.weight.sum(dim=1, keepdim=True)
        expected = torch.nn.functional.conv2d(x1, summed_weight, original_proj.bias,
                                              stride=original_proj.stride,
                                              padding=original_proj.padding)
        assert torch.allclose(out1, expected, atol=1e-5)

    def test_convert_accepts_single_channel(self, fake_dino_model):
        """After conversion, model accepts (1, 1, H, W) input."""
        import torch
        from decam_qa.embeddings import convert_patch_embed_to_single_channel

        model = fake_dino_model
        convert_patch_embed_to_single_channel(model)

        x = torch.randn(2, 1, 224, 224)
        out = model.patch_embed.proj(x)
        assert out.shape[1] == model.embed_dim
