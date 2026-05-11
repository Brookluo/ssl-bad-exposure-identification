"""Tests for decam_qa.cli — argparse subcommand routing."""
import pytest
from decam_qa.cli import build_parser


class TestCLIHelp:
    def test_cli_help(self):
        parser = build_parser()
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["--help"])
        assert exc.value.code == 0

    def test_cli_subcommand_help(self):
        parser = build_parser()
        for sub in ["embed", "train", "classify", "evaluate"]:
            with pytest.raises(SystemExit) as exc:
                parser.parse_args([sub, "--help"])
            assert exc.value.code == 0


class TestCLIEmbed:
    def test_cli_embed_requires_config(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["embed", "--dset", "data.csv", "--dr", "dr10"])

    def test_cli_embed_requires_dset(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["embed", "--config", "config.yaml", "--dr", "dr10"])

    def test_cli_embed_all_args_parsed(self):
        parser = build_parser()
        args = parser.parse_args([
            "embed", "--config", "configs/embed.yaml",
            "--dset", "data.csv", "--dr", "dr10", "--cont"])
        assert args.config == "configs/embed.yaml"
        assert args.dset == "data.csv"
        assert args.dr == "dr10"
        assert args.cont is True


class TestCLITrain:
    def test_cli_train_all_args_parsed(self):
        parser = build_parser()
        args = parser.parse_args(["train", "--config", "configs/train.yaml"])
        assert args.config == "configs/train.yaml"


class TestCLIClassify:
    def test_cli_classify_all_args_parsed(self):
        parser = build_parser()
        args = parser.parse_args(["classify", "--config", "configs/inference.yaml", "--output", "results.csv"])
        assert args.config == "configs/inference.yaml"
        assert args.output == "results.csv"


class TestCLIEvaluate:
    def test_cli_evaluate_all_args_parsed(self):
        parser = build_parser()
        args = parser.parse_args(["evaluate", "--predictions", "pred.csv", "--labels", "truth.csv"])
        assert args.predictions == "pred.csv"
        assert args.labels == "truth.csv"
