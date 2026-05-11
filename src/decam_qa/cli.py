"""Unified CLI entrypoint with embed/train/classify/evaluate subcommands."""
import argparse


def build_parser():
    parser = argparse.ArgumentParser(prog="decam_qa")
    sub = parser.add_subparsers(dest="command")

    embed_p = sub.add_parser("embed")
    embed_p.add_argument("--config", required=True)
    embed_p.add_argument("--dset", required=True)
    embed_p.add_argument("--dr", required=True)
    embed_p.add_argument("--cont", action="store_true")

    train_p = sub.add_parser("train")
    train_p.add_argument("--config", required=True)

    classify_p = sub.add_parser("classify")
    classify_p.add_argument("--config", required=True)
    classify_p.add_argument("--output", required=True)

    evaluate_p = sub.add_parser("evaluate")
    evaluate_p.add_argument("--predictions", required=True)
    evaluate_p.add_argument("--labels", required=True)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return
    print(f"decam_qa {args.command}: dispatch not yet implemented")
