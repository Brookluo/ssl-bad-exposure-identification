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
    embed_p.add_argument("--output-dir", default=None,
                         help="Override scratch_dir from config")

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

    if args.command == "embed":
        from decam_qa.config import load_config
        config = load_config(args.config, "embed")
        representation = config.get("representation", "ccd")

        if representation == "exposure_multiscale":
            from decam_qa.dataset import DECamExposureDataset
            from decam_qa.embeddings import (
                create_model, generate_exposure_multiscale_embeddings,
                convert_patch_embed_to_single_channel,
            )
            scratch = args.output_dir or config.get("scratch_dir", "./output")
            image_roots = config.get("image_roots", {})
            imdir = image_roots.get(args.dr, ".")

            ds = DECamExposureDataset(
                args.dset, imdir,
                binsize=config.get("focalplane_stamp", {}).get("binsize", 120),
                top_k=config.get("local_views", {}).get("top_k", 8),
            )
            model = create_model(config["model"]["version"], config["model"]["size"], config["model"]["use_register"])
            if config["model"].get("single_channel", True):
                convert_patch_embed_to_single_channel(model)

            generate_exposure_multiscale_embeddings(
                ds, model, device="cuda", output_dir=scratch,
                batch_size=config["data"]["batch_size"],
                num_workers=config["data"].get("num_workers", 4),
                top_k=config.get("local_views", {}).get("top_k", 8),
                crop_size=config.get("local_views", {}).get("crop_size"),
                resume=args.cont,
            )
            print("Embeddings generated successfully.")
        else:
            print(f"decam_qa {args.command}: dispatch not yet implemented")
    elif args.command == "train":
        print(f"decam_qa {args.command}: dispatch not yet implemented")
    elif args.command == "classify":
        print(f"decam_qa {args.command}: dispatch not yet implemented")
    elif args.command == "evaluate":
        print(f"decam_qa {args.command}: dispatch not yet implemented")


if __name__ == "__main__":
    main()
