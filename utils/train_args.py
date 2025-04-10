import argparse
import pathlib

import local_setting


def get_args_parser():
    parser = argparse.ArgumentParser(
        description="CLIC training", add_help=False
    )

    parser.add_argument(
        "--baseline",
        action="store_true",
        help="set to True for running negclip like our method",
    )

    parser.add_argument(
        "--hard_negatives_separate",
        action="store_true",
        help="set to True for loss of hard negative to be only with the corresponding text, Eq 5 in the paper",
    )

    parser.add_argument(
        "--clip_loss_iterate",
        action="store_true",
        help="set to True for doing the standard clip loss in part of the iterations",
    )

    parser.add_argument(
        "--hard_negative_freq",
        type=float,
        default=1,
        help="if using clip_loss_iterate, how many times to take the hard negatives and how much the standard clip loss",
    )

    parser.add_argument(
        "--limit_dataset",
        type=int,
        default=None,
        help="limit the size of the dataset, None for no limit and k for k samples",
    )

    parser.add_argument(
        "--additional_positives",
        type=int,
        default=0,
        help="how many additional positives to use (not including the shuffled one), 2 in the paper (p_3, p_4)",
    )

    parser.add_argument(
        "--cont",
        type=float,
        default=0.5,
        help="how much to multiply the contrastive loss, eq 7 in the paper",
    )

    parser.add_argument(
        "--sneg",
        type=float,
        default=0.5,
        help="how much to multiply the negative loss, eq 7 in the paper",
    )

    parser.add_argument(
        "--uni",
        type=float,
        default=1,
        help="how much to multiply the uni-modal-loss, eq 7 in the paper",
    )

    parser.add_argument(
        "--only_val",
        action="store_true",
        help="set to True for skip training",
    )

    parser.add_argument(
        "--no_eval",
        action="store_true",
        help="set to True to avoid evaluation during training",
    )

    parser.add_argument(
        "--no_concat",
        action="store_true",
        help="set to True for using standard clip training without concatenation",
    )

    parser.add_argument(
        "--shuffled_positive",
        action="store_true",
        help="set to True for using the shuffled positive (p_2) in the paper",
    )

    parser.add_argument("--model", default="ViT-L-14", type=str)
    parser.add_argument("--architecture", default="ViT-L-14", type=str)

    parser.add_argument("--dataset", default="laion_cogvlm", type=str, )

    parser.add_argument(
        "--root",
        type=pathlib.Path,
        default=local_setting.TRAIN_DATA_DIR,
        help="Root directory of images.",
    )

    parser.add_argument(
        "--hard_negatives",
        action="store_true",
        help="set to True for training with hard negatives",
    )

    parser.add_argument(
        "--uni_modal_loss",
        action="store_true",
        help="set to True for training with uni_modal_loss",
    )

    parser.add_argument("--resume", default="", type=str, help="path to resume from")
    parser.add_argument("--expname", default="", type=str)

    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--warmup-epochs", default=0.2, type=float)
    parser.add_argument("--start-epoch", default=0, type=int)
    parser.add_argument(
        "--batch-size",
        default=200,
        type=int,
        help="number of samples per-gpu",
    )

    parser.add_argument("--betas", default=(0.9, 0.98), nargs=2, type=float)
    parser.add_argument("--eps", default=1e-8, type=float)
    parser.add_argument(
        "--disable-amp",
        action="store_true",
        help="disable mixed-precision training (requires more memory and compute)",
    )

    parser.add_argument(
        "--freeze_only_vision",
        action="store_true",
        help="freezes only the vision encoder",
    )

    parser.add_argument(
        "--freeze_only_text",
        action="store_true",
        help="freezes only the text encoder",
    )

    parser.add_argument(
        "--print-freq", default=50, type=int, help="print frequency"
    )

    parser.add_argument(
        "--imagenet-root",
        default=str(local_setting.IMAGENET_DIR),
        type=str,
        help="path to imagenet dataset",
    )

    parser.add_argument(
        "--coco2017_image_root",
        default=str(local_setting.COCO2017_DIR),
        type=str,
        help="path to coco val2017 dataset",
    )

    parser.add_argument(
        "--output-dir",
        default=str(local_setting.OUTPUT_DIR),
        type=str,
        help="output dir",
    )

    parser.add_argument(
        "--load-pretrained-clip", default=None, type=str, help="Load from a pretrained model or None?"
    )

    parser.add_argument(
        "-j",
        "--workers",
        default=8,
        type=int,
        metavar="N",
        help="number of data loading workers per process",
    )
    parser.add_argument(
        "--world-size",
        default=4,
        type=int,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--rank",
        default=0,
        type=int,
        help="node rank for distributed training",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("--dist-backend", default="nccl", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")

    parser.add_argument("--lr", default=1e-6, type=float)
    parser.add_argument("--lr-start", default=1e-7, type=float, help="initial warmup lr", )
    parser.add_argument("--lr-end", default=1e-8, type=float, help="minimum final lr")

    parser.add_argument("--wd", default=0.1, type=float)
    parser.add_argument("--wandb", type=bool, default=True)

    return parser
