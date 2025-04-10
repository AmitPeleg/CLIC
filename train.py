import math
import os
import sys
import time
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.cuda.amp as amp
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

import losses
import utils.model_utils
from eval.general_eval import from_train
from models import get_model_clip
from utils import general_utils
from utils.train_args import get_args_parser
from utils.train_utils import get_optim_param_dict, get_scheduler_dict, create_normal_clip_loss_pattern, \
    backward_and_update, print_data, create_result_dict, create_result_dict_spp, set_data_type, \
    validate_zeroshot, AverageMeter, ProgressMeter, resume_from_ckpt, init_logging, load_train_dataset, \
    load_validation_dataset

cudnn.benchmark = True


def main(args):
    # Initialize distributed training if applicable
    general_utils.init_distributed_mode(args)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # fix the seed for reproducibility
    seed = args.seed + general_utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create model and move to GPU
    print(f"=> creating model: {args.model}")
    model, train_transform, val_transform, tokenizer = get_model_clip(args)
    model.cuda(args.gpu)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], bucket_cap_mb=200, find_unused_parameters=True
        )

    # Create model name with timestamp for logging and checkpoints
    now = datetime.now()
    model_name = f'{args.model}_{args.dataset}_dt_{now.month}_{now.day}_{now.hour}_{now.minute}_ep_{args.epochs}_{args.expname}'
    print(f"Model name: {model_name}")

    if general_utils.is_main_process():
        init_logging(args, model_name)
        print(args)

    # Set up optimizer
    optim_params = get_optim_param_dict(model, args)
    optimizer = torch.optim.AdamW(
        optim_params,
        lr=args.lr,
        betas=args.betas,
        eps=args.eps,
        weight_decay=args.wd,
    )

    scaler = amp.GradScaler(enabled=not args.disable_amp)

    # optionally resume from a checkpoint (takes precedence over autoresume)
    resume_from_ckpt(args, model, optimizer, scaler)

    # Data loading code
    print("=> creating dataset")
    data, train_loader = load_train_dataset(args, tokenizer, train_transform)
    val_loader = load_validation_dataset(args, val_transform)

    # Set up learning rate scheduler
    lr_scheduler_dict = get_scheduler_dict(args=args, loader_len=train_loader.num_batches)

    if not args.no_eval:
        val_stats = validate_zeroshot(val_loader, model, tokenizer, args)
        acc1 = val_stats["acc1"]
        print("Validation accuray of init: ", acc1)

    # Set up loss function
    criterion = losses.CLIPLoss(args).cuda(args.gpu)

    print("=> beginning training")
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data["train"].set_epoch(epoch)
        train_loader = data["train"].dataloader

        # train for one epoch
        train(
            train_loader,
            model,
            criterion,
            optimizer,
            scaler,
            epoch,
            lr_scheduler_dict,
            val_loader,
            tokenizer,
            val_transform,
            args,
        )

        # evaluate on imagenet, avoid evaluating on imagenet on the last epoch and when using no_eval
        if epoch != args.epochs - 1:
            if not args.no_eval:
                model.eval()
                validate_zeroshot(val_loader, model, tokenizer, args, mid=True)
                model.train()

        print("=> saving checkpoint")
        general_utils.save_on_master(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "args": args,
            },
            args.output_dir,
        )
    if general_utils.is_main_process():
        if not args.no_eval:
            model.eval()
            acc_imnet = from_train(args.output_dir, "imagenet", args)
            metrics_screpe = from_train(args.output_dir, "sugarcrepe", args)
            metrics_spp = from_train(args.output_dir, "sugarcrepe_pp", args)
            create_result_dict(args, acc_imnet, metrics_screpe)
            create_result_dict_spp(args, metrics_spp)
            coco_ret = from_train(args.output_dir, "coco2017_retrival", args)
            print(coco_ret)
            fk_ret = from_train(args.output_dir, "flickr30k_retrival", args)
            print(fk_ret)


def train(
        train_loader,
        model,
        criterion,
        optimizer,
        scaler,
        epoch,
        lr_scheduler_dict,
        val_loader,
        tokenizer,
        val_transform,
        args,
):
    """
    Train the model for one epoch

    Args:
        train_loader: DataLoader for training data
        model: Model to train
        criterion: Loss function
        optimizer: Optimizer for model parameters
        scaler: GradScaler for mixed precision training
        epoch: Current epoch number
        lr_scheduler_dict: Dictionary with learning rate schedules
        val_loader: DataLoader for validation data
        tokenizer: Text tokenizer
        val_transform: Validation data transforms
        args: Arguments
    """

    # Define metrics to track during training
    batch_time = AverageMeter("Time", ":5.2f")
    mem = AverageMeter("Mem (GB)", ":5.1f")
    metric_names = ["contrastive_loss", "clip_acc", 'combined_loss', 'uni_modal_loss', 'sneg_loss', 'img_to_txt_loss',
                    'txt_to_img_loss']
    metrics = OrderedDict([(name, AverageMeter(name, ":.3f")) for name in metric_names])

    # Get number of iterations per epoch
    loader_len = train_loader.num_batches
    iters_per_epoch = loader_len

    # Initialize progress meter for display
    progress = ProgressMeter(
        iters_per_epoch,
        # [batch_time, mem, *metrics.values()],
        [*metrics.values()],
        prefix=f"Epoch: [{epoch}]",
    )

    # Create loss pattern if using iterative CLIP loss
    data_pattern = create_normal_clip_loss_pattern(args.hard_negative_freq)

    # Switch to train mode
    model.train()

    # Start timing
    end = time.time()

    # Main training loop
    for data_iter, inputs_ in enumerate(train_loader):

        # Update data pattern if using iterative CLIP loss
        if args.clip_loss_iterate:
            set_data_type(data_pattern, data_iter, train_loader)

        # global training iteration
        it = (iters_per_epoch * epoch + data_iter)

        # Update learning rate according to schedule
        for k, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_scheduler_dict[param_group["name"]][it]

        # Handle hard negatives and additional positives if enabled
        if args.hard_negatives or args.additional_positives:
            if train_loader.dataset.clip_loss_iter == 1 and args.clip_loss_iterate:  # hard negatives
                # discard the last two elements of the inputs
                inputs_ = inputs_[:-2]
            elif train_loader.dataset.clip_loss_iter == -1 and args.clip_loss_iterate:  # simple images
                # discard all the elements of the inputs except the last two
                inputs_ = inputs_[-2:]
            inputs_[1] = torch.cat([input_k for input_k in inputs_[1:]], dim=0)
            inputs_ = inputs_[:2]

        inputs = [tensor.cuda(args.gpu, non_blocking=True) for tensor in inputs_]

        # Forward pass with mixed precision
        with amp.autocast(enabled=not args.disable_amp):
            outputs = model(*inputs)
            loss_dict = criterion(outputs)
            loss = loss_dict["combined_loss"]

        # Check for NaN or infinite loss
        if not math.isfinite(loss.item()):
            print(f"Loss is {loss.item()}, stopping training")
            sys.exit(1)

        # Backward pass and optimization (unless in validation-only mode)
        if not args.only_val:
            backward_and_update(loss, model, optimizer, scaler)

        # Clamp logit scale to prevent instability
        # exp(4.6052) ≈ 100, so this clamps to [1, 100]
        utils.model_utils.get_model(model).logit_scale.data.clamp_(0, 4.6052)

        # Update metrics
        for k in loss_dict:
            if k != 'cos-sim':
                metrics[k].update(loss_dict[k].item(), args.batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Track GPU memory usage
        mem.update(torch.cuda.max_memory_allocated() // 1e9)

        print_data(args, data_iter, loss_dict, metrics, model, optimizer, progress, tokenizer, val_loader)


if __name__ == "__main__":
    parser = general_utils.NewParser("CLIC", parents=[get_args_parser()])
    args = parser.parse_args()
    # Convert warmup epochs from relative to absolute
    args.warmup_epochs *= args.epochs

    main(args)
