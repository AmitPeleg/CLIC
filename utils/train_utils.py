import json
import os
import time

import numpy as np
import torch
import wandb
from torch import distributed as dist
from torchvision import datasets as datasets

import data_loading
import utils.general_utils
import utils.model_utils
from utils.model_utils import normalize_and_append


def get_optim_param_dict(model, args):
    all_params_wd, all_params_non_wd = split_param_list_to_wd_group(list(model.module.named_parameters()))

    optim_params = [
        {"params": all_params_wd, "weight_decay": args.wd, "name": "all_params", "is_wd": True},
        {"params": all_params_non_wd, "weight_decay": 0, "name": "all_params", "is_wd": False},
    ]

    return optim_params


def get_scheduler_dict(args, loader_len):
    return {
        "all_params": cosine_scheduler(
            args.lr,
            args.lr_end,
            args.epochs,
            loader_len + 1,
            warmup_epochs=args.warmup_epochs,
            start_warmup_value=args.lr_start,
        ),
    }


def split_param_list_to_wd_group(n_param_list):
    """
    Splits model parameters into weight decay (p_wd) and non-weight decay (p_non_wd) groups.
    non_wd group includes biases, BatchNorm and LayerNorm parameters.
    Args:
        n_param_list (list of tuples): (name, parameter) pairs.

    Returns:
        tuple: (p_wd, p_non_wd), where p_wd has weight decay and p_non_wd does not.
    """
    p_wd, p_non_wd = [], []
    for n, p in n_param_list:
        if not p.requires_grad:
            continue  # frozen weights
        if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
            p_non_wd.append(p)
        else:
            p_wd.append(p)

    return p_wd, p_non_wd


def cosine_scheduler(
        base_value,
        final_value,
        epochs,
        niter_per_ep,
        warmup_epochs=0,
        start_warmup_value=0,
):
    """
    Creates a learning rate schedule using a cosine decay with optional warmup.

    Args:
        base_value (float): Peak learning rate after warmup.
        final_value (float): Minimum learning rate at the end of training.
        epochs (int): Total number of epochs.
        niter_per_ep (int): Number of iterations per epoch.
        warmup_epochs (int, optional): Warmup period in epochs. Defaults to 0.
        start_warmup_value (float, optional): Initial warmup learning rate. Defaults to 0.

    Returns:
        np.ndarray: Learning rate schedule over all iterations.
    """

    # Warmup schedule (linear increase from start_warmup_value to base_value)
    warmup_schedule = np.array([])
    warmup_iters = int(warmup_epochs * niter_per_ep)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(
            start_warmup_value, base_value, warmup_iters
        )

    # Cosine schedule (decay from base_value to final_value)
    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (
            1 + np.cos(np.pi * iters / len(iters))
    )

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def create_normal_clip_loss_pattern(hard_negative_freq):
    """
   Generates a boolean pattern for iterating between the loss described in section 4 in the paper and standard clip loss iterations.
   - If hard_negative_freq == 1, returns [True, False].
   - If hard_negative_freq > 1, returns [False] followed by 'hard_negative_freq' True values.
   - If hard_negative_freq < 1, returns [True] followed by (1/hard_negative_freq) False values.

   Returns:
       list: A pattern of True/False values where True represents a hard negative sample.

       This list is used in set_data_type() to determine data_loader.dataset.clip_loss_iter .

   """
    if hard_negative_freq == 1:
        return [True, False]
    elif hard_negative_freq > 1:
        return [False] + [True] * int(hard_negative_freq)
    else:
        freq = int(1 / hard_negative_freq)
        return [True] + [False] * freq


def backward_and_update(loss, model, optimizer, scaler):
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    model.zero_grad(set_to_none=True)


def print_data(args, data_iter, loss_dict, metrics, model, optimizer, progress, tokenizer, val_loader):
    if data_iter % (args.print_freq * 30) == 0 and data_iter > 0:
        model.eval()
        validate_zeroshot(val_loader, model, tokenizer, args, mid=True)
        model.train()

    if data_iter % args.print_freq == 0 and data_iter > 0 or data_iter == 5:
        progress.display(data_iter)
        if utils.general_utils.is_main_process():
            log_data = {
                'data-iter': data_iter,
                'combined_loss': metrics['combined_loss'].avg,
                'clip_acc': metrics['clip_acc'].avg,
                'contrastive_loss': metrics['contrastive_loss'].avg,
                'img_to_txt_loss': metrics['img_to_txt_loss'].avg,
                'txt_to_img_loss': metrics['txt_to_img_loss'].avg,
                'uni-modal-loss': metrics['uni_modal_loss'].avg,
                'sneg_loss': metrics['sneg_loss'].avg,
                'cos-sim': loss_dict['cos-sim'].item(),
            }
            for param in optimizer.param_groups:
                if param["is_wd"]:
                    log_data[f"lr_{param['name']}"] = param["lr"]

            wandb.log(log_data)


def create_result_dict(args, acc, met):
    result1 = {
        'DATA': args.dataset,
        'epochs': args.epochs,
        'BS': args.batch_size,
        'IMNET': acc,
        'a-obj': met.pop('add_obj'),
        'a-att': met.pop('add_att'),
        'r-obj': met.pop('replace_obj'),
        'r-att': met.pop('replace_att'),
        'r-rel': met.pop('replace_rel'),
        's-obj': met.pop('swap_obj'),
        's-att': met.pop('swap_att'),
    }
    from assets.export_table import ResultsLogger
    logger = ResultsLogger('./assets/ablations_mix.csv', './assets/ablations_mix.tex')
    logger.add_result(result1)
    logger.get_latex_code()


def create_result_dict_spp(args, met):
    repl_ob = met.pop('replace_obj')
    repl_at = met.pop('replace_att')
    repl_rel = met.pop('replace_rel')
    swap_ob = met.pop('swap_obj')
    swap_at = met.pop('swap_att')
    result1 = {
        'DATA': args.dataset,
        'LR': args.lr,
        'epochs': args.epochs,
        'BS': args.batch_size,
        'R-ob(ITT)': repl_ob[0],
        'R-ob(TOT)': repl_ob[1],
        'R-at(ITT)': repl_at[0],
        'R-at(TOT)': repl_at[1],
        'R-rel(ITT)': repl_rel[0],
        'R-rel(TOT)': repl_rel[1],
        'S-ob(ITT)': swap_ob[0],
        'S-ob(TOT)': swap_ob[1],
        'S-at(ITT)': swap_at[0],
        'S-at(TOT)': swap_at[1],
    }
    from assets.export_table import ResultsLogger
    logger = ResultsLogger('./assets/ablations_scpp.csv', './assets/ablations_scpp.tex')
    logger.add_result(result1)
    logger.get_latex_code()


def set_data_type(data_pattern, data_iter, data_loader):
    """
    Sets the data type for the current iteration.
    :param data_pattern: list of boolean values generated in create_normal_clip_loss_pattern()
    :param data_iter: current iteration
    :param data_loader
    """
    data_index = data_iter % len(data_pattern)
    if data_pattern[data_index]:
        # print("Using hard negatives")
        data_loader.dataset.clip_loss_iter = 1
    else:
        # print("Using standard images")
        data_loader.dataset.clip_loss_iter = -1


def validate_zeroshot(val_loader, model, tokenizer, args, mid=False):
    batch_time = AverageMeter("Time", ":6.3f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(len(val_loader), [batch_time, top1, top5], prefix="Test: ")
    # switch to evaluate mode
    model.eval()

    print("=> ImageNet Evaluation")
    total_top1 = 0
    total_images = 0

    cwd = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join('eval', "datasets", "imagenet_templates.json")) as f:
        templates = json.load(f)
    with open(os.path.join("eval", "datasets", "imagenet_labels.json")) as f:
        labels = json.load(f)

    with torch.no_grad():
        text_features = []
        for l in labels:
            texts = [t.format(l) for t in templates]
            texts = tokenizer(texts).cuda(args.gpu, non_blocking=True)
            class_embeddings = utils.model_utils.get_model(model).encode_text(texts)

            normalize_and_append(class_embeddings, text_features)
        text_features = torch.stack(text_features, dim=0)
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # encode images
            image_features = utils.model_utils.get_model(model).encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # cosine similarity as logits
            logits_per_image = image_features @ text_features.t()

            pred = logits_per_image.argmax(dim=1)
            correct = pred.eq(target).sum()
            total_top1 += correct.item()
            total_images += images.size(0)

            acc1, acc5 = accuracy(logits_per_image, target, topk=(1, 5))
            # acc1 = 100 * total_top1 / total_images
            acc1, acc5 = utils.general_utils.scaled_all_reduce([acc1, acc5])

            top1.update(acc1.item(), total_images)
            top5.update(acc5.item(), total_images)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if i % args.print_freq == 0:
            #     progress.display(i)
            if mid and i >= 100:
                progress.display(i)
                break
    progress.synchronize()
    print(f"0-shot * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}")
    return {"acc1": top1.avg, "acc5": top5.avg, 'cos-sim': logits_per_image.mean()}


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def synchronize(self):
        if not utils.general_utils.is_dist_avail_and_initialized():
            return
        t = torch.tensor(
            [self.sum, self.count], dtype=torch.float64, device="cuda"
        )
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.sum = int(t[0])
        self.count = t[1]
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = (
                "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        )
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def synchronize(self):
        for meter in self.meters:
            meter.synchronize()

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = (
                correct[:k].reshape(-1).float().sum(0, keepdim=True)
            )
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def resume_from_ckpt(args, model, optimizer, scaler):
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading resume checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location="cpu")
            epoch = (checkpoint["epoch"] if "epoch" in checkpoint else 0)
            args.start_epoch = epoch
            result = model.load_state_dict(checkpoint["state_dict"], strict=True)  # was false
            print(result)
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                optimizer.load_state_dict(
                    checkpoint["optimizer"]
                ) if "optimizer" in checkpoint else ()
            scaler.load_state_dict(
                checkpoint["scaler"]
            ) if "scaler" in checkpoint else ()
            print(
                f"=> loaded resume checkpoint '{args.resume}' (epoch {epoch})"
            )
        else:
            print(f"=> no checkpoint found at '{args.resume}'")
    else:
        # auto-resume from latest checkpoint in output directory
        latest = os.path.join(args.output_dir, "checkpoint.pt")
        if os.path.isfile(latest):
            print(f"=> loading latest checkpoint '{latest}'")
            latest_checkpoint = torch.load(latest, map_location="cpu")
            args.start_epoch = latest_checkpoint["epoch"]
            model.load_state_dict(latest_checkpoint["state_dict"])
            optimizer.load_state_dict(latest_checkpoint["optimizer"])
            scaler.load_state_dict(latest_checkpoint["scaler"])
            print(
                "=> loaded latest checkpoint '{}' (epoch {})".format(
                    latest, latest_checkpoint["epoch"]
                )
            )


def init_logging(args, model_name):
    # Initialize W&B logging (only on main process)
    wandb.init(
        project="CLIC",
        config=vars(args),
        save_code=True,
        name=args.expname,
        mode="online" if args.wandb else "disabled",
    )
    # Set up output directory
    args.output_dir += f'/{model_name}'
    if args.resume:
        args.output_dir = os.path.dirname(args.resume)
        print(f"Resuming from {args.output_dir}")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


def load_validation_dataset(args, val_transform):
    if args.no_eval:
        return None
    # Validation dataset (ImageNet)
    val_dataset = datasets.ImageFolder(os.path.join(args.imagenet_root, "val"), transform=val_transform)
    # Set up validation data sampler
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset) if args.distributed else None
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=(val_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=val_sampler,
        drop_last=False,
    )
    return val_loader


def load_train_dataset(args, tokenizer, train_transform):
    # Load training data
    data = data_loading.get_data(args, train_transform, tokenizer=tokenizer)
    print("dataset size: %d" % data["train"].dataloader.num_samples)
    train_loader = data["train"].dataloader
    return data, train_loader
