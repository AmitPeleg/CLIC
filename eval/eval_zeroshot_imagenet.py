import json
import os.path
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from torchvision import datasets as datasets

from eval.build_dataset import build_dataset
from eval.datasets.builder import get_dataset_collate_fn
from eval.eval_utils import calc_avg_acc, shuffle_and_select
from utils.model_utils import normalize_and_append


def get_model(model):
    if isinstance(model, torch.nn.DataParallel) or isinstance(
            model, torch.nn.parallel.DistributedDataParallel
    ):
        return model.module
    else:
        return model


def validate_zeroshot_generic(
        val_loader, templates, labels, model, tokenizer, tot=-1, device='cuda'
):
    # switch to evaluate mode
    model.eval()
    total_top1 = 0
    total_images = 0
    print("... encoding class embeddings")
    with torch.no_grad():
        text_features = []
        for label in labels:
            if isinstance(label, list):
                texts = [t.format(l) for t in templates for l in label]
            else:
                texts = [t.format(label) for t in templates]

            texts = tokenizer(texts).to(device).contiguous()
            texts = texts.view(-1, model.context_length).contiguous()
            class_embeddings = get_model(model).encode_text(texts)

            normalize_and_append(class_embeddings, text_features)
        text_features = torch.stack(text_features, dim=0)
        print("... encoding images")
        for images, target in val_loader:
            images = images.to(device)
            target = target.to(device)

            # encode images
            image_features = get_model(model).encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # cosine similarity as logits
            logits_per_image = image_features @ text_features.t()

            # measure accuracy and record loss
            pred = logits_per_image.argmax(dim=1)
            correct = pred.eq(target).sum()
            total_top1 += correct.item()
            total_images += images.size(0)

            if total_images == tot * 100:
                break

    return 100 * total_top1 / total_images


def validate_zeroshot(
        val_loader, templates, labels, model, tokenizer
):
    # switch to evaluate mode
    model.eval()
    total_top1 = 0
    total_images = 0

    print("... encoding class embeddings")
    with torch.no_grad():
        text_features = []
        for label in labels:
            if isinstance(label, list):
                texts = [t.format(l) for t in templates for l in label]
            else:
                texts = [t.format(label) for t in templates]

            texts = tokenizer(texts).cuda(non_blocking=True)
            texts = texts.view(-1, model.context_length).contiguous()
            class_embeddings = get_model(model).encode_text(texts)

            normalize_and_append(class_embeddings, text_features)
        text_features = torch.stack(text_features, dim=0)
        print("... encoding images")
        for images, target in val_loader:
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # encode images
            image_features = get_model(model).encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # cosine similarity as logits
            logits_per_image = image_features @ text_features.t()

            # measure accuracy and record loss
            pred = logits_per_image.argmax(dim=1)
            correct = pred.eq(target).sum()
            total_top1 += correct.item()
            total_images += images.size(0)
    return 100 * total_top1 / total_images


def zero_shot_imagenet(args, model, model_name, pretrained_checkpoint, preprocess_val, tokenizer):
    ###### imagenet specific ##########
    with open(os.path.join("eval", "datasets", "imagenet_labels.json")) as f:
        labels = json.load(f)
    # exit()
    # Data loading code
    print("... creating dataset")
    val_dataset = datasets.ImageFolder(
        os.path.join(args.imagenet_root, "val"),
        transform=preprocess_val,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
        drop_last=False,
    )
    templates = json.load(open(os.path.join('eval', "datasets", "imagenet_templates.json")))
    acc = validate_zeroshot(val_loader, templates, labels, model, tokenizer)

    # save the results
    result_records = {"Model": f'{model_name}-{pretrained_checkpoint}', "acc": acc}
    # also dump to model json
    # json.dump(result_records, open(os.path.join(args.output, f'{model_name}-{pretrained_checkpoint}.json'), 'a+'), indent=4)

    df = pd.DataFrame(result_records, index=[0])
    # create output folder if not exists
    imagenet_output_dir = os.path.join(args.output, "imagenet")
    if not os.path.exists(imagenet_output_dir):
        os.makedirs(imagenet_output_dir)
    output_file = os.path.join(imagenet_output_dir, "imagenet.csv")
    print(f"Saving results to {output_file}")
    if os.path.exists(output_file):
        all_df = pd.read_csv(output_file, index_col=0)
        all_df = pd.concat([all_df, df])
        all_df.to_csv(output_file)

    else:
        df.to_csv(output_file)

    return acc


def zero_shot_classifications(args, model, model_name, pretrained_checkpoint, preprocess_val, tokenizer, num_samples,
                              dataset_name, output_file):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Data loading code
    print("... creating dataset")
    dataset_root = args.zs_dataset_root

    dataset = build_dataset(
        dataset_name=dataset_name,
        root=dataset_root,
        transform=preprocess_val,
        num_samples=num_samples,
        download=False,
    )

    collate_fn = get_dataset_collate_fn(dataset_name)
    if True:
        try:
            print(f"Dataset size: {len(dataset)}")
        except TypeError:
            print("IterableDataset has no len()")
        # print(f"Dataset split: {args.split}")
        if hasattr(dataset, "classes") and dataset.classes:
            try:
                print(f"Some Dataset classes: {dataset.classes[:10]}...")
                print(f"Dataset number of classes: {len(dataset.classes)}")

            except AttributeError:
                print("Dataset has no classes.")

    sampler = shuffle_and_select(len(dataset), n=num_samples)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
        sampler=sampler,
    )

    zeroshot_templates = dataset.templates if hasattr(dataset, "templates") else None

    classnames = dataset.classes if hasattr(dataset, "classes") else None
    assert (zeroshot_templates is not None and classnames is not None), "Dataset does not support classification"

    print(f"Evaluating {model_name}-{pretrained_checkpoint} on zero-shot classification dataset {dataset_name}")

    acc = validate_zeroshot_generic(
        dataloader, zeroshot_templates, classnames, model, tokenizer
    )

    # save the results
    result_records = {"Model": f'{model_name}-{pretrained_checkpoint}', "dataset": dataset_name, "acc": acc}

    df = pd.DataFrame(result_records, index=[0])
    print(f"Saving results to {output_file}")
    if os.path.exists(output_file):
        all_df = pd.read_csv(output_file, index_col=0)
        all_df = pd.concat([all_df, df])
        all_df.to_csv(output_file)

    else:
        df.to_csv(output_file)

    return acc


def zero_shot_all_classifications(args, model, model_name, pretrained_checkpoint, preprocess_val, tokenizer,
                                  num_samples):
    model.eval()

    output_dir = Path(os.path.join(args.output, "zero_shot_classification"))
    output_dir.mkdir(exist_ok=True, parents=True)
    output_file = os.path.join(output_dir, f"{model_name}-{pretrained_checkpoint}-{num_samples}_samples.csv")

    dataset_list = ['food101', 'cifar10', 'cifar100', 'cars', 'fgvc_aircraft', 'dtd', 'pets', 'caltech101',
                    'flowers', 'country211']

    accs = []
    for dataset_name in dataset_list:
        accs.append(zero_shot_classifications(args, model, model_name, pretrained_checkpoint, preprocess_val, tokenizer,
                                              num_samples, dataset_name, output_file))
    av_Acc = sum(accs) / len(accs)
    print(av_Acc)
    all_df = pd.read_csv(output_file, index_col=0)
    acc = calc_avg_acc(all_df, output_file, model_name, pretrained_checkpoint)
    return acc
