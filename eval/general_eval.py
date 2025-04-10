import argparse  # noqa: I001
import json
import os.path
import pathlib
from pathlib import Path

# sys.path.append("..")
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data
import torchmetrics  # noqa: F401
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import local_setting
from eval.clip_wrapper import CLIPWrapper
from eval.eval_utils import load_encode_text, load_encode_img, ArgumentsDefault
from eval.eval_zeroshot_imagenet import zero_shot_imagenet, zero_shot_all_classifications
from eval.retrieval_datasets import (COCO_Retrieval, Flickr30k_Retrieval, )
from local_setting import ROOT_DIR

assert pathlib.Path(".").resolve() == ROOT_DIR, "Please run the script from the root directory of the project."


@torch.no_grad()
def text_retrieval(pos_text, neg_text, image_path, model, tokenizer, transform, device):
    image_embedding = load_encode_img(device, image_path, model, transform)
    pos_text_embedding = load_encode_text(device, model, pos_text, tokenizer)
    neg_text_embedding = load_encode_text(device, model, neg_text, tokenizer)

    pos_before_pool = (pos_text_embedding * image_embedding).sum(dim=-1)
    neg_before_pool = (neg_text_embedding * image_embedding).sum(dim=-1)

    pos_score = pos_before_pool
    neg_score = neg_before_pool

    return 1 if pos_score.item() > neg_score.item() else 0


@torch.no_grad()
def sugarcrepe_evaluate(image_root, dataset, model, tokenizer, transform, device, limit, show_progress):
    metrics = {}
    image_root += "/val2017"
    for c, data_dict in dataset.items():
        correct_cnt = 0
        for i, data in tqdm(data_dict.items(), desc=f'evaluating {c}', disable=not show_progress):
            if limit is not None and int(i) >= limit:
                break
            image_path = os.path.join(image_root, data['filename'])
            correct = text_retrieval(data['caption'], data['negative_caption'], image_path, model, tokenizer, transform,
                                     device)
            correct_cnt += correct
        count = len(data_dict) if limit is None else min(len(data_dict), limit)
        metrics[c] = correct_cnt / count
    return metrics


@torch.no_grad()
def sugarcrepe_pp_evaluate(image_root, dataset, model, tokenizer, transform, device, limit, show_progress):
    metrics = {}
    image_root += "/val2017"
    for c, data_dict in dataset.items():
        total = 0
        correct_img_p1 = 0
        correct_img_p2 = 0

        correct_full = 0  ###  the main task: P1 and P2 closer to Image than Negative
        correct_text = 0
        # print(data_dict[0])
        for i, data in enumerate(tqdm(data_dict, desc=f'evaluating {c}', disable=not show_progress)):
            if limit is not None and int(i) >= limit:
                break
            p1 = data['caption']
            neg = data['negative_caption']
            p2 = data['caption2']

            model.eval()
            image_path = os.path.join(image_root, data['filename'])
            img_feats = load_encode_img(device, image_path, model, transform)
            p1_feats = load_encode_text(device, model, p1, tokenizer)
            p2_feats = load_encode_text(device, model, p2, tokenizer)
            neg_feats = load_encode_text(device, model, neg, tokenizer)

            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            cos_p1 = cos(img_feats, p1_feats)
            cos_p2 = cos(img_feats, p2_feats)
            cos_neg = cos(img_feats, neg_feats)
            cos_p1p2 = cos(p1_feats, p2_feats)
            cos_p1_neg = cos(p1_feats, neg_feats)
            cos_p2_neg = cos(p2_feats, neg_feats)

            #############  Compute the performance of the models on each subset  ###

            total += 1

            if cos_p1 > cos_neg and cos_p2 > cos_neg:
                correct_full += 1
            if cos_p1 > cos_neg:
                correct_img_p1 += 1
            if cos_p2 > cos_neg:
                correct_img_p2 += 1
            if cos_p1p2 > cos_p1_neg and cos_p1p2 > cos_p2_neg:
                correct_text += 1

        ave_score = float(correct_full) / float(total)
        ave_score_txt = float(correct_text) / float(total)

        metrics[c] = [ave_score, ave_score_txt]  # correct_cnt / count
    return metrics


def zero_shot_sugarcrepe_pp(args, model, preprocess_val, tokenizer, device, pretrained_checkpoint='', save_scores=True,
                            sugarcrepe_limit=None):
    img_path = args.coco2017_image_root
    data_dict = get_data_dict_sugarcrepe_pp()
    dataset = {}
    for c, data_path in data_dict.items():
        dataset[c] = json.load(open(data_path, encoding='utf-8'))
    metrics = sugarcrepe_pp_evaluate(img_path, dataset, model, tokenizer, preprocess_val, device,
                                     limit=sugarcrepe_limit, show_progress=save_scores)
    metric = process_sugarcrepe_pp(metrics, True)
    if save_scores:
        output_dir = Path(args.output)
        output_dir = Path(output_dir, "sugarcrepe_pp")
        output_dir.mkdir(exist_ok=True, parents=True)

        print(f"Dump results to: {os.path.join(output_dir, f'{pretrained_checkpoint}.json')}")
        json.dump(metrics, open(os.path.join(output_dir, f'{pretrained_checkpoint}.json'), 'w'), indent=4)
    return metrics


def zero_shot_sugarcrepe(args, model, preprocess_val, tokenizer, device, pretrained_checkpoint='', save_scores=True,
                         sugarcrepe_limit=None):
    data_dict = get_data_dict_sugarcrepe()

    dataset = {}
    for c, data_path in data_dict.items():
        dataset[c] = json.load(open(data_path, encoding='utf-8'))

    print("=> Begin SugarCrepe Evaluation")
    metrics = sugarcrepe_evaluate(args.coco2017_image_root, dataset, model, tokenizer, preprocess_val, device,
                                  limit=sugarcrepe_limit, show_progress=save_scores)

    # print metrics
    metrics = process_sugarcrepe(metrics, save_scores)
    if save_scores:
        output_dir = Path(args.output)
        output_dir = Path(output_dir, "sugarcrepe")
        output_dir.mkdir(exist_ok=True, parents=True)

        print(f"Dump results to: {os.path.join(output_dir, f'{pretrained_checkpoint}.json')}")
        json.dump(metrics, open(os.path.join(output_dir, f'{pretrained_checkpoint}.json'), 'w'), indent=4)
    return metrics


def get_data_dict_sugarcrepe_pp():
    data_dir = local_setting.SUGARCREPE_PP_DATA_DIR
    data_dict = {
        'replace_obj': f'{data_dir}/replace_obj.json',
        'replace_att': f'{data_dir}/replace_att.json',
        'replace_rel': f'{data_dir}/replace_rel.json',
        'swap_obj': f'{data_dir}/swap_obj.json',
        'swap_att': f'{data_dir}/swap_att.json',
    }

    return data_dict


def get_data_dict_sugarcrepe():
    data_dir = local_setting.SUGARCREPE_DATA_DIR
    data_dict = {
        'add_obj': f'{data_dir}/add_obj.json',
        'add_att': f'{data_dir}/add_att.json',
        'replace_obj': f'{data_dir}/replace_obj.json',
        'replace_att': f'{data_dir}/replace_att.json',
        'replace_rel': f'{data_dir}/replace_rel.json',
        'swap_obj': f'{data_dir}/swap_obj.json',
        'swap_att': f'{data_dir}/swap_att.json',
    }

    return data_dict


def process_sugarcrepe_pp(metrics, verbose):
    # calculate the average accuracy among add, replace, swap

    # print all the attributes, while keeping 3 decimal places
    if verbose:
        for c in metrics:
            print(f"{c}: {metrics[c]}")
    else:
        print(
            f"replace_obj: {metrics['replace_obj']} replace_att: {metrics['replace_att']} replace_rel: {metrics['replace_rel']} "
            f"swap_obj: {metrics['swap_obj']} swap_att: {metrics['swap_att']} ")

    return metrics


def process_sugarcrepe(metrics, verbose):
    # calculate the average accuracy among add, replace, swap
    metrics["avg_add"] = (metrics['add_obj'] + metrics['add_att']) / 2
    metrics["avg_replace"] = (metrics['replace_obj'] + metrics['replace_att'] + metrics['replace_rel']) / 3
    metrics["avg_swap"] = (metrics['swap_obj'] + metrics['swap_att']) / 2
    # print all the attributes, while keeping 3 decimal places
    if verbose:
        for c in metrics:
            print(f"{c}: {metrics[c]:.3f}")
    else:
        print(f"add_obj: {metrics['add_obj']:.3f} add_att: {metrics['add_att']:.3f} "
              f"replace_obj: {metrics['replace_obj']:.3f} replace_att: {metrics['replace_att']:.3f} replace_rel: {metrics['replace_rel']:.3f} "
              f"swap_obj: {metrics['swap_obj']:.3f} swap_att: {metrics['swap_att']:.3f} ")
        print(
            f"avg_add: {metrics['avg_add']:.3f} avg_replace: {metrics['avg_replace']:.3f} avg_swap: {metrics['avg_swap']:.3f}")

    return metrics


def eval_winoground(args, model, model_name, pretrained_checkpoint, preprocess_val, tokenizer, device,
                    save_scores=True):
    def text_correct(result):
        return result["c0_i0"] > result["c1_i0"] and result["c1_i1"] > result["c0_i1"]

    def image_correct(result):
        return result["c0_i0"] > result["c0_i1"] and result["c1_i1"] > result["c1_i0"]

    def group_correct(result):
        return image_correct(result) and text_correct(result)

    from datasets import load_dataset
    dataset = load_dataset('facebook/winoground')['test']
    winoground_clip_scores = []
    for example in tqdm(dataset):
        # Note that some images in winoground are RGBA and some are RGB. Need to convert all to RGB with .convert('RGB')
        # Note that we could run this example through CLIP as a batch, but I want to drive the point home that we get four independent image-caption scores for each example
        img_0 = preprocess_val(example["image_0"].convert("RGB")).to(device)
        img_1 = preprocess_val(example["image_1"].convert("RGB")).to(device)
        text_0 = tokenizer(example["caption_0"]).to(device)
        text_1 = tokenizer(example["caption_1"]).to(device)
        # print(img_0.size(), img_1.size())
        img_embed_0 = F.normalize(model.encode_image(img_0.unsqueeze(0)), dim=-1)
        img_embed_1 = F.normalize(model.encode_image(img_1.unsqueeze(0)), dim=-1)
        text_embed_0 = F.normalize(model.encode_text(text_0.unsqueeze(0)), dim=-1)
        text_embed_1 = F.normalize(model.encode_text(text_1.unsqueeze(0)), dim=-1)
        clip_score_c0_i0 = torch.sum(img_embed_0 * text_embed_0, dim=-1).item()
        clip_score_c1_i0 = torch.sum(img_embed_0 * text_embed_1, dim=-1).item()
        clip_score_c0_i1 = torch.sum(img_embed_1 * text_embed_0, dim=-1).item()
        clip_score_c1_i1 = torch.sum(img_embed_1 * text_embed_1, dim=-1).item()
        winoground_clip_scores.append(
            {"id": example["id"], "c0_i0": clip_score_c0_i0, "c0_i1": clip_score_c0_i1, "c1_i0": clip_score_c1_i0,
             "c1_i1": clip_score_c1_i1})

    text_correct_count = 0
    image_correct_count = 0
    group_correct_count = 0
    for result in winoground_clip_scores:
        text_correct_count += 1 if text_correct(result) else 0
        image_correct_count += 1 if image_correct(result) else 0
        group_correct_count += 1 if group_correct(result) else 0

    denominator = len(winoground_clip_scores)

    metrics = {"text score": text_correct_count / denominator,
               "image score": image_correct_count / denominator,
               "group score": group_correct_count / denominator
               }
    print(metrics)
    if save_scores:
        output_dir = Path(args.output)
        output_dir = Path(output_dir, "winoground")
        output_dir.mkdir(exist_ok=True, parents=True)

        print(f"Dump results to: {os.path.join(output_dir, f'{pretrained_checkpoint}.json')}")

        json.dump(metrics, open(os.path.join(output_dir, f'{pretrained_checkpoint}.json'), 'w'), indent=4)
    return metrics


def zero_shot_retrival(args, model, model_name, pretrained_checkpoint, preprocess_val, tokenizer, device, dataset_name):
    model = CLIPWrapper(model, device, tokenizer)

    if dataset_name == "coco2017_retrival":
        data_dir = Path(args.coco2017_image_root)
        # data_dir.mkdir(parents=True, exist_ok=True)
        max_words = 30
        split = "test"
        dataset = COCO_Retrieval(root_dir=data_dir, split=split, image_preprocess=preprocess_val, image_perturb_fn=None,
                                 max_words=max_words,
                                 download=False)
    elif dataset_name == "flickr30k_retrival":
        data_dir = Path(args.flickr30k_image_root)
        max_words = 30
        dataset = Flickr30k_Retrieval(root_dir=data_dir, split="test", image_preprocess=preprocess_val,
                                      image_perturb_fn=None, max_words=max_words,
                                      download=False)
    else:
        raise ValueError("Invalid dataset name")

    print(f"Evaluating {model_name}-{pretrained_checkpoint} on {dataset_name}")

    collate_fn = None

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        collate_fn=collate_fn)

    scores = model.get_retrieval_scores_dataset(args, loader)
    result_records = dataset.evaluate_scores(scores)

    print(result_records)
    if result_records:
        output_dir = Path(args.output)
        output_dir = Path(output_dir, dataset_name)
        output_dir.mkdir(exist_ok=True, parents=True)

        print(f"Dump results to: {os.path.join(output_dir, f'{pretrained_checkpoint}.json')}")
        json.dump(result_records, open(os.path.join(output_dir, f'{pretrained_checkpoint}.json'), 'w'), indent=4)

    return result_records


def main(args, modelPath=None):
    model_name = args.model

    if modelPath:
        args.output = str(Path(modelPath))
        pretrained_checkpoint = str(Path(modelPath) / "checkpoint.pt")
        args.load_pretrained_clip = pretrained_checkpoint
    else:
        pass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pretrained_checkpoint = f"{model_name}-{args.architecture}"

    if args.load_pretrained_clip:

        pretrained_checkpoint = f'{model_name}-{args.architecture}-{args.load_pretrained_clip.split("/")[-1]}'
        if 'pt' not in pretrained_checkpoint:
            args.load_pretrained_clip += "/checkpoint.pt"

    from models import get_model_clip
    model, preprocess_train, preprocess_val, tokenizer = get_model_clip(args)

    model.eval()
    model.to(device)
    cudnn.benchmark = True

    print(f"Evaluating {pretrained_checkpoint}")

    if args.evaluation_metric == "imagenet":
        acc = zero_shot_imagenet(args, model, model_name, pretrained_checkpoint, preprocess_val, tokenizer)
        print(acc)
        return acc
    elif args.evaluation_metric == "zero_shot_classification":
        acc = zero_shot_all_classifications(args, model, model_name, pretrained_checkpoint, preprocess_val, tokenizer,
                                            num_samples=1000)
        print(acc)
        return acc
    elif args.evaluation_metric == "sugarcrepe":
        met = zero_shot_sugarcrepe(args, model, preprocess_val, tokenizer, device, pretrained_checkpoint)
        print(met)
        return met
    elif args.evaluation_metric == "sugarcrepe_pp":
        met = zero_shot_sugarcrepe_pp(args, model, preprocess_val, tokenizer, device, pretrained_checkpoint)
        print(met)
        return met
    elif args.evaluation_metric == 'winoground':
        met = eval_winoground(args, model, model_name, pretrained_checkpoint, preprocess_val, tokenizer, device,
                              save_scores=True)
        print(met)
        return met

    elif args.evaluation_metric == "coco2017_retrival":  # taken from aro
        ret = zero_shot_retrival(args, model, model_name, pretrained_checkpoint, preprocess_val, tokenizer, device,
                                 "coco2017_retrival")
        return ret
    elif args.evaluation_metric == "flickr30k_retrival":  # taken from aro
        ret = zero_shot_retrival(args, model, model_name, pretrained_checkpoint, preprocess_val, tokenizer, device,
                                 "flickr30k_retrival")
        return ret
    else:
        raise ValueError("Invalid evaluation metric")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="clip evaluations", add_help=False
    )

    parser.add_argument(
        "--seed",
        default=1,
        type=int,
        help="seed for aro evaluation",
    )

    parser.add_argument(
        "--num_workers",
        default=1,
        type=int,
        help="number of workers for dataloader",
    )

    parser.add_argument(
        "--evaluation_metric",
        default="coco2014_retrival",
        type=str,
        help="metric to evaluate on",
    )

    parser.add_argument(
        "--imagenet-root",
        default=str(local_setting.IMAGENET_DIR),
        type=str,
        help="path to imagenet dataset",
    )
    parser.add_argument(
        "--coco2014_image_root",
        default=str(local_setting.COCO2014_DIR),
        type=str,
        help="path to coco test2014 dataset",
    )

    parser.add_argument(
        "--coco2017_image_root",
        default=str(local_setting.COCO2017_DIR),
        type=str,
        help="path to coco val2017 dataset",
    )

    parser.add_argument(
        "--flickr30k_image_root",
        default=str(local_setting.FLICKR30K_DIR),
        type=str,
        help="path to flickr30k dataset",
    )

    parser.add_argument(
        "--output",
        default=str(local_setting.EVAL_DIR),
        type=str,
        help="path to evaluation directory",
    )

    parser.add_argument(
        "--saved_models",
        default=str(local_setting.SAVED_MODELS_DIR),
        type=str,
        help="path to saved models directory",
    )

    parser.add_argument(
        "--data_root",
        default=str(local_setting.DATA_DIR),
        type=str,
        help="path to datasetS",
    )

    parser.add_argument(
        "--dataset_root",
        default="/mnt/datasets/",
        type=str,
        help="zero shot classification",
    )
    parser.add_argument(
        "--filter_image_idx",
        default=False,
        type=bool,
        help="filter corrupted images in problematic datasets",
    )

    parser.add_argument(
        "--load_pretrained_clip",
        default=None,
        type=str,
        help="model to test",
    )

    parser.add_argument(
        "--batch-size", default=128, type=int, help="batch_size"
    )

    parser.add_argument(
        "--sugarcrepe_limit", default=None, type=int, help="define the number of samples to evaluate on"
    )

    parser.add_argument(
        "--freeze_only_text", action="store_true", help="Use quickgelu"
    )
    parser.add_argument(
        "--freeze_only_vision", action="store_true", help="Use quickgelu"
    )
    parser.add_argument(
        "--model",
        default="ViT-B-16",
        type=str,
        help="model architecture",
    )
    parser.add_argument(
        "--architecture",
        default="ViT-L-14",
        type=str,
        help="model architecture",
    )
    parser.add_argument(
        "--quickgelu", action="store_true", help="Use quickgelu"
    )
    parser.add_argument("-j", "--workers", default=10, type=int)
    args = parser.parse_args()
    main(args)


def from_train(modelPath=None, eval_metric=None, args_run=None):
    args1 = ArgumentsDefault()
    args1.model = args_run.model
    args1.freeze_only_vision = False
    args1.evaluation_metric = eval_metric
    args1.load_pretrained_clip = None
    args1.architecture = args_run.architecture
    args1.freeze_only_text = False
    met = main(args1, modelPath)
    return met
