import json
import os
import warnings

from torchvision.datasets import (
    CIFAR10, CIFAR100, Food101, SUN397,
    StanfordCars, FGVCAircraft, DTD, OxfordIIITPet, Flowers102,
    STL10, EuroSAT, GTSRB, Country211, PCAM
)

from eval.datasets import caltech101


def build_dataset(dataset_name, root="root", transform=None, split="test", download=False, annotation_file=None,
                  language="en", task="zeroshot_classification", wds_cache_dir=None, custom_classname_file=None,
                  custom_template_file=None, num_samples=-1, **kwargs):
    """
    Main function to use in order to build a dataset instance,

    dataset_name: str
        name of the dataset

    root: str
        root folder where the dataset is downloaded and stored. can be shared among datasets.

    transform: torchvision transform applied to images

    split: str
        split to use, depending on the dataset can have different options.
        In general, `train` and `test` are available.
        For specific splits, please look at the corresponding dataset.

    annotation_file: str or None
        only for datasets with captions (used for retrieval) such as COCO
        and Flickr.

    custom_classname_file: str or None
        Custom classname file where keys are dataset names and values are list of classnames.

    custom_template_file: str or None
        Custom template file where keys are dataset names and values are list of prompts, or dicts
        where keys are classnames and values are class-specific prompts.

    """
    assert task == "zeroshot_classification", f"Task '{task}' not supported"
    current_folder = os.path.dirname(__file__)

    if custom_classname_file and not os.path.exists(custom_classname_file):
        # look at current_folder
        custom_classname_file_attempt = os.path.join(current_folder, custom_classname_file)
        assert os.path.exists(
            custom_classname_file_attempt), f"Custom classname file '{custom_classname_file}' does not exist"
        custom_classname_file = custom_classname_file_attempt
    else:
        custom_classname_file = os.path.join(current_folder, 'datasets', language + "_classnames.json")

    if custom_template_file and not os.path.exists(custom_template_file):
        # look at current_folder
        custom_template_file_attempt = os.path.join(current_folder, custom_template_file)
        assert os.path.exists(
            custom_template_file_attempt), f"Custom template file '{custom_template_file}' does not exist"
        custom_template_file = custom_template_file_attempt
    else:
        custom_template_file = os.path.join(current_folder, 'datasets',
                                            language + "_zeroshot_classification_templates.json")

    with open(custom_classname_file, "r") as f:
        classnames = json.load(f)

    with open(custom_template_file, "r") as f:
        templates = json.load(f)

    default_template = templates["imagenet1k"] if "imagenet1k" in templates else None

    name = dataset_name
    templates = templates.get(name, default_template)
    assert templates is not None, f"Templates for dataset '{dataset_name}' not found in '{custom_template_file}'"

    train = (split == "train")
    if dataset_name == "cifar10":  # 1
        ds = CIFAR10(root=root, train=train, transform=transform, download=download, **kwargs)
    elif dataset_name == "cifar100":  # 2
        ds = CIFAR100(root=root, train=train, transform=transform, download=download, **kwargs)
    elif dataset_name == "cars":
        ds = StanfordCars(root=root, split="train" if train else "test", transform=transform, download=download,
                          **kwargs)
    elif dataset_name == "fgvc_aircraft":
        ds = FGVCAircraft(root=root, annotation_level="variant", split="train" if train else "test",
                          transform=transform, download=download, **kwargs)
    elif dataset_name == "dtd":
        ds = DTD(root=root, split="train" if train else "test", transform=transform, download=download, **kwargs)
    elif dataset_name == "pets":
        ds = OxfordIIITPet(root=root, split="train" if train else "test", target_types="category", transform=transform,
                           download=download, **kwargs)
    elif dataset_name == "caltech101":
        warnings.warn(
            f"split argument ignored for `{dataset_name}`, there are no pre-defined train/test splits for this dataset")
        # broken download link (can't download google drive), fixed by this PR https://github.com/pytorch/vision/pull/5645
        # also available in "vtab/caltech101" using VTAB splits, we advice to use VTAB version rather than this one
        # since in this one (torchvision) there are no pre-defined test splits
        ds = caltech101.Caltech101(root=root, target_type="category", transform=transform, download=download, **kwargs)
        ds.classes = classnames["caltech101"]
    elif dataset_name == "flowers":
        ds = Flowers102(root=root, split="train" if train else "test", transform=transform, download=download, **kwargs)
        # class indices started by 1 until it was fixed in  a  PR (#TODO link of the PR)
        # if older torchvision version, fix it using a target transform that decrements label index
        # TODO figure out minimal torchvision version needed instead of decrementing
        if ds[0][1] == 1:
            ds.target_transform = lambda y: y - 1
        ds.classes = classnames["flowers"]
    elif dataset_name == "stl10":
        ds = STL10(root=root, split="train" if train else "test", transform=transform, download=download, **kwargs)
    elif dataset_name == "eurosat":
        warnings.warn(
            f"split argument ignored for `{dataset_name}`, there are no pre-defined train/test splits for this dataset")
        ds = EuroSAT(root=root, transform=transform, download=download, **kwargs)
        ds.classes = classnames["eurosat"]
    elif dataset_name == "pcam":
        # Dead link. Fixed by this PR on torchvision https://github.com/pytorch/vision/pull/5645
        # TODO figure out minimal torchvision version needed
        ds = PCAM(root=root, split="train" if train else "test", transform=transform, download=download, **kwargs)
        ds.classes = classnames["pcam"]
    elif dataset_name == "food101":
        assert split in ("train", "test"), f"Only `train` and `test` split available for {dataset_name}"
        ds = Food101(root=root, split=split, transform=transform, download=download, **kwargs)
        # we use the default class names, we just  replace "_" by spaces
        # to delimit words
        ds.classes = [cl.replace("_", " ") for cl in ds.classes]
    elif dataset_name == "sun397":
        warnings.warn(
            f"split argument ignored for `{dataset_name}`, there are no pre-defined train/test splits for this dataset")
        # we use the default class names, we just  replace "_" and "/" by spaces
        # to delimit words
        ds = SUN397(root=root, transform=transform, download=download, **kwargs)
        ds.classes = [cl.replace("_", " ").replace("/", " ") for cl in ds.classes]
    elif dataset_name == "gtsrb":
        assert split in ("train", "test"), f"Only `train` and `test` split available for {dataset_name}"
        ds = GTSRB(root=root, split=split, transform=transform, download=download, **kwargs)
        ds.classes = classnames["gtsrb"]
    elif dataset_name == "country211":
        assert split in (
            "train", "valid", "test"), f"Only `train` and `valid` and `test` split available for {dataset_name}"
        ds = Country211(root=root, split=split, transform=transform, download=download, **kwargs)
        ds.classes = classnames["country211"]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}.")

    ds.templates = templates
    return ds
