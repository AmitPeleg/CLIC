import csv
import logging
import os
import pathlib
import random
import sys
from dataclasses import dataclass

import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from utils.data_utils import transform_to_list, get_ratio, get_hard_neg_single_sentence, get_p1_p2, random_concat, \
    get_hard_negative

csv.field_size_limit(sys.maxsize)

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class CsvDataset(Dataset):
    """
    Dataset class for loading image-caption pairs from a CSV file.
    The CSV should have image paths in the first column, captions in the second column, and additional positive captions can be included in subsequent columns.
    """

    def __init__(
            self, input_filename, transforms, tokenizer=None, root=None, lim=None, hard_negatives=False,
            additional_positives=0):
        """
        Args:
            input_filename (pathlib.Path): Path to the CSV file containing image paths and captions.
            transforms: Image transformation pipeline to apply to loaded images.
            tokenizer: Function to tokenize caption text.
            root (pathlib.Path, optional): Root directory to prepend to image paths.
            lim (int, optional): Limit on the number of samples to load.
            hard_negatives (bool, default=False): Whether to generate hard negative examples.
            additional_positives (int, default=0): Number of additional positive captions to load.
        """
        logging.debug(f"Loading csv data from {input_filename}.")

        self.input_filename: pathlib.Path = input_filename
        self.root = root
        self.hard_negatives = hard_negatives
        self.additional_positives = additional_positives
        self.transforms = transforms
        self.tokenizer = tokenizer
        self.clip_loss_iter = 1

        # Data storage
        self.image_paths = []
        self.captions = []
        self.captions2 = []  # For additional positive captions if requested

        self._load_data(lim)

        logging.debug(f"Done loading data. Total samples: {len(self.captions)}")

    def _load_data(self, lim):
        data_counter = 0

        assert self.input_filename.suffix == ".csv"
        with open(self.input_filename) as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in tqdm(csv_reader):
                # Check if we've reached the sample limit
                if lim is not None and data_counter >= lim:
                    break

                image_path = row[0]
                if image_path.endswith((".png", ".jpg", ".jpeg")):
                    # Store image path with root directory if specified
                    full_path = os.path.join(self.root, image_path) if self.root is not None else image_path
                    self.image_paths.append(full_path)

                    self.captions.append(row[1].lower())

                    if self.additional_positives:
                        # store with lower case
                        additional_positives = [cap.lower() for cap in row[2:]]
                        self.captions2.append(additional_positives)

                    data_counter += 1

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.image_paths[idx])).convert('RGBA'))
        caption = self.captions[idx]
        texts = self.tokenizer(caption)
        if self.hard_negatives:
            hard_negative = get_hard_neg_single_sentence(caption)
            hard_negative = self.tokenizer(hard_negative)
            return images, texts, hard_negative
        else:
            return images, texts


class CombineCsvDataset(CsvDataset):
    def __init__(self, input_filename, preprocess_fn=None, root=None, tokenizer=None, hard_negatives=False,
                 shuffled_positive=False, lim=None, additional_positives=0, clip_loss_iterate=False, baseline=False):
        super().__init__(input_filename, transforms=preprocess_fn, root=root, tokenizer=tokenizer,
                         hard_negatives=hard_negatives, lim=lim,
                         additional_positives=additional_positives)

        self.target_big = 224
        self.target_small = 112

        self.post_transform = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

        self.init_transform_higher = transforms.Compose([
            transforms.Resize(size=(self.target_big, self.target_small), interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop((self.target_big, self.target_small)),
            transforms.ToTensor(),
        ])

        self.init_transform_wider = transforms.Compose([
            transforms.Resize(size=(self.target_small, self.target_big), interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop((self.target_small, self.target_big)),
            transforms.ToTensor(),
        ])

        self.shuffled_positive = shuffled_positive
        self.additional_positives = additional_positives
        self.baseline = baseline

        self.clip_loss_iterate = clip_loss_iterate

    def __getitem__(self, idx):
        """
        Process a pair of images and their captions based on the paper, transforming them into training inputs.

        Args:
            idx: Index of the first image-caption pair

        Returns:
            Processed data ready for model training based on configuration settings.
        """

        # Get the first image and caption by the index
        img1, caption1, ratio = self.get_image_caption(idx=idx)

        # Get the second image and caption with the same aspect ratio, at random
        img2, caption2, ratio = self.get_image_caption(idx=None, ratio=ratio)

        # Concatenate the two images based on the aspect ratio
        concatenated_img = self.transform_and_random_concat(img1, img2, ratio)

        if self.additional_positives:
            # Split captions into first sentences and additional positive examples
            first_sentence_caption1, additional_positives_caption1 = caption1[0], caption1[1:]
            first_sentence_caption2, additional_positives_caption2 = caption2[0], caption2[1:]

            # Combine additional positive captions from both images
            additional_pos = []
            for add_pos_cap1, add_pos_cap2 in zip(additional_positives_caption1, additional_positives_caption2):
                combined_caption = random_concat(add_pos_cap1, add_pos_cap2)
                additional_pos.append(self.tokenizer(combined_caption))
        else:
            # If no additional positives, just use the first (and only) sentence
            first_sentence_caption1 = caption1
            first_sentence_caption2 = caption2

        p1, p2 = get_p1_p2(first_sentence_caption1, first_sentence_caption2)
        neg = get_hard_negative(first_sentence_caption1, first_sentence_caption2)

        # Tokenize the captions
        p1 = self.tokenizer(p1)
        p2 = self.tokenizer(p2)
        neg = self.tokenizer(neg)

        if self.clip_loss_iterate or self.baseline:
            # Handle baseline for the ablation study
            if self.baseline:
                neg1 = get_hard_neg_single_sentence(first_sentence_caption1)
                neg1 = self.tokenizer(neg1)
                additional_positives_caption1 = [self.tokenizer(pos) for pos in additional_positives_caption1]

            # Handle CLIP loss iteration or baselines processing
            single_text = self.tokenizer(first_sentence_caption1)
            single_image = concatenated_img[0]
            concatenated_img = concatenated_img[1]

        # Case 1: Baselines for ablation study
        if self.baseline:
            if self.additional_positives and self.hard_negatives and self.shuffled_positive:
                if self.clip_loss_iterate:
                    return single_image, single_text, single_text, *additional_positives_caption1, neg1, single_image, single_text
                else:
                    return single_image, single_text, single_text, *additional_positives_caption1, neg1
            else:
                raise NotImplementedError("Baseline requires hard negatives, shuffled positive, additional positives")

        # Case 2: Standard case from the paper (p1, p2, additional positives, hard negative)
        if self.additional_positives:
            if not self.hard_negatives:
                raise NotImplementedError("Additional positives require hard negatives")
            if self.shuffled_positive:
                if self.clip_loss_iterate:
                    return concatenated_img, p1, p2, *additional_pos, neg, single_image, single_text
                return concatenated_img, p1, p2, *additional_pos, neg
            else:
                if self.clip_loss_iterate:
                    return concatenated_img, p1, *additional_pos, neg, single_image, single_text
                return concatenated_img, p1, *additional_pos, neg

        # Case 3: Ablation with only p1, p2 and hard negative
        if self.shuffled_positive:
            if not self.hard_negatives:
                raise NotImplementedError("Shuffled positives require hard negatives")
            if self.clip_loss_iterate:
                return concatenated_img, p1, p2, neg, single_image, single_text
            return concatenated_img, p1, p2, neg

        # Case 4: Ablation with only p1, and hard negative
        if self.hard_negatives:
            if self.clip_loss_iterate:
                return concatenated_img, p1, neg, single_image, single_text
            return concatenated_img, p1, neg

        # training without clip loss iteration is done only for ablations
        if self.clip_loss_iterate:
            raise NotImplementedError("CLIP loss iteration not implemented without additional positives")

        # Case 5: Ablation with only p1
        return concatenated_img, p1

    def get_random_image_with_specific_ratio(self, first_img_ratio):
        while True:
            idx = random.randint(0, len(self.image_paths) - 1)
            image_obj = Image.open(str(self.image_paths[idx]))
            ratio = get_ratio(image_obj)
            if first_img_ratio == "both":
                # if the ratio of the original image is "both" all image are valid
                image = image_obj.convert('RGBA')
                caption = self.get_captions(idx)
                return image, caption, ratio
            elif ratio == "both":
                # if the ratio of the current image is "both" it is valid
                image = image_obj.convert('RGBA')
                caption = self.get_captions(idx)
                return image, caption, first_img_ratio
            elif first_img_ratio == ratio:
                # if the ratio of the original image is equal to the current image, it is valid
                image = image_obj.convert('RGBA')
                caption = self.get_captions(idx)
                return image, caption, ratio

    def get_image_caption(self, idx=None, ratio=None):
        # select row, by random if there is no idx, else using an index
        if idx is not None:
            image = Image.open(str(self.image_paths[idx])).convert('RGBA')
            caption = self.get_captions(idx)
            ratio = get_ratio(image)
            return image, caption, ratio
        else:
            return self.get_random_image_with_specific_ratio(ratio)

    def get_captions(self, idx):
        if not self.additional_positives:
            return self.captions[idx]

        # transform the string into a list
        caption2 = transform_to_list(self.captions2[idx])

        num_sentences = len(caption2)
        # takes k captions from the list at random, if not repeat to arrive at k
        if num_sentences >= self.additional_positives:
            caption2 = random.sample(caption2, self.additional_positives)  # No repetition
        else:
            caption2 = caption2 + random.choices(caption2,
                                                 k=self.additional_positives - num_sentences)  # Take all + sample remaining with replacement

        caption = [self.captions[idx], *caption2]
        return caption

    def transform_and_random_concat(self, img1, img2, ratio):
        if self.clip_loss_iterate or self.baseline:
            # duplicate img1
            img1_copy = img1.copy()
        if ratio == "both":
            # decide at random if it is higher or wider
            ratio = random.choice(["higher", "wider"])
        if ratio == "higher":
            # take the last 3 channels
            img1 = self.init_transform_higher(img1)[:3]  # RGBA
            img2 = self.init_transform_higher(img2)[:3]  # RGBA
            # Concatenate images one on top of the other, with the first image on top w.p 0.5
            if random.randint(0, 1):
                concatenated_img = torch.cat([img1, img2], dim=2)
            else:
                concatenated_img = torch.cat([img2, img1], dim=2)
        else:
            img1 = self.init_transform_wider(img1)[:3]  # RGBA
            img2 = self.init_transform_wider(img2)[:3]  # RGBA
            # Concatenate images side by side
            if random.randint(0, 1):
                concatenated_img = torch.cat([img1, img2], dim=1)
            else:
                concatenated_img = torch.cat([img2, img1], dim=1)

        if self.clip_loss_iterate or self.baseline:
            return self.transforms(img1_copy), self.post_transform(concatenated_img)
        else:
            return self.post_transform(concatenated_img)


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None

    def set_epoch(self, epoch):
        if self.sampler is not None and isinstance(
                self.sampler, DistributedSampler
        ):
            self.sampler.set_epoch(epoch)


def get_csv_dataset(
        args, preprocess_fn, is_train, tokenizer=None
):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    if not args.no_concat:
        dataset = CombineCsvDataset(
            input_filename,
            preprocess_fn,
            root=args.root,
            tokenizer=tokenizer,
            hard_negatives=args.hard_negatives,
            shuffled_positive=args.shuffled_positive,
            lim=args.limit_dataset,
            additional_positives=args.additional_positives,
            clip_loss_iterate=args.clip_loss_iterate,
            baseline=args.baseline,
        )
    else:
        dataset = CsvDataset(
            input_filename,
            preprocess_fn,
            root=args.root,
            tokenizer=tokenizer,
            lim=args.limit_dataset,
            hard_negatives=args.hard_negatives,
        )

    num_samples = len(dataset)
    sampler = (
        DistributedSampler(dataset)
        if args.distributed and is_train
        else None
    )
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_data(args, preprocess_train, tokenizer=None):
    data = {
        "train": get_csv_dataset(
            args, preprocess_train, is_train=True, tokenizer=tokenizer
        )
    }

    return data
