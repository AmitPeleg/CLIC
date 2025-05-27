import numpy as np
import pandas as pd
from PIL import Image
from torch.nn import functional as F, Parameter

import local_setting


def shuffle_and_select(dataset_length, n=1000):
    # Create indices for the dataset
    indices = list(range(dataset_length))
    # Shuffle the indices
    np.random.shuffle(indices)
    # Select the first n indices
    selected_indices = indices[:n]
    return selected_indices


def calc_avg_acc(all_df, output_file, model_name, pretrained_checkpoint, averaged_metric='acc',
                 key_to_replace='dataset'):
    # Calculate the average of the 'acc' column
    average_acc = all_df[averaged_metric].mean()

    # Select the row based on the provided index
    row = all_df.iloc[0]
    # Convert the row to a dictionary
    new_row = row.to_dict()
    # Update the 'preposition' key with the new value
    new_row[key_to_replace] = f'avg_across_{key_to_replace}'
    new_row[averaged_metric] = average_acc

    # Create a new DataFrame with the average value
    new_row = pd.DataFrame(new_row, index=[0])
    # Append the new row to the original DataFrame
    all_df = pd.concat([all_df, new_row])
    # Save the updated DataFrame
    all_df.to_csv(output_file)


def load_encode_text(device, model, text, tokenizer, unusual_model=False):
    text = tokenizer(text).to(device)
    # if text shape is (num_tokens,) then add a batch dimension:
    if len(text.shape) == 1:
        text = text.unsqueeze(0)
    text_embedding = model.encode_text(text)

    text_embedding = F.normalize(text_embedding, dim=-1)

    return text_embedding


def load_encode_img(device, image_path, model, transform):
    image = Image.open(image_path)
    if image.layers != 3:
        # print("Image has 1 channel, converting to RGB")
        image = image.convert('RGB')
    image = transform(image).to(device)
    if len(image.shape) == 3:
        image = image.unsqueeze(dim=0)
    image_embedding = model.encode_image(image)
    image_embedding = F.normalize(image_embedding, dim=-1)
    return image_embedding


def load_my_state_dict(model, state_dict):
    own_state = model.state_dict()
    for name, param in state_dict.items():
        name = name.replace("module.", "")
        if name not in own_state:
            continue
        if isinstance(param, Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        # print(name)
        own_state[name].copy_(param)
    return own_state


class ArgumentsDefault:
    def __init__(self):
        self.seed = 1
        self.num_workers = 8
        self.evaluation_metric = 'coco2017_retrival'
        self.imagenet_root = str(local_setting.IMAGENET_DIR)
        self.flickr30k_image_root = str(local_setting.FLICKR30K_DIR)
        self.coco2017_image_root = str(local_setting.COCO2017_DIR)
        self.coco2017_annotation_root = str(local_setting.COCO2017_ANNOTATIONS_DIR)
        self.output = str(local_setting.EVAL_DIR)
        self.data_root = str(local_setting.DATA_DIR)
        self.filter_image_idx = False
        self.batch_size = 200
        self.model = "ViT-B-32"
        self.workers = 8
        self.quickgelu = False
        self.sugarcrepe_limit = None
