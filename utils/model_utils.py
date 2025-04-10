import torch
from torch.nn import Parameter


def normalize_and_append(class_embeddings, text_features):
    class_embeddings = (class_embeddings / class_embeddings.norm(dim=-1, keepdim=True))
    class_embeddings = class_embeddings.mean(dim=0)
    class_embeddings = (class_embeddings / class_embeddings.norm(dim=-1, keepdim=True))
    text_features.append(class_embeddings)


def get_model(model):
    if isinstance(model, torch.nn.DataParallel) or isinstance(
            model, torch.nn.parallel.DistributedDataParallel
    ):
        return model.module
    else:
        return model


def freeze_text_layers(model):
    for name, param in model.named_parameters():
        if 'visual' in name:
            print(name + ' is not frozen')
            param.requires_grad = True
        else:
            print(name + ' is frozen')
            param.requires_grad = False


def freeze_visual_layers(model):
    for name, param in model.named_parameters():
        if 'visual' in name:
            print(name + ' is frozen')
            param.requires_grad = False
        else:
            print(name + ' is not frozen')
            param.requires_grad = True


def load_my_state_dict(model, state_dict):
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            continue
        if isinstance(param, Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)
    return own_state
