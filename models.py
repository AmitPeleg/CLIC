from collections import OrderedDict

import clip
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MODELKEYS = {
    'ViT-B-16': 'openai',
    'ViT-B-32': 'openai',
    'ViT-L-14': 'openai',
    'DAC': 'DAC',
    'TSLVC': 'TSLVC',
    'TripletCLIP': 'TripletCLIP',
    'con-CLIP': 'con-CLIP',
    'CLIPS': 'CLIPS',
    'CLIPA': 'CLIPA',
    'EVA': 'EVA'
}


class CLIP(nn.Module):

    def __init__(self, modelname='ViT-L-14', fqg=True):
        super().__init__()
        if modelname == 'CLIPS':

            from clips import create_model_from_pretrained as load_clips
            self.model, _ = load_clips('hf-hub:UCSC-VLAA/ViT-L-14-CLIPS-224-Recap-DataComp-1B')

        elif modelname == 'CLIPA':
            from open_clip import create_model_from_pretrained

            self.model, _ = create_model_from_pretrained('hf-hub:UCSC-VLAA/ViT-L-14-CLIPA-datacomp1B')
        elif modelname == 'EVA':
            from open_clip import create_model_from_pretrained, get_tokenizer
            self.model, _, _ = open_clip.create_model_and_transforms(
                model_name="EVA02-L-14",
                pretrained="merged2b_s4b_b131k"
            )
        elif 'HF-' in modelname:
            rm_hf_name = '-'.join(modelname.split("-")[1:])
            self.model = open_clip.create_model_and_transforms(f'hf-hub:nmndeep/{rm_hf_name}')[0]

        else:
            self.model = open_clip.create_model_and_transforms(
                model_name=modelname,
                force_quick_gelu=fqg,
                pretrained=MODELKEYS[modelname]
            )[0]

        self.modelname = modelname
        self.context_length = self.model.context_length
        self.logit_scale = self.model.logit_scale.exp()

    def encode_text(self, input_ids):
        return self.model.encode_text(input_ids)

    def encode_image(self, input_pixels):
        return self.model.encode_image(input_pixels)

    def get_embedings(self, image_embed, text_embed):
        # normalized features
        image_embed = F.normalize(image_embed, dim=-1, p=2)  # (batch_size, num_img_tokens, embed_dim)
        text_embed = F.normalize(text_embed, dim=-1,
                                 p=2)  # (batch_size / batch_size * 2 for hard_negatives, num_text_tokens, embed_dim)
        # gather with gradient
        if self.args.distributed:
            image_embed_all = torch.cat(torch.distributed.nn.all_gather(image_embed),
                                        dim=0)  # (batch_size * num_gpus, num_img_tokens, embed_dim)
            text_embed_all = torch.cat(torch.distributed.nn.all_gather(text_embed),
                                       dim=0)  # (batch_size* num_gpus / batch_size * num_gpus * 2 for hard_negatives, num_text_tokens, embed_dim)
        else:
            image_embed_all = image_embed
            text_embed_all = text_embed
        return image_embed, image_embed_all, text_embed, text_embed_all

    def forward(
            self,
            input_pixels: torch.Tensor,
            input_ids: torch.Tensor,
    ):

        img_embeds = self.encode_image(input_pixels)
        text_embeds = self.encode_text(input_ids)

        image_embed, image_embed_all, text_embed, text_embed_all = self.get_embedings(img_embeds, text_embeds)
        cos_sim = F.cosine_similarity(img_embeds, text_embed[:len(image_embed)], dim=-1).mean()

        # (batch_size, 1, 1, 512) * (1, gpus*batch_size, num_tokens, 512)  -> (batch_size, gpus*batch_size, num_tokens)
        logit_scale = self.model.logit_scale.exp()

        text_all = logit_scale * (image_embed.unsqueeze(dim=1) * text_embed_all.unsqueeze(dim=0)).sum(dim=-1)
        image_all = logit_scale * (text_embed.unsqueeze(dim=1) * image_embed_all.unsqueeze(dim=0)).sum(dim=-1)
        logits_per_image = text_all
        logits_per_text = image_all

        return {"logits_per_image": logits_per_image, "logits_per_text": logits_per_text, 'text_only': text_embed,
                'image_only': image_embed, 'cos-sim': cos_sim.detach()}

    def __getattr__(self, item):
        # Function for bypassing the DistributedDataParallel module
        if item == 'module':
            return self
        return super().__getattr__(item)


def load_checkpoint(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path)
    model = model.float()
    model.load_state_dict(ckpt["model"], strict=True)
    return model


def get_model_clip(args):
    if 'con' in args.model or 'Triplet' in args.model:
        model, train_transform = clip.load(args.architecture.replace("-", "/", 1), jit=False)

        try:
            tokeniz = open_clip.get_tokenizer(args.architecture)
        except:
            tokeniz = open_clip.get_tokenizer("ViT-L-14")
        tokenizer = lambda x: tokeniz(x).squeeze()


    elif args.model == 'CLIPS':
        model = CLIP(modelname="CLIPS")
        from clips import create_model_from_pretrained as load_clips
        from clips import get_tokenizer
        _, train_transform = load_clips('hf-hub:UCSC-VLAA/ViT-L-14-CLIPS-224-Recap-DataComp-1B')
        tokeniz = get_tokenizer('hf-hub:UCSC-VLAA/ViT-L-14-CLIPS-224-Recap-DataComp-1B')
        tokenizer = lambda x: tokeniz(x).squeeze()

    elif args.model == 'CLIPA':
        model = CLIP(modelname="CLIPA")
        from open_clip import create_model_from_pretrained, get_tokenizer
        _, train_transform = create_model_from_pretrained('hf-hub:UCSC-VLAA/ViT-L-14-CLIPA-datacomp1B')
        tokeniz = get_tokenizer('hf-hub:UCSC-VLAA/ViT-L-14-CLIPA-datacomp1B')
        tokenizer = lambda x: tokeniz(x).squeeze()

    elif args.model == 'EVA':
        model = CLIP(modelname="EVA")
        from open_clip import create_model_from_pretrained, get_tokenizer
        _, _, train_transform = open_clip.create_model_and_transforms(
            model_name="EVA02-L-14",
            pretrained="merged2b_s4b_b131k"
        )
        tokeniz = get_tokenizer("EVA02-L-14")
        tokenizer = lambda x: tokeniz(x).squeeze()

    elif 'HF-' in args.model:
        print(f"Loading {args.model}")
        model = CLIP(modelname=args.model)

        rm_hf_name = '-'.join(args.model.split("-")[1:])
        train_transform = open_clip.create_model_and_transforms(f'hf-hub:nmndeep/{rm_hf_name}')[2]

        if 'CLIPS' not in args.model:
            tokeniz = open_clip.get_tokenizer(f'hf-hub:nmndeep/{rm_hf_name}')
        else:
            # since clips is notyet on openclip
            from clips import get_tokenizer
            tokeniz = get_tokenizer('hf-hub:UCSC-VLAA/ViT-L-14-CLIPS-224-Recap-DataComp-1B')
        tokenizer = lambda x: tokeniz(x).squeeze()

    else:
        model = CLIP(modelname=args.architecture)

        _, _, train_transform = open_clip.create_model_and_transforms(
            model_name=args.architecture,
            force_quick_gelu=True,
            pretrained=MODELKEYS[args.model]
        )
        try:
            tokeniz = open_clip.get_tokenizer(args.architecture)
        except:
            tokeniz = open_clip.get_tokenizer(args.architecture.replace("-", "/", 1))
        tokenizer = lambda x: tokeniz(x).squeeze()

    model.args = args

    print(args.model)
    if args.model == 'con-CLIP':
        # load func from original repo
        model = load_checkpoint(model, args.load_pretrained_clip)
    else:
        if args.load_pretrained_clip:
            print(args.load_pretrained_clip)

            ckpt = torch.load(args.load_pretrained_clip, map_location='cpu')  # ['model']

            state_dict = OrderedDict()
            try:
                for k, v in ckpt['state_dict'].items():
                    state_dict[k.replace('module.', '')] = v
            except:
                for k, v in ckpt.items():
                    state_dict[k.replace('module.', '')] = v
            state_dict = utils.model_utils.load_my_state_dict(model, state_dict)
            try:
                # model = load_checkpoint(model, args.load_pretrained_clip)
                model.load_state_dict(state_dict, strict=True)
                print("Loaded from ckpt")
            except:
                print("Failed a bit")

    if bool(args.freeze_only_vision):
        utils.model_utils.freeze_visual_layers(model)
        print("Freezing visual layers")
    elif bool(args.freeze_only_text):
        utils.model_utils.freeze_text_layers(model)
        print("Freezing text layers")

    val_transform = train_transform

    model.eval()
    return model, train_transform, val_transform, tokenizer
