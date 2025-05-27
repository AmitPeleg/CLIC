<div align="center">

# Advancing Compositional Awareness in CLIP with Efficient Fine-Tuning
	
#### [Project page](https://***.github.io) | [Paper](https://arxiv.org/abs/***) 

</div>

> Advancing Compositional Awareness in CLIP with Efficient Fine-Tuning |
> [Amit Peleg*](mailto:amit.peleg@uni-tuebingen.de), [Naman Singh*](mailto:naman-deep.singh@uni-tuebingen.de), [Matthias Hein](https://uni-tuebingen.de/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/informatik/lehrstuehle/maschinelles-lernen/team/prof-dr-matthias-hein/) |
> arXiv, 2025

### Installation

``` bash
conda create --name clic python=3.12
conda activate clic
conda install pytorch==2.2.2 torchvision==0.17.2 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r pip_reqs.txt
python -m spacy download en_core_web_sm
```
### Evaluation

```bash
bash eval.sh
```
- Choose the `architecture` in the [bash file](eval.sh).
- Choose the `modelName` in the [bash file](eval.sh).
  - For pre-trained non-CLIC models,  use the `Pre-train key` from the table below.
  - For CLIC models, use the `CLIC FT-key` from the table below.
  - For evaluating your own checkpoints, use the `Pre-train key` from the training and add the argument `--load_pretrained_clip path\to\ckpt\folder` to the eval file.
- Evaluation datasets (ImageNet, COCO, SugarCrepe, SugarCrepe++, etc) need to be downloaded by the user.
- Make sure the evaluation dataset paths in [local_settings](/local_setting.py) are correct.


<br>

<div align="center">
	
| Model name              | Pre-train key    | CLIC FT-key | CLIC-model HF-link                             |
|-------------------------|------------|-------------------------|-------------------------------------------|
ViT-B-32-CogVLM  | ViT-B-32 |HF-CLIC-ViT-B-32-224-CogVLM | [HF-Link](https://huggingface.co/nmndeep/CLIC-ViT-B-32-224-CogVLM) |
ViT-B-32-PixelProse  | ViT-B-32 |HF-CLIC-ViT-B-32-224-PixelProse| [HF-Link](https://huggingface.co/nmndeep/CLIC-ViT-B-32-224-PixelProse) |
ViT-B-16-CogVLM  | ViT-B-16 |HF-CLIC-ViT-B-16-224-CogVLM| [HF-Link](https://huggingface.co/nmndeep/CLIC-ViT-B-16-224-CogVLM) |
ViT-L-14-CogVLM  | ViT-L-14 |HF-CLIC-ViT-L-14-224-CogVLM| [HF-Link](https://huggingface.co/nmndeep/CLIC-ViT-L-14-224-CogVLM) |
ViT-L-14-PixelProse  | ViT-L-14 |HF-CLIC-ViT-L-14-224-PixelProse| [HF-Link](https://huggingface.co/nmndeep/CLIC-ViT-L-14-224-PixelProse) |
CLIPA-CogVLM  | CLIPA |HF-CLIC-CLIPA-ViT-L-14-224-CogVLM| [HF-Link](https://huggingface.co/nmndeep/CLIC-CLIPA-ViT-L-14-224-CogVLM) |
CLIPA-PixelProse  | CLIPA |HF-CLIC-CLIPA-ViT-L-14-224-PixelProse| [HF-Link](https://huggingface.co/nmndeep/CLIC-CLIPA-ViT-L-14-224-PixelProse) |
CLIPS-CogVLM  | CLIPS |HF-CLIC-CLIPS-ViT-L-14-224-CogVLM| [HF-Link](https://huggingface.co/nmndeep/CLIC-CLIPS-ViT-L-14-224-CogVLM) |
CLIPS-PixelProse  | CLIPS |HF-CLIC-CLIPS-ViT-L-14-224-PixelProse| [HF-Link](https://huggingface.co/nmndeep/CLIC-CLIPS-ViT-L-14-224-PixelProse) |
-------------------------------------------------------------------------------------------------
Note: with the correct key in `modelName` variable in `eval.sh`, the models would be downloaded automatically. 

</div>

<br>

### Training datasets
We fine-tune different models with CLIC using 

-  [CogVLM relabelled 1M Laion samples](https://huggingface.co/datasets/nmndeep/CLIC-CogVLM-relabelled-Laion)
-  RedCaps subset from the [PixelProse dataset](https://huggingface.co/datasets/tomg-group-umd/pixelprose)

The default location for the datasets is in the [`data`](data) folder. 
You can change the location of each dataset in the [local_settings](local_setting.py) file.

#### 1M subset of the LIAON dataset
```bash
mkdir data
# Download the csv file with the images urls
wget -O data/CLIC-CogVLM-relabelled-Laion.csv https://huggingface.co/datasets/nmndeep/CLIC-CogVLM-relabelled-Laion 
# Download the 1M Laion subset and create csv with the image locations
python -m assets.download_cogvlm
```
#### RedCaps subset of the PixelProse dataset
```bash
# Download the redcaps images as described in https://huggingface.co/datasets/tomg-group-umd/pixelprose
python -m assets.download_redcaps
# Process the captions and create the csv file
# If you changed the default location, make sure to change the output path argument as well
python -m assets.create_dataset --input_file data/path/to/downloaded/csv/file.csv --output_file data/redcaps_pixelprose/redcaps_pixelprose.csv
```

### CLIC Training

- Change the `dataset` variable in the [`trigger_train.sh`](trigger_train.sh) to `laion_cogvlm`/`redcaps_pixelprose`
- Change the `modelName` and `architecture` variables as desired in [`trigger_train.sh`](trigger_train.sh).
  - For the `modelName` use the `Pre-train key` from the table above.
- Make sure the csv file paths in [local_settings](/local_setting.py) are correct.
- You can run training without evaluation by adding the `--no_eval` argument to the training script.

```bash
bash trigger_train.sh
```

#### For training with single image (non concat) version - our NegCLIP:
```bash
bash trigger_train_negclip.sh
```

#### For training the baseline of CLIC (Single-Image Baseline in Table 10):
```bash
bash trigger_train_baseline.sh
```

### Acknowledgements
This work uses code/models from:

1. [OpenCLIP](https://github.com/mlfoundations/open_clip/tree/main)

2. [https://github.com/UCSC-VLAA/CLIPS](https://github.com/UCSC-VLAA/CLIPS)

3. [https://github.com/UCSC-VLAA/CLIPA](https://github.com/UCSC-VLAA/CLIPA)

4. [https://github.com/LijieFan/LaCLIP](https://github.com/LijieFan/LaCLIP)


