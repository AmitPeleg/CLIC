import pathlib

ROOT_DIR = pathlib.Path(__file__).parent
OUTPUT_DIR = ROOT_DIR / "output"
EVAL_DIR = ROOT_DIR / "CompEvals"

SAVED_MODELS_DIR = OUTPUT_DIR
DATA_DIR = ROOT_DIR / "data"

# Training data
TRAIN_DATA_DIR = DATA_DIR
COGVLM_DATA_DIR = DATA_DIR / "laion_cogvlm" / "images"
COGVLM_CSV_FILE = DATA_DIR / "laion_cogvlm" / "laion_cogvlm.csv"
PIXPROSE_DATA_DIR = DATA_DIR / "redcaps_pixelprose" / "images"
PIXPROSE_CSV_FILE = DATA_DIR / "redcaps_pixelprose" / "redcaps_pixelprose.csv"

# Evaluation data
IMAGENET_DIR = DATA_DIR / "imagenet"
COCO2017_TRAIN_DIR = DATA_DIR / "train2017"
COCO2014_DIR = DATA_DIR / "coco"
COCO2017_DIR = DATA_DIR / "coco"
FLICKR30K_DIR = DATA_DIR / "flickr30k"
SUGARCREPE_DATA_DIR = DATA_DIR / "sugarcrepe"
SUGARCREPE_PP_DATA_DIR = DATA_DIR / "SugarCrepe_pp/data"