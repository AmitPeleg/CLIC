from datasets import load_dataset

from local_setting import PIXPROSE_DATA_DIR

PIXPROSE_DATA_DIR.mkdir(parents=True, exist_ok=True)
ds_redcaps = load_dataset("tomg-group-umd/pixelprose", split="redcaps", cache_dir=PIXPROSE_DATA_DIR)

