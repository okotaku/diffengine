import argparse
import os.path as osp

import joblib
import mmengine
import numpy as np
import pandas as pd
from mmengine.config import Config
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process a checkpoint to be published")
    parser.add_argument("config", help="Path to config")
    parser.add_argument("--n_jobs", help="Number of jobs.", type=int,
                        default=4)
    parser.add_argument("--out", help="Output path", default="bucked_ids.pkl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = Config.fromfile(args.config)
    data_dir = cfg.train_dataloader.dataset.get("dataset")
    img_dir = cfg.train_dataloader.dataset.get("img_dir", "")
    csv = cfg.train_dataloader.dataset.get("csv", "metadata.csv")
    csv_path = osp.join(data_dir, csv)
    img_df = pd.read_csv(csv_path)

    sizes = cfg.train_dataloader.dataset.pipeline[0].get("sizes")
    aspect_ratios = np.array([s[0] / s[1] for s in sizes])

    def get_bucket_id(file_name):
        image = osp.join(data_dir, img_dir, file_name)
        image = Image.open(image)
        aspect_ratio = image.height / image.width
        return np.argmin(np.abs(aspect_ratios - aspect_ratio))

    bucket_ids = joblib.Parallel(n_jobs=args.n_jobs, verbose=10)(
        joblib.delayed(get_bucket_id)(file_name)
        for file_name in tqdm(img_df.file_name.values))

    print(pd.DataFrame(bucket_ids).value_counts())

    mmengine.dump(bucket_ids, args.out)

if __name__ == "__main__":
    main()
