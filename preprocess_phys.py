import sys
sys.path.append("./")

import os
import os.path as osp
import yaml
import pathlib
import argparse
from tqdm import tqdm

from data_utils import process_amass_seq

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset_cfg", type=str, default=osp.join(osp.dirname(__file__), "../test_cfg.yaml"))
    args = parser.parse_args()

    # load yaml, read directory and motions
    with open(args.dataset_cfg, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # input/output dirs
    amass_dir = "/home/leo/experiment/retarget/test/ORIG_DATA"
    output_dir = os.path.join(os.path.dirname(__file__), "phys_processed")

    os.makedirs(output_dir, exist_ok=True)

    # selected motions
    candidates = cfg["motions"]
    
    pbar = tqdm(candidates)
    for seq in pbar:

        pbar.set_description(seq)  # display progress bar

        fname = os.path.join(amass_dir, seq.replace("+__+", "/")) # replace custom delimiter with directory separator
        output_path = os.path.join(output_dir, seq[:-4] + ".npy")
        
        process_amass_seq(fname, output_path)

    print("Processed {} sequences!".format(len(candidates)))
