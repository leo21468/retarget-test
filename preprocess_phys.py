import sys
sys.path.append("./")

import os
import os.path as osp
import yaml
import pathlib
import argparse
from tqdm import tqdm
import multiprocessing

from data_utils import process_amass_seq

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process AMASS sequences with enhanced error handling")
    parser.add_argument("--dataset_cfg", type=str, default=osp.join(osp.dirname(__file__), "../test_cfg.yaml"))
    parser.add_argument("--num_workers", type=int, default=None, 
                        help="Number of parallel workers (default: auto-detect based on CPU count)")
    args = parser.parse_args()

    # load yaml, read directory and motions
    try:
        with open(args.dataset_cfg, "r") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
    except FileNotFoundError:
        print(f"Error: Configuration file not found: {args.dataset_cfg}")
        print(f"Hint: Make sure the config file exists or provide a valid path using --dataset_cfg")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    # input/output dirs
    amass_dir = "/home/leo/experiment/retarget/test/ORIG_DATA"
    output_dir = os.path.join(os.path.dirname(__file__), "phys_processed")

    os.makedirs(output_dir, exist_ok=True)

    # selected motions
    candidates = cfg["motions"]
    
    # Dynamic CPU usage adjustment (for future parallel implementation)
    # Note: Currently processes sequentially; num_workers is reserved for future parallel processing
    if args.num_workers is None:
        cpu_count = multiprocessing.cpu_count()
        # Use at most half of available CPUs or the number of files, whichever is smaller
        num_workers = min(max(1, cpu_count // 2), len(candidates))
        print(f"Auto-detected {cpu_count} CPUs (sequential processing, {num_workers} workers calculated for future use)")
    else:
        num_workers = max(1, args.num_workers)
        print(f"Set to use {num_workers} workers (sequential processing, for future parallel implementation)")
    
    # Process sequences with error tracking
    success_count = 0
    error_count = 0
    errors = []
    
    pbar = tqdm(candidates)
    for seq in pbar:
        pbar.set_description(seq)  # display progress bar

        fname = os.path.join(amass_dir, seq.replace("+__+", "/")) # replace custom delimiter with directory separator
        output_path = os.path.join(output_dir, seq[:-4] + ".npy")
        
        try:
            if process_amass_seq(fname, output_path):
                success_count += 1
            else:
                error_count += 1
                errors.append(seq)
        except Exception as e:
            error_count += 1
            errors.append((seq, str(e)))
            print(f"\nError processing {seq}: {e}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Processing Summary:")
    print(f"  Total sequences: {len(candidates)}")
    print(f"  Successfully processed: {success_count}")
    print(f"  Failed: {error_count}")
    if errors:
        print(f"\nFailed sequences:")
        for error in errors:
            if isinstance(error, tuple):
                print(f"  - {error[0]}: {error[1]}")
            else:
                print(f"  - {error}")
    print(f"{'='*60}")
