#!/bin/bash

cfg_path="./test_cfg.yaml"

echo "Processing physical data..."
printf "\n\n"
python ./preprocess_phys.py --dataset_cfg $cfg_path
python ./phys_to_smpl.py
python ./smpl_to_smplx.py --src_folder ./test/phys_smpl --tgt_folder ./test/phys_smplx
printf "\n\n"
echo "Retargeting to Unitree G1..."
python ./smplx_to_g1_repo.py --src_folder ./test/phys_smplx --tgt_folder ./test --robot unitree_g1
printf "\n\n"
echo "Done!"
