#!/bin/sh
srun -p gpu -N 1 -n 1 python -u train.py \
--exp-name 5-14-rel-formation-only-avoid-5-form-0_005-dist-0_02 \
--scenario rel_formation_only \
--save-dir model_maddpg_rel_formation_only.ckpt \
--avoid-rew-weight 5 \
--form-rew-weight 0.005 \
--dist-rew-weight 0.02 \
--action_space_dim 4 
