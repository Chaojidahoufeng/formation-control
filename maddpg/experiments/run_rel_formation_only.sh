srun -p gpu -N 1 -n 1 \
python -u train.py \
--scenario rel_formation_only \
--episode-dir ../policy/model_maddpg_rel_formation_only.npy
--save-dir ../policy/model_maddpg_rel_formation_only.ckpt