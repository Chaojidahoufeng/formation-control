srun -p gpu -N 1 -n 1 python -u train.py \
--scenario rel_based_formation_stream_avoidance_4 \
--episode-dir ../policy/model_maddpg_episode.npy \
--save-dir ../policy/model_maddpg.ckpt \
--restore 