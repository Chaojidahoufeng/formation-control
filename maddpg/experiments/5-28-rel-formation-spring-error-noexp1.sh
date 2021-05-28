srun -p gpu -N 1 -n 1 python -u train.py \
--exp-name 5-18-rel-formation-spring-error-avoid-5-form-0_05-dist-0 \
--scenario rel_formation_form_error \
--save-dir model_maddpg_rel_formation_only.ckpt \
--avoid-rew-weight 5 \
--form-rew-weight 0.05 \
--dist-rew-weight 0. \
--action_space_dim 4 \
--num-episodes 200000 