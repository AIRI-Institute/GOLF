#! /bin/bash -ex

cuda=$1

CUDA_VISIBLE_DEVICES=$cuda \
python main.py --n_parallel 216 \
--n_threads 24 \
--db_path env/data/train_4k_mff_with_forces_wooutlier.db \
--eval_db_path env/data/test_4k_mff_optimized.db \
--num_initial_conformations -1 \
--sample_initial_conformations True \
--timelimit_train 100 \
--timelimit_eval 100 \
--terminate_on_negative_reward True \
--max_num_negative_rewards 1 \
--reward dft \
--minimize_on_every_step True \
--molecules_xyz_prefix env/molecules_xyz \
--backbone painn \
--n_interactions 3 \
--cutoff 5.0 \
--n_rbf 50 \
--n_atom_basis 128 \
--actor AL \
--conformation_optimizer LBFGS \
--conf_opt_lr 1.0 \
--conf_opt_lr_scheduler Constant \
--experience_saver reward_threshold \
--grad_missmatch_threshold 1.0 \
--store_only_initial_conformations False \
--max_iter 5 \
--lbfgs_device cpu \
--momentum 0.0 \
--lion_beta1 0.9 \
--lion_beta2 0.99 \
--batch_size 64 \
--lr 2e-6 \
--lr_scheduler OneCycleLR \
--optimizer lion \
--clip_value 1.0 \
--energy_loss_coef 0.01 \
--force_loss_coef 0.99 \
--replay_buffer_size 1000000 \
--initial_conf_pct 0.1 \
--max_timesteps 700000 \
--subtract_atomization_energy True \
--action_norm_limit 1.0 \
--eval_freq 1080 \
--n_eval_runs 64 \
--eval_termination_mode fixed_length \
--exp_name DFT_LBFGS \
--full_checkpoint_freq 10000 \
--light_checkpoint_freq 50000 \
--save_checkpoints True \
--log_dir results \
