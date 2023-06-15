
python3.9 main.py --n_parallel 240 \
--n_threads 24 \
--db_path "env/data/train_4k_mff_with_forces_wooutlier.db" \
--eval_db_path "env/data/test_4k_mff.db" \
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
--optimizer lion \
--lr_scheduler OneCycleLR \
--clip_value 1.0 \
--energy_loss_coef 0.01 \
--force_loss_coef 0.99 \
--replay_buffer_size 500000 \
--max_timesteps 500000 \
--subtract_atomization_energy True \
--action_norm_limit 1.0 \
--eval_freq 1000 \
--n_eval_runs 10 \
--eval_termination_mode fixed_length \
--exp_name DFT_LBFGS \
--seed 100 \
--full_checkpoint_freq 10000 \
--light_checkpoint_freq 50000 \
--save_checkpoints True \
--load_baseline "models/DFT-baseline/full_cp_iter_500000" \
--log_dir results \
