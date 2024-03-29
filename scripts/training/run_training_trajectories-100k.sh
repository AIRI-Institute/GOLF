#! /bin/bash -ex

cuda=$1

CUDA_VISIBLE_DEVICES=$cuda \
python ../../main.py --n_parallel 240 \
--n_threads 24 \
--db_path ../../data/D-traj-100k.db \
--eval_db_path ../../D-test.db \
--num_initial_conformations -1 \
--sample_initial_conformations True \
--timelimit_train 1 \
--timelimit_eval 50 \
--terminate_on_negative_reward True \
--max_num_negative_rewards 1 \
--reward dft \
--minimize_on_every_step True \
--backbone painn \
--n_interactions 3 \
--cutoff 5.0 \
--n_rbf 50 \
--n_atom_basis 128 \
--actor GOLF \
--experience_saver reward_threshold \
--store_only_initial_conformations True \
--conformation_optimizer LBFGS \
--conf_opt_lr 1.0 \
--conf_opt_lr_scheduler Constant \
--max_iter 5 \
--lbfgs_device cpu \
--momentum 0.0 \
--lion_beta1 0.9 \
--lion_beta2 0.99 \
--batch_size 64 \
--lr 1e-4 \
--lr_scheduler CosineAnnealing \
--optimizer adam \
--clip_value 1.0 \
--energy_loss_coef 0.01 \
--force_loss_coef 0.99 \
--replay_buffer_size 1000000 \
--initial_conf_pct 1.0 \
--max_oracle_steps 100000 \
--utd_ratio 5 \
--subtract_atomization_energy True \
--action_norm_limit 1.0 \
--eval_freq 1200 \
--n_eval_runs 64 \
--eval_termination_mode fixed_length \
--exp_name traj-100k \
--full_checkpoint_freq  10000 \
--light_checkpoint_freq 50000 \
--save_checkpoints True \
--load_baseline ../../checkpoints/baseline-NNP/NNP_checkpoint \
--log_dir ../../results \
