#! /bin/bash -ex

cuda=$1

CUDA_VISIBLE_DEVICES=$cuda \
python ../../main.py --n_parallel 48 \
--n_threads 24 \
--db_path ../../data/D-0.db \
--eval_db_path ../../data/D-test.db \
--num_initial_conformations -1 \
--sample_initial_conformations True \
--timelimit_train 100 \
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
--store_only_initial_conformations False \
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
--initial_conf_pct 0.1 \
--max_oracle_steps 1000 \
--utd_ratio 500 \
--subtract_atomization_energy True \
--action_norm_limit 1.0 \
--eval_freq 48 \
--n_eval_runs 48 \
--eval_termination_mode fixed_length \
--exp_name GOLF-1k \
--host_file_path ../../env/host_names.txt \
--full_checkpoint_freq  96 \
--light_checkpoint_freq 192 \
--save_checkpoints True \
--load_baseline ../../checkpoints/baseline-NNP/NNP_checkpoint \
--log_dir ../../results \
