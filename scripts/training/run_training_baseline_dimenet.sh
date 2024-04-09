#! /bin/bash -ex

cuda=$1

CUDA_VISIBLE_DEVICES=$cuda \
python ../../main.py --n_parallel 240 \
--n_threads 24 \
--db_path ../../data/D-0.db \
--eval_db_path ../../data/D-test.db \
--num_initial_conformations -1 \
--eval_num_initial_conformations 2000 \
--sample_initial_conformations True \
--timelimit_train 1 \
--timelimit_eval 50 \
--terminate_on_negative_reward True \
--max_num_negative_rewards 1 \
--reward dft \
--minimize_on_every_step True \
--nnp_type DimenetPlusPlusOC \
--nnp_config_path /mnt/2tb/tsypin/MARL/GOLF-pyg/configs/dimenetplusplus_oc.yaml \
--actor GOLF \
--experience_saver reward_threshold \
--store_only_initial_conformations True \
--conformation_optimizer LBFGS \
--conf_opt_lr 1.0 \
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
--forces_norm_limit 1.0 \
--eval_freq 1200 \
--n_eval_runs 64 \
--eval_termination_mode fixed_length \
--exp_name dimenetplusplus_oc \
--full_checkpoint_freq  10000 \
--light_checkpoint_freq 50000 \
--save_checkpoints True \
--log_dir ../../results \
--project_name GOLF-pyg-baseline \
# --load_model ../../results/baseline-NNP-painn_2024_03_14_01_47_05_561583/full_cp_iter_49200 \
