# Question 1.1

# python cs224r/scripts/run_hw1.py \
# --expert_policy_file cs224r/policies/experts/Ant.pkl \
# --env_name Ant-v4 --exp_name bc_ant --n_iter 1 --n_layers 2 \
# --expert_data cs224r/expert_data/expert_data_Ant-v4.pkl \
# --video_log_freq -1 \
# --seed 1


# Question 1.2: Changing the training batch size

# python cs224r/scripts/run_hw1.py \
# --expert_policy_file cs224r/policies/experts/Ant.pkl \
# --env_name Ant-v4 --exp_name bc_ant --n_iter 1 --n_layers 6 \
# --expert_data cs224r/expert_data/expert_data_Ant-v4.pkl \
# --video_log_freq -1 

# Question 2.1

python cs224r/scripts/run_hw1.py \
--expert_policy_file cs224r/policies/experts/Ant.pkl \
--env_name Ant-v4 --exp_name dagger_ant --n_iter 10 \
--do_dagger \
--expert_data cs224r/expert_data/expert_data_Ant-v4.pkl \