# Question 1.1

# python cs224r/scripts/run_hw1.py \
# --expert_policy_file cs224r/policies/experts/Ant.pkl \
# --env_name Ant-v4 --exp_name bc_ant --n_iter 1 --n_layers 2 \
# --expert_data cs224r/expert_data/expert_data_Ant-v4.pkl \
# --video_log_freq -1 \
# --seed 1 --batch_size 1000 --train_batch_size 100 --eval_batch_size 5000


# Question 1.1 Environment where it fails

# python cs224r/scripts/run_hw1.py \
# --expert_policy_file cs224r/policies/experts/Walker2d.pkl \
# --env_name Walker2d-v4 --exp_name bc_ant --n_iter 1 --n_layers 2 \
# --expert_data cs224r/expert_data/expert_data_Walker2d-v4.pkl \
# --video_log_freq -1 \
# --seed 1 --batch_size 1000 --train_batch_size 100 --eval_batch_size 5000




# Question 1.2: Changing the training batch size

python cs224r/scripts/run_hw1.py \
--expert_policy_file cs224r/policies/experts/Ant.pkl \
--env_name Ant-v4 --exp_name bc_ant --n_iter 1 --n_layers 10 \
--expert_data cs224r/expert_data/expert_data_Ant-v4.pkl \
--video_log_freq -1 --batch_size 1000 --train_batch_size 100 --eval_batch_size 5000

# # Question 2.1


# python cs224r/scripts/run_hw1.py \
# --expert_policy_file cs224r/policies/experts/Ant.pkl \
# --env_name Ant-v4 --exp_name dagger_ant --n_iter 10 \
# --do_dagger \
# --expert_data cs224r/expert_data/expert_data_Ant-v4.pkl \
# --batch_size 1000 --train_batch_size 100 --eval_batch_size 5000




# python cs224r/scripts/run_hw1.py \
# --expert_policy_file cs224r/policies/experts/Walker2d.pkl \
# --env_name Walker2d-v4 --exp_name dagger_ant --n_iter 10 \
# --do_dagger \
# --expert_data cs224r/expert_data/expert_data_Walker2d-v4.pkl \
# --batch_size 1000 --train_batch_size 100 --eval_batch_size 5000



