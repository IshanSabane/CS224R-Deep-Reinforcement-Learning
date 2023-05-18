#python cs224r/scripts/run_iql.py --env_name PointmassEasy-v0 \
#--exp_name iql_tau_0.9_rnd --use_rnd \
#--num_exploration_steps=20000 \
#--unsupervised_exploration \
#--awac_lambda=1 \
#--iql_expectile=0.9

python cs224r/scripts/run_iql.py --env_name PointmassMedium-v0 \
--exp_name iql_tau_0.9_rnd --use_rnd \
--num_exploration_steps=20000 \
--unsupervised_exploration \
--awac_lambda=1 \
--iql_expectile=0.9
