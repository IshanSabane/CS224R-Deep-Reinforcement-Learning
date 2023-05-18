#python cs224r/scripts/run_cql.py --env_name PointmassMedium-v0 \
#--exp_name cql_alpha_0.1_random \
#--unsupervised_exploration \
#--offline_exploitation --cql_alpha=0.1


python cs224r/scripts/run_cql.py --env_name PointmassMedium-v0 \
--exp_name cql_alpha_0.1_rnd --use_rnd \
--unsupervised_exploration \
--offline_exploitation --cql_alpha=0.1
