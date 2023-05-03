import pickle 

path_policy = "/home/ishan05/Spring 2023/CS224R/CS224R-Deep-Reinforcement-Learning-/Homeworks/Homework1/hw1_starter_code/cs224r/policies/experts/Ant.pkl"

f = open("/home/ishan05/Spring 2023/CS224R/CS224R-Deep-Reinforcement-Learning-/Homeworks/Homework1/hw1_starter_code/cs224r/expert_data/expert_data_Ant-v4.pkl", 'rb')

f = open(path_policy,'rb')
data = pickle.load(f)


print(data.keys())



