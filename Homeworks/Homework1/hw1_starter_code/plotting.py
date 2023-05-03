import seaborn as sns 
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
import os 

# sns.set_style('ggplot2')



df = pd.read_csv('plot_data/layer_variation.csv')


plt.figure(figsize=(8,6))

plt.plot(df['Layers'], df['avg_return'] , color = 'red' , marker = 'o', label= 'Average Return')



plt.xticks(df['Layers'].tolist())

plt.title(f'Averge Return of BC Agent in Ant Environment  vs Number of Layers of the Network')
plt.xlabel('Number of Hidden Layers in the Neural Network')
plt.ylabel('Average Return')
plt.legend()

plt.savefig(f'BC_Ant_Variation.png')
plt.show()

exit()










# env='Ant'

env='Walker2d'



df = pd.read_csv(f'/home/ishan05/Spring 2023/CS224R/CS224R-Deep-Reinforcement-Learning-/Homeworks/Homework1/hw1_starter_code/plot_data/{env}_dagger.csv')

print(df)
print(df.columns)

expert_reward = {"Ant" : 4713.6533203125, "Walker2d":5566.845}
bc_reward = {"Ant" : 4190.13, "Walker2d":377.017120361328}
 

df['lb'] = df['avg_return'] - df['std_return']

df['ub'] = df['avg_return'] + df['std_return']

errors = list( zip(df['lb'].tolist(), df['ub'].tolist()) )



plt.figure(figsize=(8,6))


# plt.plot(data_mean, marker='o', color='black')

# sns.lineplot(data = df, x= 'Step', y='avg_return',  errorbar = func, err_style='bars')


plt.errorbar(df['Step'].tolist(), df['avg_return'].tolist(), yerr = df['std_return'].tolist() , marker ='o', color = 'green', ecolor='red',  capsize=5, solid_capstyle='projecting', label = ' DAgger Averge Return')
plt.axhline(bc_reward[env], color = 'orange', linestyle = 'dashed', label = 'BC Agent Return')
plt.axhline(expert_reward[env], color = 'red', linestyle = 'dashed', label = 'Expert Return')


plt.xticks(df['Step'].tolist())

plt.title(f'{env} DAgger Average Return vs Iterations')
plt.xlabel('Iterations')
plt.ylabel('Return')
plt.legend()

plt.savefig(f'Dagger_{env}.png')
plt.show()




