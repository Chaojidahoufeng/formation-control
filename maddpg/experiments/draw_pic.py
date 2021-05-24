# -*- coding: UTF8 -*-
import pickle
import json
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns


exp_name = '5-18-rel-formation-form-error-avoid-5-form-0_05-dist-0'

with open('../policy/'+exp_name+'/'+'commandline_args.txt', 'r', encoding="utf-8") as f:
  arglist = json.load(f)


rew_file_name = arglist['plots_dir'] + arglist['exp_name'] + '/' + arglist['exp_name'] + '_rewards.pkl'
step_file_name = arglist['plots_dir'] + arglist['exp_name'] + '/' + arglist['exp_name'] + '_steps.pkl'
crash_file_name = arglist['plots_dir'] + arglist['exp_name'] + '/' + arglist['exp_name'] + '_crashes.pkl'
done_file_name = arglist['plots_dir'] + arglist['exp_name'] + '/' + arglist['exp_name'] + '_done.pkl'

# 重点是rb和r的区别，rb是打开2进制文件，文本文件用r
with open(rew_file_name,'rb') as f:
  rew_data = pickle.load(f)

with open(step_file_name,'rb') as f:
  step_data = pickle.load(f)


df=pd.DataFrame({'epoch':range(len(rew_data)),'reward':rew_data})

ax = sns.lineplot(x="epoch", y="reward", data=df)
ax.set_title(exp_name)
plt.savefig(arglist['plots_dir'] + arglist['exp_name'] + '/'+'output_reward.png')
print('ok')