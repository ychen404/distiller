import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from numpy import genfromtxt


# Data for plotting
# filenames = ['try_this_0.csv', 'try_this_1.csv', 'try_this_2.csv', 'try_this_3.csv', 'try_this_4.csv',
# 'try_this_5.csv', 'try_this_6.csv', 'try_this_7.csv']

# filenames = ['no_edge_distill0.csv', 'no_edge_distill1.csv', 'no_edge_distill2.csv', 'no_edge_distill3.csv']


# filenames = ['no_edge_distillation_cloud.csv', 
#               'no_edge_distill_avg.csv',
#               'edge_distill_avg.csv',
#               'edge_distillation_cloud.csv',
#               'fedavg_edge_avg.csv',
#               'fedavg_noupdate_avg.csv'              
#               ]
# labels = {}
# labels['no_edge_distillation_cloud.csv'] = 'no_edge_distillation_cloud'
# labels['no_edge_distill_avg.csv'] = 'no_edge_distill_edge'
# labels['edge_distill_avg.csv'] = 'edge_distill_edge'
# labels['edge_distillation_cloud.csv'] = 'edge_distillation_cloud'
# labels['fedavg_edge_avg.csv'] = 'fedavg_edge'
# labels['fedavg_noupdate_avg.csv'] = 'fedavg_noupdate_edge'

filenames = [
              'fedavg_ep5_avg.csv',
              'fedavg_ep5_noupdate_avg.csv',
              'fedavg_20_client_avg.csv',
              'fedavg_20_client_noupdate_avg.csv'
              ]
labels = {}

labels['fedavg_ep5_avg.csv'] = 'fedavg_8_client'
labels['fedavg_ep5_noupdate_avg.csv'] = 'fedavg_8_client_noupdate'
labels['fedavg_20_client_avg.csv'] = 'fedavg_20_client'
labels['fedavg_20_client_noupdate_avg.csv'] = 'fedavg_20_client_noupdate'

# plot_title = filenames[0].split('.')[0]

data = [None for filename in filenames]


for i, filename in enumerate(filenames):
       data[i] = genfromtxt(filename, delimiter=',')
       # print(len(data[i]))
       # remove the trailing comma
       data[i] = data[i][:-1]

# t = np.arange(1, 101, 1)
t = np.arange(1, 1251, 1)


fig, ax = plt.subplots()
# fig.set_label('label via method')
# ax.legend()

for i, filename in enumerate(filenames):
       ax.plot(t, data[i], label=labels[filename])
       ax.legend()
# plt.legend(handles=[line])
# handles, labels = ax.get_legend_handles_lables()
# ax.legend(handles, labels)
plot_title = 'Accuracy with different clients'

ax.set(xlabel='Epochs', ylabel='Accuracy',
       title=str(plot_title))
ax.grid()
print(os.curdir)
plt.tight_layout()
fig.savefig(plot_title + ".png")
# plt.show()


