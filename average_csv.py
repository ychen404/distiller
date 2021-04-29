import csv
from numpy import genfromtxt
import numpy as np


filenames = ['fedavg_ep5_0.csv', 
                'fedavg_ep5_1.csv', 
                'fedavg_ep5_2.csv',
                'fedavg_ep5_3.csv',
                'fedavg_ep5_4.csv',
                'fedavg_ep5_5.csv',
                'fedavg_ep5_6.csv',
                'fedavg_ep5_7.csv',
            ]

data = [None for filename in filenames]
output_name = 'fedavg_ep5_avg.csv'

for i, filename in enumerate(filenames):
       data[i] = genfromtxt(filename, delimiter=',')
       # print(len(data[i]))
       # skip removing the trailing comma, only remove during plotting
    #    data[i] = data[i][:-1]
       print(f"Edge data length: {len(data[i])}")

# print(type(data))
# np.array(data)
avg = np.mean(data, axis=0)

# print(list(avg))
print(f"Cloud data length: {len(avg)}")

with open(output_name, 'w+') as fd:
    writer = csv.writer(fd)
    # fd.write(str(list(avg)))
    writer.writerow(list(avg))
