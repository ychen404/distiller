import csv
from numpy import genfromtxt
import numpy as np


# filenames = ['fedavg_ep5_0.csv', 
#                 'fedavg_ep5_1.csv', 
#                 'fedavg_ep5_2.csv',
#                 'fedavg_ep5_3.csv',
#                 'fedavg_ep5_4.csv',
#                 'fedavg_ep5_5.csv',
#                 'fedavg_ep5_6.csv',
#                 'fedavg_ep5_7.csv',
#             ]

filenames = ['fedavg_20_client_noupdate0.csv', 
                'fedavg_20_client_noupdate1.csv', 
                'fedavg_20_client_noupdate2.csv',
                'fedavg_20_client_noupdate3.csv',
                'fedavg_20_client_noupdate4.csv',
                'fedavg_20_client_noupdate5.csv',
                'fedavg_20_client_noupdate6.csv',
                'fedavg_20_client_noupdate7.csv',
                'fedavg_20_client_noupdate8.csv',
                'fedavg_20_client_noupdate9.csv',
                'fedavg_20_client_noupdate10.csv',
                'fedavg_20_client_noupdate11.csv',
                'fedavg_20_client_noupdate12.csv',
                'fedavg_20_client_noupdate13.csv',
                'fedavg_20_client_noupdate14.csv',
                'fedavg_20_client_noupdate15.csv',
                'fedavg_20_client_noupdate16.csv',
                'fedavg_20_client_noupdate17.csv',
                'fedavg_20_client_noupdate18.csv',
                'fedavg_20_client_noupdate19.csv'
            ]

data = [None for filename in filenames]
output_name = 'fedavg_20_client_noupdate_avg.csv'

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