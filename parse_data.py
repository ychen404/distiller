from pathlib import Path
import os
import os, os.path
import json
import csv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from numpy import genfromtxt
import sys

# Data for plotting
filenames = ['try_this_0.csv', 'try_this_1.csv','try_this_2.csv', 'try_this_3.csv']

def plot_results(filenames, output_path):

    data = [None for filename in filenames]
    for i, filename in enumerate(filenames):
           data[i] = genfromtxt(filename, delimiter=',')
           data[i] = data[i][:-1]
    
    # plot for 100 epochs    
    t = np.arange(1, 101, 1)

    fig, ax = plt.subplots()

    for i, filename in enumerate(filenames):
           ax.plot(t, data[i])

    ax.set(xlabel='Epochs', ylabel='Accuracy',
           title='Accuracy result')
    ax.grid()
    print(os.curdir)

    fig.savefig(output_path + "/test.png")
    plt.show()


def write_to_file(filename, data):
    with open(filename, 'a+') as fd:
        fd.write(data + ',')


def parse_edge(path, csv_name):

    # path = "results/n_users_8_eps_10_cloud_eps_10_comm_10_e_r8_c_r8_frac_1_edge_no_update_constantlr_20210424-005146"
    new_path = Path(path).joinpath('cifar10')
    print(f"new_path: {new_path}")
    content = os.listdir(new_path)
    print(content)

    test_config = str(new_path) + '/test_config.json' 
    f = open(test_config)
    data = json.load(f)

    num_users = data['num_users']
    num_rounds = data['communication_round']

    content = content[0:len(content)-1]
    # prefix = 'resnet8_edge_0_acc.csv'
    prefix = 'resnet8_edge_'
    suffix = '_acc.csv'

    output_filenames = []
    for curr_round in range(num_rounds):
        temp_path = Path(new_path).joinpath(str(curr_round))
        # print(len(os.listdir(temp_path)))
        # print(temp_path)
        for curr_user in range(num_users):
            user_path = str(temp_path) + '/' 
            user_path = user_path + prefix + str(curr_user) + suffix
            print(user_path)
            with open(str(user_path), 'r') as fd:
                reader = csv.reader(fd, delimiter=',')
                next(reader)
                next(reader)
                for line in reader:
                    print(line)
                    # output_filename = str(new_path) + '/try_this_' + str(curr_user) + '.csv'
                    output_filename = csv_name + str(curr_user) + '.csv'
                    write_to_file(output_filename, line[1])
                    output_filenames.append(output_filename)


def parse_cloud(path, csv_name):

    new_path = Path(path).joinpath('cifar10')
    print(f"new_path: {new_path}")
    content = os.listdir(new_path)
    print(content)

    test_config = str(new_path) + '/test_config.json' 
    f = open(test_config)
    data = json.load(f)

    num_users = data['num_users']
    num_rounds = data['communication_round']

    content = content[0:len(content)-1]
    # prefix = 'resnet8_edge_0_acc.csv'
    prefix = 'resnet8_cloud'
    suffix = '_acc.csv'

    output_filenames = []
    for curr_round in range(num_rounds):
        temp_path = Path(new_path).joinpath(str(curr_round))
        temp_path = Path(temp_path).joinpath('cloud_model')
        # print(len(os.listdir(temp_path)))
        # print(temp_path)
        #     
        # for curr_user in range(num_users):
        user_path = str(temp_path) + '/' 
        user_path = user_path + prefix + suffix
        print(user_path)
        with open(str(user_path), 'r') as fd:
            reader = csv.reader(fd, delimiter=',')
            next(reader)
            next(reader)
            for line in reader:
                print(line)
                # output_filename = str(new_path) + '/try_this_' + str(curr_user) + '.csv'
                output_filename = csv_name + '.csv'
                write_to_file(output_filename, line[1])
                output_filenames.append(output_filename)
    
if __name__ == "__main__":
    path = sys.argv[1]
    parse_edge(path, 'fedavg_20_client_noupdate')
    # parse_cloud(path, 'edge_distillation_cloud')
    # output_path = str(new_path)
    # print(output_path)
    # plot_results(output_filenames, output_path)