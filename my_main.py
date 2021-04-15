import argparse
from pathlib import Path

# from distillers import *
from data_loader import get_cifar
from models.model_factory import create_model
from trainer import BaseTrainer, KDTrainer, MultiTeacher
from plot import plot_results
import util
import torch.nn as nn
from sampling import cifar_iid
import psutil
import os
import pdb
import logging
import time
import numpy as np

BATCH_SIZE = 128
TESTFOLDER = "results"
USE_ID = True
COMMUNICATION_ROUND = 10
EPOCHS = 100
NTHREAD = 4
ACC_NAME = 'cloud_model_test_acc.csv'

timestr = time.strftime("%Y%m%d-%H%M%S")
fmt_str = '[%(levelname)s] %(filename)s @ line %(lineno)d: %(message)s'

def parse_arguments():
    parser = argparse.ArgumentParser(description="Test parameters")
    
    
    # federated arguments
    parser.add_argument("--epochs", default=EPOCHS, type=int,
                        help="number of total epochs to run")
    parser.add_argument("--num_users", default=2,
                        type=int, help="number of users") 
    parser.add_argument("--communication_round", default=COMMUNICATION_ROUND, type=int,
                        help="number of total global epochs to run")
    parser.add_argument('--frac', default=0.1, type=float, help="the fraction of clients: C")
    parser.add_argument("--batch-size", default=BATCH_SIZE, type=int, help="batch_size")
    
    # model arguments
    parser.add_argument("--dataset", default="cifar100", type=str,
                        help="dataset. can be either cifar10 or cifar100")
    parser.add_argument("--edge_distillation", default=1,
                        type=int, help="enable edge distillation")
    parser.add_argument("--learning-rate", default=0.1,
                        type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9,
                        type=float, help="SGD momentum")
    parser.add_argument("--weight-decay", default=5e-4,
                        type=float, help="SGD weight decay (default: 5e-4)")
    parser.add_argument("--edge", default="resnet8", type=str,
                        dest="e_name", help="edge model name")
    parser.add_argument("--cloud", "--model", default="resnet8",
                        dest="c_name", type=str, help="cloud model name")
    parser.add_argument("--optimizer", default="sgd",
                        dest="optimizer", type=str,
                        help="Which optimizer to use")
    
    # other arguments
    parser.add_argument('--gpu_id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--ngpu', default=1, type=int,
                    help='number of GPUs to use for training')
    parser.add_argument("--scheduler", default="multisteplr",
                        dest="scheduler", type=str,
                        help="Which scheduler to use")
    parser.add_argument("--teacher-checkpoint", default="",
                        dest="t_checkpoint", type=str,
                        help="optional pretrained checkpoint for teacher")
    parser.add_argument("--mode", default=["KD"], dest="modes",
                        type=str, nargs='+',
                        help="What type of distillation to use")
    parser.add_argument("--nthread", default=NTHREAD,
                        type=int, help="number of threads for dataloader")                    
    parser.add_argument("--workspace", default="",
                        type=str, help="The prefix for the output files")                    
    parser.add_argument("--results-dir", default=TESTFOLDER,
                        dest="results_dir", type=str,
                        help="Where all results are collected")
    parser.add_argument("--log", default="info",
                        help="Provide logging level")

    args = parser.parse_args()

    return args

def setup_n_train_edge(edge_name, edge_params, partition_idx, edge_suffix):
    # Teacher Model
    # This function is not used. I separate the setup and the training part. 
    # See setup_edge()
    logger.info("Setting up edge model")
    num_classes = edge_params["num_classes"]
    t_net = create_model(edge_name, num_classes, edge_params["device"])
    edge_config = edge_params.copy()

    edge_config["test_name"] = edge_name + edge_suffix
    
    dict_users = edge_params['dict_users']

    if edge_params["t_checkpoint"]:
        # Just validate the performance
        # print("---------- Loading Edge Model -------")
        logger.info('+'*20 + " Loading Edge Model " + '+'*20)
        edge_ckpt = edge_params["t_checkpoint"]
        logger.debug(f"pretrained edge model: {edge_ckpt}" )
    else:
        # Teacher training
        # print("---------- Training Edge Model -------")
        logger.info('-'*20 + ' Training Edge Model ' + '-'*20)
        edge_trainer = BaseTrainer(t_net, config=edge_config, idxs=dict_users[partition_idx])
        edge_trainer.train()
        edge_ckpt = edge_trainer.model_file
        print(f"Teacher checkpoint: {edge_ckpt}")

    # load teacher's checkpoint
    # t_net = util.load_checkpoint(t_net, edge_ckpt) 
    # edge_trainer = BaseTrainer(t_net, config=edge_config)
    
    # best_t_acc = edge_trainer.validate()
  
    # also save this information in a csv file for plotting
    # name = edge_config["test_name"] + "_val"

    # acc_file_name = edge_params["results_dir"].joinpath(f"{name}.csv")
    # with acc_file_name.open("w+") as acc_file:
    #     acc_file.write("Training Loss,Validation Loss\n")
    #     for _ in range(edge_params["epochs"]):
    #         acc_file.write(f"0.0,{best_t_acc}\n")
        
    return t_net, edge_ckpt

def setup_edge(edge_name, edge_params):
    # Teacher Model
    logger.debug("Setting up edge model")
    num_classes = edge_params["num_classes"]

    # t_net is the model itself
    t_net = create_model(edge_name, num_classes, edge_params["device"])
    
    return t_net

def train_edge(current_round, t_net, edge_params, edge_name, partition_idx, edge_suffix):
    
    # logger.info("Training edge model")
    # defrost because from the seoncd round the edge models are frozen during distillation
    t_net = defrost_net(t_net)
    
    edge_config = edge_params.copy()
    edge_config["test_name"] = edge_name + edge_suffix
    dict_users = edge_params['dict_users']

    # Trainer wraps the training related functions together
    edge_trainer = BaseTrainer(t_net, config=edge_config, idxs=dict_users[partition_idx])
    edge_trainer.train(current_round)
    edge_ckpt = edge_trainer.model_file

    return t_net, edge_ckpt

   
def continue_training_single_edge(t_net, t_name, params):
    
    t_net = defrost_net(t_net)
    # print(f"t_name: {t_name}, type: {type(t_net)}")
    teacher_config = params.copy()
    teacher_config["test_name"] = t_name + "_teacher"
    
    teacher_trainer = BaseTrainer(t_net, config=teacher_config)
    teacher_trainer.train()
    best_teacher = teacher_trainer.best_model_file

    # print(f"best_teacher: {best_teacher}")
    
    logger.debug(f"best_teacher: {best_teacher}")
    best_t_acc = teacher_trainer.validate()

    return t_net, best_teacher, best_t_acc

def continue_training_multi_edge(t_nets, t_name, params):
    
    best_teachers = []
    best_t_accs = []
    for idx, t_net in enumerate(t_nets):
        t_net = defrost_net(t_net)
        # print(f"t_name: {t_name}, type: {type(t_net)}")
        logger.debug(f"t_name: {t_name}, type: {type(t_net)}")
        teacher_config = params.copy()
        teacher_config["test_name"] = t_name + "_teacher"
        dict_users = params['dict_users']
        teacher_trainer = BaseTrainer(t_net, config=teacher_config, idxs=dict_users[idx])
        
        teacher_trainer.train()
        best_teacher = teacher_trainer.best_model_file
        logger.debug(f"best_teacher: {best_teacher}")

        best_t_acc = teacher_trainer.validate()
        best_teachers.append(best_teacher)
        best_t_accs.append(best_t_acc)

    return t_nets, best_teachers, best_t_accs

def setup_cloud(s_name, params):
    # Student Model
    num_classes = params["num_classes"]
    s_net = create_model(s_name, num_classes, params["device"])
    return s_net

def freeze_net(net):
    # freeze the layers of a model
    for param in net.parameters():
        param.requires_grad = False
    # set the teacher net into evaluation mode
    net.eval()
    return net

def defrost_net(net):
    # freeze the layers of the teacher
    for param in net.parameters():
        param.requires_grad = True
    # set the teacher net into train mode
    net.train()
    return net

def test_kd(s_net, t_net, params):
    t_net = freeze_net(t_net)
    logger.info("---------- Training KD -------")
    kd_config = params.copy()
    kd_trainer = KDTrainer(s_net, t_net=t_net, config=kd_config)
    best_acc = kd_trainer.train()
    
    return best_acc

def single_edge_kd(s_net, t_net, params):
    t_net = freeze_net(t_net)
    logger.info("---------- KD: single edge model teaching cloud model -------")
    kd_config = params.copy()
    kd_trainer = KDTrainer(s_net, t_net=t_net, config=kd_config)
    best_acc = kd_trainer.train()
    
    return best_acc, s_net, t_net

def multi_edge_kd(current_round, cloud_net, edge_nets, params):
    # Use CIFAR-100 unlabeled data for KD with multiple edge models
    logger.info("---------- KD: multiple edge models teaching cloud model -------")
    frozen_edge_nets = []
    for e_net in edge_nets:
        frozen_edge_nets.append(freeze_net(e_net))
    
    kd_config = params.copy()
    kd_trainer = MultiTeacher(cloud_net, t_nets=frozen_edge_nets, config=kd_config)
    cloud_acc = kd_trainer.train(current_round)
    return cloud_acc, cloud_net, edge_nets

def run_benchmarks(modes, args, params_edge, params_cloud, c_name, e_name):
    results = {}     
    mode = modes[0]
    mode = mode.lower()

    # Setup teacher models
    edge_nets = []
    logger.debug(f"params['num_users'] = {params_edge['num_users']}")

    # Teacher model initial training with local data

    m = max(int(args.frac * args.num_users), 1)
    idxs_users = np.random.choice(range(args.num_users), m, replace=False)

    # # for user in range(params_edge['num_users']):
    # for user, partition_idx in enumerate (idxs_users):
    #     logger.info(f"Edge model: {user + 1}, partition_idx: {partition_idx}\n")
    #     edge_net, _, = setup_n_train_edge(e_name, params_edge, partition_idx, edge_suffix='_edge_'+ str(user))
    #     edge_nets.append(edge_net)


    # Setup the edge model
    for user, partition_idx in enumerate (idxs_users):
        logger.info(f"Edge model: {user + 1}, partition_idx: {partition_idx}")
        edge_net = setup_edge(e_name, params_edge)
        edge_nets.append(edge_net)

    # Setup the cloud model 
    cloud_net = setup_cloud(c_name, params_cloud)
    
    # Setup directories
    
    params_cloud["test_name"] = c_name + '_cloud'
    # params_cloud["results_dir"] = params_cloud["results_dir"].joinpath('cloud_model')
    # util.check_dir(params_cloud["results_dir"])

    # Edge and cloud share the same base path for the results
    base_path = params_cloud["results_dir"] 
    
    # Looping through the communication rounds
    start = time.time()
    for i in range(0, params_edge["communication_round"]):
        logger.info(f"Starting {i} communication_round")
        current_round = i
        # Save every round of data in separate directories
        
        params_cloud["results_dir"] = base_path.joinpath(str(i)).joinpath('cloud_model')
        params_edge["results_dir"] = base_path.joinpath(str(i))

        # cloud_path = cloud_path.joinpath('cloud_model')
        util.check_dir(params_cloud["results_dir"])
        
        # Train all the edge models
        for user, partition_idx in enumerate (idxs_users):
            logger.info(f"Training {user + 1}th edge model")
            edge_net, _, = train_edge(current_round, edge_nets[user], params_edge, e_name, partition_idx, edge_suffix='_edge_'+ str(user))

        if mode == "kd":
            logger.info("==== Running kd mode ====\n")
            results[mode], _cloud_net, _t_net = single_edge_kd(cloud_net, edge_net, params_cloud)
            continue_training_single_edge(_t_net, e_name, params_edge)
        
        elif mode == "multiteacher_kd":
            logger.info("==== Multiteacher_kd mode ====")

            # Multi edge models perform distillation to the cloud model
            results[mode], _cloud_net, _edge_nets = multi_edge_kd(current_round, cloud_net, edge_nets, params_cloud)                

            if params_edge["edge_distillation"] == 0:
                logger.info('Disabled edge distillation, do nothing here')
                # Multi edge models perform distillation to the cloud model
                # results[mode], _cloud_net, _edge_nets = multi_edge_kd(cloud_net, edge_nets, params_cloud)                
                # Teacher model continue to next round of local training
                #################################################
                # Move the follow out 
                # if params_edge["communication_round"] > 1:
                #     logger.info("continue_training_multi_edge")
                #     continue_training_multi_edge(_edge_nets, e_name, params_edge)
                #################################################

            else: 
                logger.info('Enabled edge distillation')                              
                # Return the all the edge models after performing edge distillation in case need it
                _edge_nets_after_distillation = edge_distillation(_edge_nets, _cloud_net, params_edge)
            
        else:
            print("No kd mode selected")
            exit()
    
    end = time.time()
    logger.debug(f"Time elapse: {(end - start)}")

    # # Dump the overall results
    #     for name, acc in results.items():
    #         # Save the test accuracy at the end of the training
    #         final_acc_file_name = params_edge["results_dir"].joinpath(f"{ACC_NAME}")
    #         with final_acc_file_name.open("w+") as acc_file:
    #             acc_file.write("Test accuracy\n")
    #             acc_file.write(f"{acc}\n")

def edge_distillation(_edge_nets, _cloud_net, params_edge, _edge_nets_edge_distillation):

    _edge_nets_edge_distillation = []

    # Swap the teacher and the student model's position in the kd function
    for edge_net in _edge_nets:
        edge_net = defrost_net(edge_net)
        _, edge_n, _ = single_edge_kd(edge_net, _cloud_net, params_edge)
        _edge_nets_edge_distillation.append(edge_n)
    _cloud_net = defrost_net(_cloud_net)

    return _edge_nets_edge_distillation

def start_evaluation(args):
    
    device = util.setup_torch()
    num_classes = 100 if args.dataset == "cifar100" else 10
    num_classes_distillation = 100
    
    train_loader, test_loader, train_dataset, test_dataset = get_cifar(num_classes,
                                          batch_size=args.batch_size, num_users=args.num_users)
    dict_users = cifar_iid(train_dataset, args.num_users)

    # Load distillation data
    train_loader_cifar100, test_loader_cifar100, train_dataset_cifar100, test_dataset_cifar100 = get_cifar(num_classes_distillation,
                                          batch_size=args.batch_size, num_users=args.num_users)

    # for benchmarking, decided whether we want to use unique test folders
    if USE_ID:
        # test_id = util.generate_id()
        test_id = args.workspace + '_' + timestr
        # test_id = args.workspace

    else:
        test_id = ""
    results_dir = Path(args.results_dir).joinpath(test_id)
    results_dir = Path(results_dir).joinpath(args.dataset)
    util.check_dir(results_dir)

    # Parsing arguments and prepare settings for training
    params_edge = {
        "epochs": args.epochs,
        "communication_round": args.communication_round,
        "modes": args.modes,
        "t_checkpoint": args.t_checkpoint,
        "results_dir": results_dir,
        "num_users": args.num_users,
        "nthread": args.nthread,
        "edge_distillation": args.edge_distillation,
        # "train_loader": train_loader,
        # "train_loader": train_loader_cifar100, # replace with cifar100 as distillation dataset
        "train_loader": train_loader, # used when training teacher from scratch
        "test_loader": test_loader,

        # dataset objects are used to support indexing
        # the dict_users used for indexing the training dataset
        "train_dataset": train_dataset, 
        "test_dataset": test_dataset,
        "dict_users": dict_users, 
        
        "batch_size": args.batch_size,
        # model configuration
        "device": device,
        "edge_name": args.e_name,
        "cloud_name": args.c_name,
        "num_classes": num_classes,
        # hyperparameters
        "weight_decay": args.weight_decay,
        "learning_rate": args.learning_rate,
        "momentum": args.momentum,
        "sched": args.scheduler,
        "optim": args.optimizer,
        # fixed knowledge distillation parameters
        # Change lambda_student from 0.5 to 1 to remove the impact from labels
        "lambda": 1,
        "T": 5,
    }

    # This set of parameters is for distillation 
    params_cloud = {
        "epochs": args.epochs,
        "communication_round": args.communication_round,
        "modes": args.modes,
        "t_checkpoint": args.t_checkpoint,
        "results_dir": results_dir,
        "nthread": args.nthread,
        # "train_loader": train_loader,
        "train_loader": train_loader_cifar100, # replace with cifar100 as distillation dataset
        # "train_loader": train_loader, # used when training teacher from scratch
        "test_loader": test_loader,
        # dataset objects are used to support indexing
        # the dict_users used for indexing the training dataset
        "train_dataset": train_dataset, 
        "test_dataset": test_dataset,
        "dict_users": dict_users, 

        "batch_size": args.batch_size,
        # model configuration
        "device": device,
        "teacher_name": args.e_name,
        "student_name": args.c_name,
        "num_classes": num_classes,
        # hyperparameters
        "weight_decay": args.weight_decay,
        "learning_rate": args.learning_rate,
        "momentum": args.momentum,
        "sched": args.scheduler,
        "optim": args.optimizer,
        # fixed knowledge distillation parameters
        # Change lambda_student from 0.5 to 1 to remove the impact from labels
        "lambda": 1,
        "T": 5,
    }

    test_conf_name = results_dir.joinpath("test_config.json")
    util.dump_json_config(test_conf_name, params_edge)
    run_benchmarks(args.modes, args, params_edge, params_cloud, args.c_name, args.e_name)
    
    # skip plot for now, too confusing
    # plot_results(results_dir, test_id=test_id)


if __name__ == "__main__":
    ARGS = parse_arguments()
    print(ARGS)

    # set the logging level
    levels = {
    'critical': logging.CRITICAL,
    'error': logging.ERROR,
    'warning': logging.WARNING,
    'info': logging.INFO,
    'debug': logging.DEBUG
    }

    level = levels.get(ARGS.log.lower())

    if level is None:
        raise ValueError(
            f"log level given: {ARGS.log}"
            f" -- must be one of: {' | '.join(levels.keys())}")

    logging.basicConfig(level=level, format=fmt_str)
    logger = logging.getLogger()

    start_evaluation(ARGS)