import argparse
from pathlib import Path

# from distillers import *
from data_loader import *
from models.model_factory import create_model

from trainer import BaseTrainer, KDTrainer, MultiTeacher, KDTrainerNoLabel
# from plot import plot_results
import util
import torch.nn as nn
import torch
from sampling import cifar_iid
import psutil
import os
import pdb
import logging
import time
import numpy as np
from utils.timer import Timer
from aggregate_method.Fed import FedAvg, cosine_annealing
import copy

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
    parser.add_argument("--epochs", default=EPOCHS, type=int, help="number of total epochs to run on edge")
    parser.add_argument("--cloud_epochs", default=EPOCHS, type=int, help="number of epochs epochs to run on cloud")
    parser.add_argument("--num_users", default=2, type=int, help="number of users") 
    parser.add_argument("--communication_round", default=COMMUNICATION_ROUND, type=int, help="number of total global epochs to run")
    parser.add_argument('--frac', default=0.1, type=float, help="the fraction of clients: C")

    parser.add_argument('--random_seed', default=2, type=int)
    parser.add_argument('--alpha', default=100, type=int, help="alpha for dirichlet distribution")
    
    parser.add_argument("--batch-size", default=BATCH_SIZE, type=int, help="batch_size")

    
    # model arguments
    parser.add_argument("--dataset", default="cifar10", type=str, help="dataset. can be either cifar10 or cifar100")
    parser.add_argument("--edge_update", default="edge_distillation", type=str, help="copy, edge_distillation, and no_update")
    parser.add_argument("--ed_nolabel", action='store_true', help="remove label in edge distillation")
    parser.add_argument("--temperature", default=1, type=float, help="temperature for distillation")    
    parser.add_argument("--edge", default="resnet8", type=str, dest="e_name", help="edge model name")
    parser.add_argument("--cloud", default="resnet8", type=str, dest="c_name", help="cloud model name")
    parser.add_argument("--optimizer", default="sgd", dest="optimizer", type=str, help="Which optimizer to use")
    parser.add_argument("--cloud_optimizer", default="adam", type=str, help="Which optimizer to use")

    
    # learning rate scheme
    parser.add_argument("--learning_rate", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD momentum")
    parser.add_argument("--weight-decay", default=5e-4, type=float, help="SGD weight decay (default: 5e-4)")
    parser.add_argument("--cloud_temperature", default=1, type=float, help="temperature for distillation")
    parser.add_argument("--cloud_learning_rate", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--cloud_momentum", default=0.9, type=float, help="SGD momentum")
    parser.add_argument("--cloud_weight_decay", default=5e-4, type=float, help="SGD weight decay (default: 5e-4)")
    parser.add_argument("--no_nesterov", action='store_true',help="disable nesterov on edge")
    parser.add_argument("--adam_beta_1", default=0.9, type=float)
    parser.add_argument("--adam_beta_2", default=0.999, type=float)
    parser.add_argument("--adam_eps", default=1e-8, type=float)

    # other arguments
    parser.add_argument('--gpu_id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--ngpu', default=1, type=int, help='number of GPUs to use for training')
    parser.add_argument("--scheduler", default="constant", dest="scheduler", type=str, help="Which scheduler to use")
    parser.add_argument("--cloud_scheduler", default="constant", dest="cloud_scheduler", type=str, help="Which scheduler to use for cloud")
    parser.add_argument("--teacher-checkpoint", default="", dest="t_checkpoint", type=str,
                        help="optional pretrained checkpoint for teacher")
    parser.add_argument("--mode", default=["FedAvg"], dest="modes", type=str, nargs='+', help="What type of distillation to use")
    parser.add_argument("--nthread", default=NTHREAD, type=int, help="number of threads for dataloader")                    
    parser.add_argument("--workspace", default="", type=str, help="The prefix for the output files")                    
    parser.add_argument("--results-dir", default=TESTFOLDER, type=str, help="Where all results are collected")
    parser.add_argument("--log", default="info", help="Provide logging level")

    args = parser.parse_args()

    return args

@Timer(text='setup_edge in {:.4f} seconds')
def setup_edge(edge_name, edge_params):
    # Teacher Model
    logger.debug("Setting up edge model")
    num_classes = edge_params["num_classes"]

    # t_net is the model itself
    t_net = create_model(edge_name, num_classes, edge_params["device"])
    
    return t_net

@Timer(text='train_edge in {:.4f} seconds')
def train_edge(current_round, edge_net, user_id, edge_params, edge_name, partition_idx, edge_suffix):
    
    # logger.info("Training edge model")
    # defrost because from the seoncd round the edge models are frozen during distillation
    edge_net = defrost_net(edge_net)
    edge_config = edge_params.copy()
    edge_config["test_name"] = edge_name + edge_suffix
    dict_users = edge_params['dict_users']
    # pdb.set_trace()

    # Trainer wraps the training related functions together
    logger.debug(f"The partition idx is {partition_idx}")
    # edge_trainer = BaseTrainer(edge_net, config=edge_config, idxs=dict_users[partition_idx])
    edge_trainer = BaseTrainer(edge_net, config=edge_config, user_id=user_id)
    edge_trainer.train(current_round)
    edge_ckpt = edge_trainer.model_file

    return edge_net, edge_ckpt, edge_config

@Timer(text='setup_cloud in {:.4f} seconds')
def setup_cloud(s_name, params):
    # Student Model
    num_classes = params["num_classes"]
    s_net = create_model(s_name, num_classes, params["device"])
    return s_net

@Timer(text='free_net in {:.4f} seconds')
def freeze_net(net):
    # freeze the layers of a model
    for param in net.parameters():
        param.requires_grad = False
    # set the teacher net into evaluation mode
    net.eval()
    return net

@Timer(text='defrost_net in {:.4f} seconds')
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

@Timer(text='multi_edge_kd in {:.4f} seconds')
def multi_edge_kd(current_round, cloud_net, edge_nets, params):
    # Use CIFAR-100 unlabeled data for KD with multiple edge models
    logger.info("---------- KD: multiple edge models teaching cloud model -------")
    cloud_net = defrost_net(cloud_net)

    frozen_edge_nets = []
    for e_net in edge_nets:
        frozen_edge_nets.append(freeze_net(e_net))
    
    kd_config = params.copy()
    # cloud_lr = kd_config["cloud_learning_rate"]
    # logger.debug(f"Cloud lr: {cloud_lr}")
    kd_trainer = MultiTeacher(cloud_net, t_nets=frozen_edge_nets, config=kd_config)
    cloud_acc = kd_trainer.train(current_round)
    cloud_ckpt = kd_trainer.model_file
    return cloud_acc, cloud_ckpt, cloud_net, edge_nets

def dummy_trainer(current_round, cloud_net, params):
    # Calculates the test accuracy of the aggregated model
    logger.info("---------- Testing aggregated model -------")
    
    kd_config = params.copy()
    cloud_trainer = BaseTrainer(cloud_net, config=kd_config, idxs=None)
    cloud_trainer.get_accuracy(current_round)


@Timer(text='edge_distillation in {:.4f} seconds')
def edge_distillation(current_round, edge_net, cloud_net, params_edge, edge_name, partition_idx, edge_suffix):

    edge_net = defrost_net(edge_net)
    # logger.info("---------- Edge distillation -------")
    cloud_net = freeze_net(cloud_net)
    kd_config = params_edge.copy()
    
    # Save the results after distillation separately for a better comparison
    kd_config["test_name"] = edge_name + edge_suffix
    dict_users = kd_config['dict_users']
    logger.debug(f"The partition idx is {partition_idx}")

    if params_edge["ed_nolabel"]:
        logger.debug("Edge distillation without labels")
        kd_trainer = KDTrainerNoLabel(s_net=edge_net, t_net=cloud_net, config=kd_config, idxs=dict_users[partition_idx])
    else:
        kd_trainer = KDTrainer(s_net=edge_net, t_net=cloud_net, config=kd_config, idxs=dict_users[partition_idx])

    edge_acc = kd_trainer.train(current_round)

    return edge_net

def copy_weights(net):

    weights = net.state_dict()
    return weights

def load_checkpoint(model, ckpt_path):

    model.load_state_dict(torch.load(ckpt_path))
    return model

def print_model_parameters(model):
    for name, param in model.named_parameters():
        logger.debug(f'name: {name}')
        logger.debug(type(param))
        logger.debug(f'param.shape: {param.shape} ')
        logger.debug(f'param.requires_grad: {param.requires_grad}')
        logger.debug(f'=====')

def run_benchmarks(modes, args, params_edge, params_cloud, c_name, e_name):

    t0 = time.time()

    # Extract the mode
    mode = modes[0]
    mode = mode.lower()
    cloud_acc = {}

    # Setup the edge model
    edge_nets = []
    logger.debug(f"params['num_users'] = {params_edge['num_users']}")
    m = max(int(args.frac * args.num_users), 1)
    idxs_users = np.random.choice(range(args.num_users), m, replace=False)
    
    logger.debug(f"idx_users: {idxs_users}")
    
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
    
    # Loop through the communication rounds
    start = time.time()
    # t1 = start
    logger.info(f"Prep time: {start - t0}")
    for i in range(0, params_edge["communication_round"]):
        t1 = time.time()
        logger.info(f"Starting {i} communication_round")
        
        # Save every round of data in separate directories
        current_round = i
        params_cloud["results_dir"] = base_path.joinpath(str(i)).joinpath('cloud_model')
        params_edge["results_dir"] = base_path.joinpath(str(i))

        # cloud_path = cloud_path.joinpath('cloud_model')
        util.check_dir(params_cloud["results_dir"])
        
        # Train all the edge models
        
        # skip edge for now
        for user, partition_idx in enumerate (idxs_users):
            logger.info(f"Training {user + 1}th edge model")
            # TODO: The edge_config here may need to change for heterogeneous model case
            edge_net, _, edge_config = train_edge(current_round, edge_nets[user], user, params_edge, e_name, 
                                                            partition_idx, edge_suffix='_edge_'+ str(user))
        

        # We select how to aggregate the edge models
        if mode == "feddf":
            logger.info("==== FedDF mode ====")

            # Update learning rate 
            # def cosine_annealing(epoch, n_epochs, n_cycles, lrate_max):
            # cloud_learning_rate = cosine_annealing()

            # Multi edge models perform distillation to the cloud model
            
            cloud_acc[mode], cloud_ckpt, trained_cloud_net, _edge_nets = multi_edge_kd(current_round, cloud_net, 
                                                                    edge_nets, params_cloud)                

            if params_edge["edge_update"] == "no_update": 
                # do nothing here
                logger.info('No edge update, do nothing here')

            elif params_edge["edge_update"] == "edge_distillation": 
                # use edge distillation here
                logger.info('Enabled edge distillation')                              
                # Return the all the edge models after performing edge distillation in case need it
                # edge_config instead of edge_params is used here becase the test_name field is added to the edge_config only
        
                for user, partition_idx in enumerate (idxs_users):
                    logger.info(f"Edge distillation {user + 1}th edge model")
                    # edge_net, _, edge_config = train_edge(current_round, edge_nets[user], params_edge, e_name, 
                    #                                             partition_idx, edge_suffix='_edge_'+ str(user))    
                    _edge_nets_after_distillation = edge_distillation(current_round, edge_nets[user], trained_cloud_net, 
                                                                            edge_config, e_name, partition_idx, edge_suffix='_edge_ds_'+ str(user))

            elif params_edge["edge_update"] == "copy":
                assert c_name == e_name
                # directly copy the cloud model
                # assert the cloud and edge model arch first 
                
                # print_model_parameters(trained_cloud_net)
                # try loading the cloud model instead
                # pdb.set_trace()
                logger.debug(f"Loading cloud model: {cloud_ckpt}")
                cloud_loaded = util.load_checkpoint(cloud_net, cloud_ckpt)
                cloud_weights = cloud_loaded.state_dict()
                cloud_weights_copy = copy.deepcopy(cloud_weights)

                logger.debug(f"idxs_users: {idxs_users}")
                for user, partition_idx in enumerate (idxs_users):
                    logger.info(f"Copying cloud model to the {user + 1}th edge model")                   
                    edge_nets[user].load_state_dict(cloud_weights_copy)

            else:
                logger.info("Please provide edge update method")
                exit()

            # round_time = int((time.time()-t1))
            # # logger.debug(f"round time: {round_time}")
            # remaining_rounds = params_edge['communication_round'] - current_round - 1
            # #  /current_round*(params_edge['communication_round']-current_round))
            # remaing_time = remaining_rounds * round_time
            # logger.info(f"Remaining Time (approx.): {remaing_time // 3600 :02d}:{remaing_time % 3600 // 60 :02d}:{remaing_time % 60 :02d}")
            
        elif mode == "fedavg":
            # need to merge fedavg to here
            logger.info("==== FedAvg mode ====")
            w_edge_models = []
            # create placeholder for all the edge weights
            all_edge_weights = [None for user in idxs_users]
            
            # Perform FedAvg            
            for user, partition_idx in enumerate (idxs_users):
                w = edge_nets[user].state_dict()
                all_edge_weights[user] = copy.deepcopy(w)               
            averaged_weights = FedAvg(all_edge_weights)

            if params_edge["edge_update"] == "no_update": 
                # do nothing here
                logger.info('No edge update, do nothing here')
                # Test the aggregated model
                cloud_net.load_state_dict(averaged_weights)
                dummy_trainer(current_round, cloud_net, params_cloud)

            elif params_edge["edge_update"] == "copy":
                
                # Test the aggregated model
                cloud_net.load_state_dict(averaged_weights)
                # pdb.set_trace()
                dummy_trainer(current_round, cloud_net, params_cloud)

                assert c_name == e_name
                # directly copy from the averaged weights produced by FedAvg
                # assert the cloud and edge model arch first 
                # for user in idxs_users:
                for user, partition_idx in enumerate (idxs_users):

                    logger.info(f"Copying cloud model to the {user + 1}th edge model")
                    edge_nets[user].load_state_dict(averaged_weights)
            else:
                logger.info("Please provide edge update method for FedAVG (copy or no_update")
                exit()

        else:
            print("No kd mode selected")
            exit()
        
        # Calculate and the end of each round
        round_time = int((time.time()-t1))
        logger.debug(f"round time: {round_time}")
        remaining_rounds = params_edge['communication_round'] - current_round - 1
        #  /current_round*(params_edge['communication_round']-current_round))
        remaing_time = remaining_rounds * round_time
        logger.info(f"Remaining Time (approx.): {remaing_time // 3600 :02d}:{remaing_time % 3600 // 60 :02d}:{remaing_time % 60 :02d}")

    end = time.time()
    logger.debug(f"Time elapse: {(end - start)}")
    results_dir_path = params_edge["results_dir"]
    logger.info(f"The results are saved to {results_dir_path}")


def start_evaluation(args):
    
    device = util.setup_torch(args.gpu_id)
    num_classes = 100 if args.dataset == "cifar100" else 10
    num_classes_distillation = 100

    # Save 5% data for validation in the cloud
    # train_loader, test_loader, valid_loader, train_dataset, test_dataset = get_cifar(num_classes,
    #                                       batch_size=args.batch_size, split=0.05)

    _, _, _, train_dataset, test_dataset = get_cifar(num_classes, batch_size=args.batch_size, split=0.05)
    train_set, val_set = split_train_data(train_dataset, split=0.05)

    ################# non-iid distribution #################

    client_loaders, val_loader, test_loader = get_loaders(train_set, 
                                                train_dataset.targets, 
                                                val_set,
                                                test_dataset, 
                                                n_clients=args.num_users, 
                                                alpha=args.alpha, 
                                                batch_size=args.batch_size, 
                                                n_data=None, 
                                                num_workers=args.nthread, 
                                                seed=args.random_seed, 
                                                split=0.05)

    # pdb.set_trace()
    # client_loaders, test_loader = get_loaders(train_dataset, test_dataset, n_clients=args.num_users, 
    #     alpha=args.alpha, batch_size=args.batch_size, n_data=None, num_workers=args.nthread, seed=args.random_seed)
    #########################################################

    dict_users = cifar_iid(train_dataset, args.num_users)

    # Load distillation data
    train_loader_cifar100, test_loader_cifar100, valid_loader_cifar100, train_dataset_cifar100, test_dataset_cifar100 = get_cifar(num_classes_distillation,
                                          batch_size=args.batch_size, split=0.2)

    # for benchmarking, decided whether we want to use unique test folders
    if USE_ID:
        # test_id = util.generate_id()
        # test_id = args.workspace + '_' + timestr

        # test_id = f"{args.workspace}_n_users_{args.num_users}_eps_{args.epochs}_cloud_eps_{args.cloud_epochs}_comm_{args.communication_round}_e_{args.e_name}_c_{args.c_name}_frac_{args.frac}_mode_{args.modes[0]}_{args.edge_update}_lrsched_{args.scheduler}_bs_{args.batch_size}_cloud_lr_{args.cloud_learning_rate}" + f"_{timestr}"  

        test_id = (f"{args.workspace}_n_users_{args.num_users}"
                    f"_eps_{args.epochs}"
                    f"_cloud_eps_{args.cloud_epochs}"
                    f"_comm_{args.communication_round}"
                    f"_e_{args.e_name}_c_{args.c_name}"
                    f"_frac_{args.frac}"
                    f"_mode_{args.modes[0]}"
                    f"_{args.edge_update}"
                    f"_lrsched_{args.scheduler}"
                    f"_cloud_lrsched_{args.cloud_scheduler}"
                    f"_bs_{args.batch_size}"
                    f"_cloud_lr_{args.cloud_learning_rate}"
                    f"_{timestr}"  
                )

    else:
        test_id = ""
    results_dir = Path(args.results_dir).joinpath(test_id)
    results_dir = Path(results_dir).joinpath(args.dataset)
    util.check_dir(results_dir)

    # Parsing arguments and prepare settings for training
    params_edge = {
        "epochs": args.epochs,
        "communication_round": args.communication_round,
        "frac": args.frac,
        "modes": args.modes,
        "t_checkpoint": args.t_checkpoint,
        "results_dir": results_dir,
        "num_users": args.num_users,
        "nthread": args.nthread,
        # "edge_distillation": args.edge_distillation,
        "edge_update": args.edge_update,
        "ed_nolabel": args.ed_nolabel,
        "model_type": "edge",
        # "train_loader": train_loader,
        # "train_loader": train_loader_cifar100, # replace with cifar100 as distillation dataset
        
        # No need of train_loader, use dataset to create train loader for each edge model
        # "train_loader": train_loader, # used when training teacher from scratch
        "test_loader": test_loader,
        # dataset objects are used to support indexing
        
        # client loaders
        "client_loaders": client_loaders,
        "train_dataset": train_dataset, 
        # "test_dataset": test_dataset,
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
        "no_nesterov": args.no_nesterov,
        # fixed knowledge distillation parameters
        # Change lambda_student from 0.5 to 1 to remove the impact from labels
        "lambda": 1,
        "T": args.temperature,

    }

    # This set of parameters is for distillation 
    params_cloud = {
        "epochs": args.cloud_epochs,
        "communication_round": args.communication_round,
        "modes": args.modes,
        "t_checkpoint": args.t_checkpoint,
        "results_dir": results_dir,
        "nthread": args.nthread,
        # "train_loader": train_loader,
        "train_loader": train_loader_cifar100, # replace with cifar100 as distillation dataset
        "valid_loader": valid_loader_cifar100,
        "test_loader": test_loader,
        "model_type": "cloud",        
        "batch_size": args.batch_size,
        # model configuration
        "device": device,
        "teacher_name": args.e_name,
        "student_name": args.c_name,
        "num_classes": num_classes,
        # hyperparameters
        # "weight_decay": args.weight_decay,
        # "momentum": args.momentum,
        # "sched": args.scheduler,
        "optim": args.cloud_optimizer,
        "cloud_learning_rate": args.cloud_learning_rate,
        "cloud_weight_decay": args.cloud_weight_decay,
        "cloud_momentum": args.cloud_momentum,
        "cloud_sched": args.cloud_scheduler,

        # fixed knowledge distillation parameters
        # Change lambda_student from 0.5 to 1 to remove the impact from labels
        "lambda": 1,
        "T": args.cloud_temperature,

    }

    test_conf_name = results_dir.joinpath("test_config.json")
    util.dump_json_config(test_conf_name, params_edge)
    test_conf_name = results_dir.joinpath("cloud_test_config.json")
    util.dump_json_config(test_conf_name, params_cloud)
    run_benchmarks(args.modes, args, params_edge, params_cloud, args.c_name, args.e_name)
    
    # skip plot for now, too confusing when there are multiple edge models
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