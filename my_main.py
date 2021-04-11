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


BATCH_SIZE = 128
TESTFOLDER = "results"
USE_ID = True
COMMUNICATION_ROUND = 10
EPOCHS = 100
NTHREAD = 4
ACC_NAME = 'cloud_model_test_acc.csv'

fmt_str = '[%(levelname)s] %(message)s'

logging.basicConfig(level=logging.DEBUG, format=fmt_str)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# logger.debug(f'hello logged world')
# logger.info(f'FYI')


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Test parameters")
    parser.add_argument("--epochs", default=EPOCHS, type=int,
                        help="number of total epochs to run")
    parser.add_argument("--communication_round", default=COMMUNICATION_ROUND, type=int,
                        help="number of total global epochs to run")
    parser.add_argument("--dataset", default="cifar100", type=str,
                        help="dataset. can be either cifar10 or cifar100")
    parser.add_argument("--batch-size", default=BATCH_SIZE,
                        type=int, help="batch_size")
    parser.add_argument("--nthread", default=NTHREAD,
                        type=int, help="number of threads for dataloader")                    
    parser.add_argument('--gpu_id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--ngpu', default=1, type=int,
                    help='number of GPUs to use for training')
    
    parser.add_argument("--num_users", default=2,
                        type=int, help="number of users") 
    parser.add_argument("--workspace", default="",
                        type=str, help="The prefix for the output files")                    
    parser.add_argument("--edge_distillation", default=1,
                        type=int, help="enable edge distillation")
    parser.add_argument("--learning-rate", default=0.1,
                        type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9,
                        type=float, help="SGD momentum")
    parser.add_argument("--weight-decay", default=5e-4,
                        type=float, help="SGD weight decay (default: 5e-4)")
    parser.add_argument("--teacher", default="WRN22_4", type=str,
                        dest="t_name", help="teacher student name")
    parser.add_argument("--student", "--model", default="resnet18",
                        dest="s_name", type=str, help="teacher student name")
    parser.add_argument("--optimizer", default="sgd",
                        dest="optimizer", type=str,
                        help="Which optimizer to use")
    parser.add_argument("--scheduler", default="multisteplr",
                        dest="scheduler", type=str,
                        help="Which scheduler to use")
    parser.add_argument("--teacher-checkpoint", default="",
                        dest="t_checkpoint", type=str,
                        help="optional pretrained checkpoint for teacher")
    parser.add_argument("--mode", default=["KD"], dest="modes",
                        type=str, nargs='+',
                        help="What type of distillation to use")
    parser.add_argument("--results-dir", default=TESTFOLDER,
                        dest="results_dir", type=str,
                        help="Where all results are collected")
    args = parser.parse_args()
    return args

def setup_edge(edge_name, edge_params, user, edge_suffix):
    # Teacher Model
    logger.info("Setting up edge model")
    num_classes = edge_params["num_classes"]
    t_net = create_model(edge_name, num_classes, edge_params["device"])
    # print(f"t_name: {t_name}, type: {type(t_net)}")
    edge_config = edge_params.copy()

    edge_config["test_name"] = edge_name + edge_suffix
    

    dict_users = edge_params['dict_users']

    if edge_params["t_checkpoint"]:
        # Just validate the performance
        # print("---------- Loading Edge Model -------")
        logger.info('+'*20 + " Loading Edge Model " + '+'*20)
        best_teacher = edge_params["t_checkpoint"]
        logger.debug(f"pretrained edge model: {best_teacher}" )
    else:
        # Teacher training
        # print("---------- Training Edge Model -------")
        logger.info('-'*20 + ' Training Edge Model ' + '-'*20)
        teacher_trainer = BaseTrainer(t_net, config=edge_config, idxs=dict_users[user])
        teacher_trainer.train()
        best_teacher = teacher_trainer.best_model_file
        print(f"best_teacher: {best_teacher}")

    # load teacher's checkpoint
    t_net = util.load_checkpoint(t_net, best_teacher) 
    teacher_trainer = BaseTrainer(t_net, config=edge_config)
    
    best_t_acc = teacher_trainer.validate()
    
    # print(f":::::: best_t_acc: ::::::\n {best_t_acc}")

    # also save this information in a csv file for plotting
    name = edge_config["test_name"] + "_val"
    ## resnet8_test is the "test_name"

    acc_file_name = edge_params["results_dir"].joinpath(f"{name}.csv")
    with acc_file_name.open("w+") as acc_file:
        acc_file.write("Training Loss,Validation Loss\n")
        for _ in range(edge_params["epochs"]):
            acc_file.write(f"0.0,{best_t_acc}\n")
        
    return t_net, best_teacher, best_t_acc
   
def continue_training_single_edge(t_net, t_name, params):
    
    t_net = defrost_net(t_net)
    # print(f"t_name: {t_name}, type: {type(t_net)}")
    teacher_config = params.copy()
    teacher_config["test_name"] = t_name + "_teacher"
    
    teacher_trainer = BaseTrainer(t_net, config=teacher_config)
    teacher_trainer.train()
    best_teacher = teacher_trainer.best_model_file
    print(f"best_teacher: {best_teacher}")

    best_t_acc = teacher_trainer.validate()
    # print(f":::::: best_t_acc: ::::::\n {best_t_acc}")

    return t_net, best_teacher, best_t_acc

def continue_training_multi_edge(t_nets, t_name, params):
    
    best_teachers = []
    best_t_accs = []
    for idx, t_net in enumerate(t_nets):
        t_net = defrost_net(t_net)
        print(f"t_name: {t_name}, type: {type(t_net)}")
        teacher_config = params.copy()
        teacher_config["test_name"] = t_name + "_teacher"
        dict_users = params['dict_users']
        teacher_trainer = BaseTrainer(t_net, config=teacher_config, idxs=dict_users[idx])
        
        teacher_trainer.train()
        best_teacher = teacher_trainer.best_model_file
        print(f"best_teacher: {best_teacher}")

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
    # set the teacher net into evaluation mode
    net.train()
    return net

def test_kd(s_net, t_net, params):
    t_net = freeze_net(t_net)
    print("---------- Training KD -------")
    kd_config = params.copy()
    kd_trainer = KDTrainer(s_net, t_net=t_net, config=kd_config)
    best_acc = kd_trainer.train()
    
    return best_acc

def single_edge_kd(s_net, t_net, params):
    t_net = freeze_net(t_net)
    print("---------- KD: single edge model -> cloud model -------")
    kd_config = params.copy()
    kd_trainer = KDTrainer(s_net, t_net=t_net, config=kd_config)
    best_acc = kd_trainer.train()
    
    return best_acc, s_net, t_net

def multi_edge_kd(cloud_net, edge_nets, params):
    # Use CIFAR-100 unlabeled data for KD with multiple edge models
    print("---------- KD: multiple edge models -> cloud model -------")
    frozen_edge_nets = []
    for t_net in edge_nets:
        frozen_edge_nets.append(freeze_net(t_net))
        kd_config = params.copy()

    kd_trainer = MultiTeacher(cloud_net, t_nets=frozen_edge_nets, config=kd_config)
    best_acc = kd_trainer.train()
    return best_acc, cloud_net, edge_nets

def test_kdparam(s_net, t_net, params):
    temps = [1, 5, 10, 15, 20]
    alphas = [0.1, 0.4, 0.5, 0.7, 1.0]
    param_pairs = [(a, T) for T in temps for a in alphas]
    accs = {}

    for alpha, T, in param_pairs:
        params_s = params.copy()
        params_s["lambda_student"] = alpha
        params_s["T_student"]: T
        s_name = params_s["student_name"]
        s_net = setup_cloud(s_name, params_s)
        params_s["test_name"] = f"{s_name}_{T}_{alpha}"
        print(f"Testing {s_name} with alpha {alpha} and T {T}.")
        best_acc = test_kd(s_net, t_net, params_s)
        accs[params_s["test_name"]] = (alpha, T, best_acc)

    best_kdparam_acc = 0
    for test_name, acc in accs.items():
        alpha = acc[0]
        T = acc[1]
        kd_acc = acc[2]
        if acc[2] > best_kdparam_acc:
            best_kdparam_acc = acc[2]
        print(f"Best results for {s_name} with a {alpha} and T {T}: {kd_acc}")

    return best_kdparam_acc

def run_benchmarks(modes, params_edge, params_cloud, s_name, t_name):
    results = {}    
   
    # Parse the mode 
    mode = modes[0]
    # print(f"mode: {mode}")
    mode = mode.lower()

    # Setup teacher models
    edge_nets = []
    logger.debug(f"params['num_users'] = {params_edge['num_users']}")

    # Step 1: Teacher model initial training with local data
    for user in range(params_edge['num_users']):
        print(f"Setting up the {user + 1}th edge\n")
        edge_net, _, _ = setup_edge(t_name, params_edge, user, edge_suffix='_edge')
        edge_nets.append(edge_net)

    params_c = params_cloud
    print(f"params_s dict_users: {len(params_c['dict_users'])}")
    print(f"params_s dict_users[0]: {len(params_c['dict_users'][0])}")

    # Setup the cloud model 
    cloud_net = setup_cloud(s_name, params_edge)
    
    # Setup directories
    params_c["test_name"] = s_name
    # params_s["results_dir"] = params_s["results_dir"].joinpath(mode)

    params_c["results_dir"] = params_c["results_dir"].joinpath('cloud_model')
    
    # pdb.set_trace()

    util.check_dir(params_c["results_dir"])

    params_edge["test_name"] = s_name

    for i in range(1, params_edge["communication_round"]+1):
    # Knowledge distillation
        print(f"Starting {i} communication_round")
        
        if mode == "kd":
            print("==== Running kd mode ====\n")
            results[mode], _cloud_net, _t_net = single_edge_kd(cloud_net, edge_net, params_c)
            continue_training_single_edge(_t_net, t_name, params_edge)
        
        elif mode == "multiteacher_kd":
            print("==== Multiteacher_kd mode ====")
            
            if params_edge["edge_distillation"] == 1:
                logger.info('Enabled edge distillation')
                
                # Step 2: edge -> cloud distillation
                results[mode], _cloud_net, _edge_nets = multi_edge_kd(cloud_net, edge_nets, params_c)
                
                # Step 1: Teacher model continue to local training
                # continue_training_multiteacher(_t_nets, t_name, params)
                
                # Step 3: student -> teacher distillation on each teacher model
                # swap the position of the teacher and the student
                _edge_nets_edge_distillation = []
                for edge_net in _edge_nets:
                    edge_net = defrost_net(edge_net)
                    _, edge_n, _ = single_edge_kd(edge_net, _cloud_net, params_edge)
                    _edge_nets_edge_distillation.append(edge_n)
                _cloud_net = defrost_net(_cloud_net)
                # Step 1: Teacher model continue to local training
                continue_training_multi_edge(_edge_nets_edge_distillation, t_name, params_edge)
            
            else:
                logger.info('Disabled edge distillation')
                # Step 2: teacher -> student distillation
                results[mode], _cloud_net, _edge_nets = multi_edge_kd(cloud_net, edge_nets, params_c)                
               
                # Step 1: Teacher model continue to next round of local training
                # TODO: the following part does not look correct 
                # continue to train is called more than once when communication round is >=1
                #################################################
                if params_edge["communication_round"] > 1:
                    logger.info("continue_training_multi_edge")
                    continue_training_multi_edge(_edge_nets, t_name, params_edge)
                #################################################
        else:
            print("No kd mode selected")
            exit()
    
    # Dump the overall results
        for name, acc in results.items():
            # Save the test accuracy at the end of the training
            final_acc_file_name = params_edge["results_dir"].joinpath(f"{ACC_NAME}")
            with final_acc_file_name.open("w+") as acc_file:
                acc_file.write("Test accuracy\n")
                acc_file.write(f"{acc}\n")

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
        test_id = args.workspace

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
        "teacher_name": args.t_name,
        "student_name": args.s_name,
        "num_classes": num_classes,
        # hyperparameters
        "weight_decay": args.weight_decay,
        "learning_rate": args.learning_rate,
        "momentum": args.momentum,
        "sched": args.scheduler,
        "optim": args.optimizer,
        # fixed knowledge distillation parameters
        # Change lambda_student from 0.5 to 1 to remove the impact from labels
        "lambda_student": 1,
        "T_student": 5,
    }

    # This set of parameters is for distillation 
    params_cloud = {
        "epochs": args.epochs,
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
        "teacher_name": args.t_name,
        "student_name": args.s_name,
        "num_classes": num_classes,
        # hyperparameters
        "weight_decay": args.weight_decay,
        "learning_rate": args.learning_rate,
        "momentum": args.momentum,
        "sched": args.scheduler,
        "optim": args.optimizer,
        # fixed knowledge distillation parameters
        # Change lambda_student from 0.5 to 1 to remove the impact from labels
        "lambda_student": 1,
        "T_student": 5,
    }

    test_conf_name = results_dir.joinpath("test_config.json")
    util.dump_json_config(test_conf_name, params_edge)
    run_benchmarks(args.modes, params_edge, params_cloud, args.s_name, args.t_name)
    
    # skip plot for now, too confusing
    # plot_results(results_dir, test_id=test_id)


if __name__ == "__main__":
    ARGS = parse_arguments()
    print(ARGS)
    start_evaluation(ARGS)