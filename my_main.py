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

BATCH_SIZE = 128
TESTFOLDER = "results"
USE_ID = True
COMMUNICATION_ROUND = 10
EPOCHS = 100


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
    parser.add_argument("--num_users", default=2,
                        type=int, help="number of users")                    
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

def setup_teacher(t_name, params, user):
    # Teacher Model
    print(f"### Setting up teacher")
    num_classes = params["num_classes"]
    t_net = create_model(t_name, num_classes, params["device"])
    print(f"t_name: {t_name}, type: {type(t_net)}")
    # exit()
    teacher_config = params.copy()
    teacher_config["test_name"] = t_name + "_teacher"
    # print(f":::::: teacher_config ::::::\n {teacher_config}")
    # print(t_net)
    
    # exit()

    dict_users = params['dict_users']
    # print(f":::: dict_users[0] length: {len(teacher_config['dict_users'][0])}")
    # teacher_trainer = BaseTrainer(t_net, config=teacher_config, idxs=dict_users[idx])

    if params["t_checkpoint"]:
        # Just validate the performance
        print("---------- Loading Teacher -------")
        # pretrained_teacher = params["t_checkpoint"]
        # stick with the same naming of best_teacher 
        best_teacher = params["t_checkpoint"]
        print(f"pretrained teacher: {best_teacher}" )
    else:
        # Teacher training
        print("---------- Training Teacher -------")
        teacher_trainer = BaseTrainer(t_net, config=teacher_config, idxs=dict_users[user])
        teacher_trainer.train()
        best_teacher = teacher_trainer.best_model_file
        print(f"best_teacher: {best_teacher}")

    # load teacher's checkpoint
    t_net = util.load_checkpoint(t_net, best_teacher) 
    teacher_trainer = BaseTrainer(t_net, config=teacher_config)
    best_t_acc = teacher_trainer.validate()
    print(f":::::: best_t_acc: ::::::\n {best_t_acc}")

    # also save this information in a csv file for plotting
    
    name = teacher_config["test_name"] + "_val"
    acc_file_name = params["results_dir"].joinpath(f"{name}.csv")
    with acc_file_name.open("w+") as acc_file:
        acc_file.write("Training Loss,Validation Loss\n")
        for _ in range(params["epochs"]):
            acc_file.write(f"0.0,{best_t_acc}\n")
        
    return t_net, best_teacher, best_t_acc
   
def continue_training_teacher(t_net, t_name, params):
    
    t_net = defrost_teacher(t_net)
    print(f"t_name: {t_name}, type: {type(t_net)}")
    teacher_config = params.copy()
    teacher_config["test_name"] = t_name + "_teacher"
    # print(f":::::: teacher_config ::::::\n {teacher_config}")
    
    teacher_trainer = BaseTrainer(t_net, config=teacher_config)
    teacher_trainer.train()
    best_teacher = teacher_trainer.best_model_file
    print(f"best_teacher: {best_teacher}")

    best_t_acc = teacher_trainer.validate()
    print(f":::::: best_t_acc: ::::::\n {best_t_acc}")

    # # also save this information in a csv file for plotting
    
    # name = teacher_config["test_name"] + "_val"
    # acc_file_name = params["results_dir"].joinpath(f"{name}.csv")
    # with acc_file_name.open("w+") as acc_file:
    #     acc_file.write("Training Loss,Validation Loss\n")
    #     for _ in range(params["epochs"]):
    #         acc_file.write(f"0.0,{best_t_acc}\n")
        
    return t_net, best_teacher, best_t_acc

def continue_training_multiteacher(t_nets, t_name, params):
    
    best_teachers = []
    best_t_accs = []
    for idx, t_net in enumerate(t_nets):
        t_net = defrost_teacher(t_net)
        print(f"t_name: {t_name}, type: {type(t_net)}")
        teacher_config = params.copy()
        teacher_config["test_name"] = t_name + "_teacher"
        # print(f":::::: teacher_config ::::::\n {teacher_config}")
        dict_users = params['dict_users']
        print(f":::: dict_users[0] length: {len(teacher_config['dict_users'][0])}")

        teacher_trainer = BaseTrainer(t_net, config=teacher_config, idxs=dict_users[idx])
        
        # for k, v in teacher_config.items():
        #     print(k, v)
        teacher_trainer.train()
        best_teacher = teacher_trainer.best_model_file
        print(f"best_teacher: {best_teacher}")

        best_t_acc = teacher_trainer.validate()
        print(f":::::: best_t_acc: ::::::\n {best_t_acc}")

        best_teachers.append(best_teacher)
        best_t_accs.append(best_t_acc)
    # # also save this information in a csv file for plotting
    
    # name = teacher_config["test_name"] + "_val"
    # acc_file_name = params["results_dir"].joinpath(f"{name}.csv")
    # with acc_file_name.open("w+") as acc_file:
    #     acc_file.write("Training Loss,Validation Loss\n")
    #     for _ in range(params["epochs"]):
    #         acc_file.write(f"0.0,{best_t_acc}\n")
        
    return t_nets, best_teachers, best_t_accs

def setup_student(s_name, params):
    # Student Model
    num_classes = params["num_classes"]
    s_net = create_model(s_name, num_classes, params["device"])
    return s_net

def freeze_teacher(t_net):
    # freeze the layers of the teacher
    for param in t_net.parameters():
        param.requires_grad = False
    # set the teacher net into evaluation mode
    t_net.eval()
    return t_net

def defrost_teacher(t_net):
    # freeze the layers of the teacher
    for param in t_net.parameters():
        param.requires_grad = True
    # set the teacher net into evaluation mode
    t_net.train()
    return t_net


def test_nokd(s_net, t_net, params):
    print("---------- Training NOKD -------")
    nokd_config = params.copy()
    nokd_trainer = BaseTrainer(s_net, config=nokd_config)
    best_acc = nokd_trainer.train()
    return best_acc

def test_kd(s_net, t_net, params):
    t_net = freeze_teacher(t_net)
    print("---------- Training KD -------")
    kd_config = params.copy()
    kd_trainer = KDTrainer(s_net, t_net=t_net, config=kd_config)
    best_acc = kd_trainer.train()
    
    return best_acc

def my_test_kd(s_net, t_net, params):
    t_net = freeze_teacher(t_net)
    print("---------- Training KD -------")
    kd_config = params.copy()
    kd_trainer = KDTrainer(s_net, t_net=t_net, config=kd_config)
    best_acc = kd_trainer.train()
    
    return best_acc, s_net, t_net

def my_test_multi_teacher_kd(s_net, t_nets, params):
    frozen_t_nets = []
    print("---------- Training MULTITEACHER -------")
    for t_net in t_nets:
        frozen_t_nets.append(freeze_teacher(t_net))
        kd_config = params.copy()

    kd_trainer = MultiTeacher(s_net, t_nets=frozen_t_nets, config=kd_config)
    best_acc = kd_trainer.train()
    return best_acc, s_net, t_nets

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
        s_net = setup_student(s_name, params_s)
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

def run_benchmarks(modes, params, params_student, s_name, t_name):
    results = {}    
    print(f"### run_benchmark ###\n")
    # print(f"params {params}\n")
    
    # Training the teacher model
    # t_net, pretrained_teacher, best_t_acc = setup_teacher(t_name, params)
    
    # Parse the mode 
    mode = modes[0]
    print(f"mode: {mode}")
    mode = mode.lower()

    # Setup teacher models
        
    t_nets = []
    print(f"params['num_users'] = {params['num_users']}")
    # exit()
    for user in range(params['num_users']):
        print(f"Setting up the {user + 1}th teacher\n")
        t_net, _, _ = setup_teacher(t_name, params, user)
        t_nets.append(t_net)
    # t_net, _, _ = setup_teacher(t_name, params)
    # t_net_1, _, _ = setup_teacher(t_name, params)
    
    # t_nets = [t_net, t_net_1]

    params_s = params_student
    print(f"params_s dict_users: {len(params_s['dict_users'])}")
    print(f"params_s dict_users[0]: {len(params_s['dict_users'][0])}")

    # Setup the student 
    s_net = setup_student(s_name, params)
    print(f"s_name: {s_name}")
    
    # Setup directories
    params_s["test_name"] = s_name
    params_s["results_dir"] = params_s["results_dir"].joinpath(mode)
    print(f"s_name: {s_name}")
    util.check_dir(params_s["results_dir"])

    for i in range(1, params["communication_round"]+1):
    # Knowledge distillation
        print(f"Starting {i} global epoch")
        # Single teacher and student mode is working
        if mode == "kd":
            print("==== Running kd mode ====\n")
            results[mode], _s_net, _t_net = my_test_kd(s_net, t_net, params_s)
            continue_training_teacher(_t_net, t_name, params)
        elif mode == "multiteacher_kd":
            print("==== Running multiteacher_kd mode ====\n")
            results[mode], _s_net, _t_nets = my_test_multi_teacher_kd(s_net, t_nets, params_s)
            continue_training_multiteacher(_t_nets, t_name, params)
        else:
            print("No kd mode selected")
            exit()
    
    # Dump the overall results
    # print(f"Best results teacher {t_name}: {best_t_acc}")
        for name, acc in results.items():
            print(f"Best results for {s_name} with {name} method: {acc}")
            # Save the test accuracy at the end of the training
            final_acc_file_name = params["results_dir"].joinpath(f"{name}_test_acc.csv")
            with final_acc_file_name.open("w+") as acc_file:
                acc_file.write("Test accuracy\n")
                acc_file.write(f"{acc}\n")
                
        

##### Commented out the support of multiple modes ######
    # for mode in modes:
    #     print(f"modes = {modes}")
    #     mode = mode.lower()
        
    #     # params_s = params.copy()
    #     params_s = params_student
    #     print(f"params_s: {params_s}")
    #     # reset the teacher
    #     # why reset the teacher again? Comment out for now
    #     # t_net = util.load_checkpoint(t_net, best_teacher, params["device"])
    
    #     # load the student and create a results directory for the mode
    #     s_net = setup_student(s_name, params)
    #     print(f"s_name: {s_name}")
    #     params_s["test_name"] = s_name
    #     params_s["results_dir"] = params_s["results_dir"].joinpath(mode)
    #     print(f"s_name: {s_name}")
    #     util.check_dir(params_s["results_dir"])
    #     # start the test
    #     try:
    #         # Haven't seen this way of calling a function before. 
    #         # It is quite convenient but not easy to read
            
    #         # run_test = globals()[f"test_{mode}"]
    #         # results[mode] = run_test(s_net, t_net, params_s)
    #         # call the function i need explicitly
    #         # results[mode] = test_kd(s_net, t_net, params_s)
    #         results[mode] = my_test_kd(s_net, t_net, params_s)
    #     except KeyError:
    #         raise RuntimeError(f"Training mode {mode} not supported!")
        
    # # Dump the overall results
    # # print(f"Best results teacher {t_name}: {best_t_acc}")
    # for name, acc in results.items():
    #     print(f"Best results for {s_name} with {name} method: {acc}")
##### Commented out the support of multiple modes ######

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
        test_id = util.generate_id()
    else:
        test_id = ""
    results_dir = Path(args.results_dir).joinpath(test_id)
    results_dir = Path(results_dir).joinpath(args.dataset)
    util.check_dir(results_dir)

    # Parsing arguments and prepare settings for training
    params = {
        "epochs": args.epochs,
        "communication_round": args.communication_round,
        "modes": args.modes,
        "t_checkpoint": args.t_checkpoint,
        "results_dir": results_dir,
        "num_users": args.num_users,
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
    params_student = {
        "epochs": args.epochs,
        "modes": args.modes,
        "t_checkpoint": args.t_checkpoint,
        "results_dir": results_dir,
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
    util.dump_json_config(test_conf_name, params)
    run_benchmarks(args.modes, params, params_student, args.s_name, args.t_name)
    plot_results(results_dir, test_id=test_id)


if __name__ == "__main__":
    ARGS = parse_arguments()
    print(ARGS)
    start_evaluation(ARGS)