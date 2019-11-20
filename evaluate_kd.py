import argparse
import copy
import torch

from data_loader import get_cifar
from models.model_factory import create_cnn_model
from ta_distiller import run_teacher_assistant
from ab_distiller import run_ab_distillation
from rkd_distiller import run_relational_kd
from trainer import load_checkpoint, BaseTrainer, KDTrainer
from optimizer import get_optimizer, get_scheduler


BATCH_SIZE = 128

MODES = ["KD", "RKD", "AB", "TAKD"]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Knowledge Distillation Params")
    parser.add_argument("--epochs", default=200, type=int,
                        help="number of total epochs to run")
    parser.add_argument("--dataset", default="cifar100", type=str,
                        help="dataset. can be either cifar10 or cifar100")
    parser.add_argument("--batch-size", default=BATCH_SIZE,
                        type=int, help="batch_size")
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
    parser.add_argument("--teacher-checkpoint", default="",
                        dest="t_checkpoint", type=str,
                        help="optional pretrained checkpoint for teacher")
    parser.add_argument("--mode", default="KD", choices=MODES,
                        dest="mode", type=str,
                        help="What type of distillation to use")
    args = parser.parse_args()
    return args


def init_teacher(t_name, params):
    # Teacher Model
    num_classes = params["num_classes"]
    # Teacher training
    t_net = create_cnn_model(t_name, num_classes, params["device"])
    teacher_train_config = copy.deepcopy(params)
    teacher_name = params["t_name"]
    trial_id = params["trial_id"]
    best_teacher = f"{teacher_name}_{trial_id}_best.pth.tar"
    teacher_train_config["name"] = teacher_name

    if params["t_checkpoint"]:
        print("---------- Loading Teacher -------")
        t_net = load_checkpoint(t_net, params["t_checkpoint"])

    teacher_trainer = BaseTrainer(t_net, train_config=teacher_train_config)

    if params["t_checkpoint"]:
        best_t_acc = teacher_trainer.validate()
    else:
        print("---------- Training Teacher -------")
        best_t_acc = teacher_trainer.train()
        t_net = load_checkpoint(t_net, best_teacher)
    return t_net, best_t_acc


def init_student(s_net, params):
    # Student Model
    num_classes = params["num_classes"]
    s_name = params["s_name"]
    s_net = create_cnn_model(s_name, num_classes, params["device"])
    return s_net


def test_kd(s_net, t_net, params):
    print("---------- Training KD -------")
    kd_train_config = copy.deepcopy(params)
    kd_train_config["name"] = params["s_name"]
    kd_trainer = KDTrainer(s_net, t_net=t_net, train_config=kd_train_config)
    best_kd_acc = kd_trainer.train()
    return best_kd_acc


def test_ta(s_net, t_net, params):
    num_classes = params["num_classes"]
    # Arguments specifically for the teacher assistant approach
    params["ta_name"] = "resnet8"
    ta_model = create_cnn_model(
        params["ta_name"], num_classes, params["device"])
    best_ta_acc = run_teacher_assistant(s_net, ta_model, t_net, **params)
    return best_ta_acc


def test_ab(s_net, t_net, params):
    # Arguments specifically for the ab approach
    best_ab_acc = run_ab_distillation(s_net, t_net, **params)
    return best_ab_acc


def test_rkd(s_net, t_net, params):
    # Arguments specifically for the ab approach
    best_rkd_acc = run_relational_kd(s_net, t_net, **params)
    return best_rkd_acc


def run_benchmarks(mode, params):
    t_name = params["t_name"]
    s_name = params["s_name"]
    t_net, best_t_acc = init_teacher(t_name, params)
    s_net = init_student(t_name, params)

    if mode == "KD":
        best_kd_acc = test_kd(s_net, t_net, params)
        print(f"Best results kd method {s_name}: {best_kd_acc}")
    elif mode == "TAKD":
        best_takd_acc = test_ta(s_net, t_net, params)
        print(f"Best results takd method {s_name}: {best_takd_acc}")
    elif mode == "AB":
        best_ab_acc = test_ab(s_net, t_net, params)
        print(f"Best results ab method {s_name}: {best_ab_acc}")
    elif mode == "RKD":
        best_rkd_acc = test_rkd(s_net, t_net, params)
        print(f"Best results rkd method {s_name}: {best_rkd_acc}")
    else:
        raise RuntimeError("Training mode not supported!")

    print(f"Best results teacher {t_name}: {best_t_acc}")


def setup_torch():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.backends.cudnn.benchmark = True
    # Maximum determinism
    torch.manual_seed(1)
    print(f"Using {device} to train.")
    return device


def start_evaluation(args):
    device = setup_torch()
    num_classes = 100 if args.dataset == "cifar100" else 10
    train_loader, test_loader = get_cifar(num_classes,
                                          batch_size=args.batch_size)

    # Parsing arguments and prepare settings for training
    params = {
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "t_checkpoint": args.t_checkpoint,
        "device": device,
        "s_name": args.s_name,
        "t_name": args.t_name,
        "num_classes": num_classes,
        "train_loader": train_loader,
        "test_loader": test_loader,
        "trial_id": 1,
        # {"_type": "quniform", "_value": [0.05, 1.0, 0.05]},
        "lambda_student": 0.4,
        # {"_type": "choice", "_value": [1, 2, 5, 10, 15, 20]},
        "T_student": 20,
    }
    # Retrieve preconfigured optimizers and schedulers for all runs
    params["optim"] = get_optimizer("SGD", params)
    params["sched"] = get_scheduler("multisteplr", params)
    run_benchmarks(args.mode, params)


if __name__ == "__main__":
    args = parse_arguments()
    start_evaluation(args)
