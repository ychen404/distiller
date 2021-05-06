import sys
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from optimizer import get_optimizer, get_scheduler
import pdb
from aggregate_method.early_stopping import EarlyStopping


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

def init_progress_bar(train_loader):
    # pdb.set_trace()
    batch_size = train_loader.batch_size
    bar_format = "{desc}{percentage:3.0f}%"
    # bar_format += "|{bar}|"
    bar_format += " {n_fmt}/{total_fmt} [{elapsed} < {remaining}]"
    bar_format += "{postfix}"
    # if stderr has no tty disable the progress bar
    disable = not sys.stderr.isatty()
    # print(f"###################### disable = {disable} ###########################")

    t = tqdm(total=len(train_loader) * batch_size,
             bar_format=bar_format, disable=disable, position=0, leave=True)
    if disable:
        # a trick to allow execution in environments where stderr is redirected
        t._time = lambda: 0.0
    return t

class Trainer():
    # def __init__(self, net, config):
    def __init__(self, net, config, idxs=None):
        self.net = net
        self.device = config["device"]
        self.name = config["test_name"]
        # Retrieve preconfigured optimizers and schedulers for all runs
        optim = config["optim"]
        
        # The sampler works with the dataset object instead of the dataloader object
        self.model_type = config["model_type"]
        
        if self.model_type == "edge":
            sched = config["sched"]
        else:
            sched = config["cloud_sched"]

        self.optim_cls, self.optim_args = get_optimizer(optim, config)
        self.sched_cls, self.sched_args = get_scheduler(sched, config)
        self.optimizer = self.optim_cls(net.parameters(), **self.optim_args)
        self.scheduler = self.sched_cls(self.optimizer, **self.sched_args)

        self.loss_fun = nn.CrossEntropyLoss()
        # self.local_bs = config["batch_size"]
        # remove the full data loader

        self.test_loader = config["test_loader"]
        

        # initialize the early_stopping object

        if self.model_type == "cloud":
        
            # to track the training loss as the model trains
            train_losses = []
            # to track the validation loss as the model trains
            valid_losses = []
            # to track the average training loss per epoch as the model trains
            avg_train_losses = []
            # to track the average validation loss per epoch as the model trains
            avg_valid_losses = [] 
        
            early_stopping = EarlyStopping(patience=20, verbose=True)

        print(f"Model type: {self.model_type}")
        # There is no train_dataset for cloud model; it loads the cifar100 train loader directly
        if self.model_type == "edge":
            self.train_dataset = config["train_dataset"]
            self.num_users = config["num_users"]
        else:
            self.num_users = 1
        
        self.indexs = idxs
         
        # self.batch_size = self.train_loader.batch_size
        self.batch_size = config["batch_size"]
        self.config = config
        

        # Override the train_loader with the sampled one if indexes are provided 
        if self.model_type == "edge":
            # edge models need data partition
            self.data_split = DatasetSplit(self.train_dataset, self.indexs)
            self.train_loader = DataLoader(self.data_split, batch_size=self.batch_size, 
                    shuffle=True, num_workers=self.config["nthread"], pin_memory=torch.cuda.is_available())

        elif self.model_type == "cloud":
            # cloud model uses cifar100 train loader from the config file
            self.train_loader = config["train_loader"]

        # self.train_loader = DataLoader(DatasetSplit(self.train_dataset, self.indexs), batch_size=self.batch_size, 
        #         shuffle=True, num_workers=self.config["nthread"], pin_memory=torch.cuda.is_available())

        # tqdm bar
        self.t_bar = None
        folder = config["results_dir"]
        self.model_file = folder.joinpath(f"{self.name}_ckpt.pth.tar")
        acc_file_name = folder.joinpath(f"{self.name}_acc.csv")
        lr_file_name = folder.joinpath(f"{self.name}_lr.csv")
        
        self.acc_file = acc_file_name.open("w+")


        self.acc_file.write("Accuracy numbers are recorded every epoch\n")
        self.acc_file.write("Training Acc,Test Acc\n")

        if self.model_type == "cloud":
            self.lr_file = lr_file_name.open("w+")
            self.lr_file.write("Cloud learning rate every epoch\n")
            self.lr_file.write("Learning rate\n")

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def set_loss_fun(self, loss_fun):
        self.loss_fun = loss_fun

    def calculate_loss(self, data, target):
        raise NotImplementedError(
            "calculate_loss should be implemented by subclass!")

    def train_single_epoch(self, t_bar):
        self.net.train()
        total_correct = 0.0
        total_loss = 0.0
               
        if self.model_type == "edge":
            len_train_set = len(self.train_dataset) / self.num_users
        
        else: 
            len_train_set = len(self.train_loader.dataset)

        # division = 20
        # portion_data = len(self.train_loader) // division
        # print(self.train_loader.dataset)
        # pdb.set_trace()
        for batch_idx, (x, y) in enumerate(self.train_loader):
            # if batch_idx > portion_data:
            #     break
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            self.optimizer.zero_grad()

            # this function is implemented by the subclass
            y_hat, loss = self.calculate_loss(x, y)

            # Metric tracking boilerplate
            pred = y_hat.data.max(1, keepdim=True)[1]
            total_correct += pred.eq(y.data.view_as(pred)).sum()
            total_loss += loss
            curr_acc = 100.0 * (total_correct / float(len_train_set))
            curr_loss = (total_loss / float(batch_idx))
        
            t_bar.update(self.batch_size)
            t_bar.set_postfix_str(f"Acc {curr_acc:.3f}% Loss {curr_loss:.3f}")
        
        total_acc = float(total_correct / len_train_set)
        return total_acc

    def get_accuracy(self, current_round):
        val_acc = self.validate(current_round)
        self.acc_file.write(f"None,{val_acc}\n")


    def train(self, current_round):
        epochs = self.config["epochs"]
        
        if self.model_type == "cloud":
            cloud_lr = self.config["cloud_learning_rate"]

        t_bar = init_progress_bar(self.train_loader)
        for epoch in range(epochs):
            # update progress bar
            t_bar.reset()
            t_bar.set_description(f"Round {current_round} | Epoch {epoch}")
            # perform training
            train_acc = self.train_single_epoch(t_bar)
            # validate the output 
            val_acc = self.validate(current_round, epoch)
            
            # Save the intermediate model
            # self.save(epoch, name='test_' + str(epoch))

            # if val_acc > best_acc:
            #     best_acc = val_acc
             
            self.save(epoch, name=self.model_file)

            # update the scheduler
            if self.scheduler:
                self.scheduler.step()
                if self.model_type == "cloud":
                    cloud_lr = self.scheduler.get_last_lr()
                    self.lr_file.write(f"{cloud_lr}\n")
                    print(f"Cloud lr: {cloud_lr}")

            self.acc_file.write(f"{train_acc},{val_acc}\n")
            
            # if self.model_type == "cloud":
            #     self.lr_file.write(f"{cloud_lr}\n")

            # print(f"acc_file, train_acc, val_acc: {self.acc_file}, {train_acc}, {val_acc}")
            # exit()
        tqdm.clear(t_bar)
        t_bar.close()
        self.acc_file.close()


    def validate(self, current_round, epoch=0):
        self.net.eval()
        acc = 0.0
        with torch.no_grad():
            correct = 0
            acc = 0
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                output = self.net(images)
                # Standard Learning Loss ( Classification Loss)
                loss = self.loss_fun(output, labels)
                # get the index of the max log-probability
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(labels.data.view_as(pred)).cpu().sum()

            acc = float(correct) / len(self.test_loader.dataset)
            print(f"\nRound {current_round} | Epoch {epoch}: Validation set: Average loss: {loss:.4f},"
                  f" Accuracy: {correct}/{len(self.test_loader.dataset)} "
                  f"({acc * 100.0:.3f}%)")
        return acc

    def save(self, epoch, name):
        torch.save({"model_state_dict": self.net.state_dict(), }, name)


class BaseTrainer(Trainer):

    def calculate_loss(self, data, target):
        # Standard Learning Loss ( Classification Loss)
        output = self.net(data)
        loss = self.loss_fun(output, target)
        loss.backward()
        self.optimizer.step()
        return output, loss

class KDTrainer(Trainer):
    # def __init__(self, s_net, t_net, config):
    # super(KDTrainer, self).__init__(s_net, config)
    # Inherite the idxs from Trainer
    def __init__(self, s_net, t_net, config, idxs=None):
        super(KDTrainer, self).__init__(s_net, config, idxs)
        # the student net is the base net
        self.s_net = self.net
        self.t_net = t_net
        self.kd_fun = nn.KLDivLoss(size_average=False)

    def kd_loss(self, out_s, out_t, target):
        lambda_ = self.config["lambda"]
        T = self.config["T"]
        # Standard Learning Loss ( Classification Loss)
        # Remove the Standard learning loss for unlabel case
        # loss = self.loss_fun(out_s, target)

        # Knowledge Distillation Loss
        # batch_size = target.shape[0]
        batch_size = out_s.shape[0]
        s_max = F.log_softmax(out_s / T, dim=1)
        t_max = F.softmax(out_t / T, dim=1)
        # print(f"s_max: {s_max.shape} t_max: {t_max.shape}")

        loss_kd = self.kd_fun(s_max, t_max) / batch_size
        # loss = (1 - lambda_) * loss + lambda_ * T * T * loss_kd
        loss = loss_kd
        return loss

    def calculate_loss(self, data, target):
        out_s = self.s_net(data)
        out_t = self.t_net(data)
        loss = self.kd_loss(out_s, out_t, target)
        loss.backward()
        self.optimizer.step()
        return out_s, loss

class KDTrainerNoLabel(Trainer):
    # def __init__(self, s_net, t_net, config):
    # super(KDTrainer, self).__init__(s_net, config)
    # Inherite the idxs from Trainer
    def __init__(self, s_net, t_net, config, idxs=None):
        super(KDTrainerNoLabel, self).__init__(s_net, config, idxs)
        # the student net is the base net
        self.s_net = self.net
        self.t_net = t_net
        self.kd_fun = nn.KLDivLoss(size_average=False)

    def kd_loss(self, out_s, out_t):
        lambda_ = self.config["lambda"]
        T = self.config["T"]
        # Standard Learning Loss ( Classification Loss)
        # Remove the Standard learning loss for unlabel case
        # loss = self.loss_fun(out_s, target)

        # Knowledge Distillation Loss
        # batch_size = target.shape[0]
        batch_size = out_s.shape[0]
        s_max = F.log_softmax(out_s / T, dim=1)
        t_max = F.softmax(out_t / T, dim=1)
        # print(f"s_max: {s_max.shape} t_max: {t_max.shape}")

        loss_kd = self.kd_fun(s_max, t_max) / batch_size
        # loss = (1 - lambda_) * loss + lambda_ * T * T * loss_kd
        loss = loss_kd
        return loss

    def calculate_loss(self, data, target):
        out_s = self.s_net(data)
        out_t = self.t_net(data)
        loss = self.kd_loss(out_s, out_t)
        loss.backward()
        self.optimizer.step()
        return out_s, loss


class MultiTrainer(KDTrainer):
    def __init__(self, s_net, t_nets, config):
        super(MultiTrainer, self).__init__(s_net, s_net, config)
        # the student net is the base net
        self.s_net = self.net
        self.t_nets = t_nets

    def kd_loss(self, out_s, out_t, target):
        T = self.config["T"]
        # Knowledge Distillation Loss
        batch_size = target.shape[0]

        s_max = F.log_softmax(out_s / T, dim=1)
        t_max = F.softmax(out_t / T, dim=1)
        loss_kd = self.kd_fun(s_max, t_max) / batch_size
        return loss_kd

    def calculate_loss(self, data, target):
        lambda_ = self.config["lambda"]
        T = self.config["T"]
        out_s = self.s_net(data)
        # Standard Learning Loss ( Classification Loss)
        loss = self.loss_fun(out_s, target)
        # Knowledge Distillation Loss
        loss_kd = 0.0
        for t_net in self.t_nets:
            out_t = t_net(data)
            loss_kd += self.kd_loss(out_s, out_t, target)
        loss_kd /= len(self.t_nets)
        loss = (1 - lambda_) * loss + lambda_ * T * T * loss_kd
        loss.backward()
        self.optimizer.step()
        return out_s, loss

# For multiple teacher. 
# It is similar to the MultiTrainer class just remove the learning from labels 
class MultiTeacher(KDTrainer):
    def __init__(self, s_net, t_nets, config):
        super(MultiTeacher, self).__init__(s_net, s_net, config)
        # the student net is the base net
        self.s_net = self.net
        self.t_nets = t_nets

    def kd_loss(self, out_s, out_t):
        T = self.config["T"]
        # Knowledge Distillation Loss
        # batch_size = target.shape[0]
        batch_size = out_s.shape[0]
        s_max = F.log_softmax(out_s / T, dim=1)
        t_max = F.softmax(out_t / T, dim=1)
        loss_kd = self.kd_fun(s_max, t_max) / batch_size
        return loss_kd

    def calculate_loss(self, data, target):
        lambda_ = self.config["lambda"]
        T = self.config["T"]
        out_s = self.s_net(data)

        # Standard Learning Loss ( Classification Loss)
        # loss = self.loss_fun(out_s, target)
        # Knowledge Distillation Loss
        
        loss_kd = 0.0
        # Go through all the edge models
        # Use the same batch of data to run inferece of each edge model
        for i, t_net in enumerate(self.t_nets):
            out_t = t_net(data)
            loss_kd += self.kd_loss(out_s, out_t)
        loss_kd /= len(self.t_nets)
        # loss = (1 - lambda_) * loss + lambda_ * T * T * loss_kd
        loss = loss_kd
        loss.backward()
        self.optimizer.step()
        
        return out_s, loss