import math
import torch
from torch import optim
import pdb


def get_optimizer(optim_str, params):
    optim_args = {}

    if optim_str.lower() == "sgd":
        # optim_args["momentum"] = params["momentum"]
        # optim_args["weight_decay"] = params["weight_decay"]
        if params["model_type"] == "edge":
        # add a switch to nesterov for edge models
        # if no_nesterov is not set, use it by default
            optim_args["momentum"] = params["momentum"]
            optim_args["lr"] = params["learning_rate"]
            optim_args["weight_decay"] = params["weight_decay"]
            optim_args["nesterov"] = not(params["no_nesterov"])
        else:
            optim_args["nesterov"] = True
            optim_args["lr"] = params["cloud_learning_rate"]
            optim_args["momentum"] = params["cloud_momentum"]
            optim_args["weight_decay"] = params["cloud_weight_decay"]
        
        return optim.SGD, optim_args

    # Use on the cloud mostly
    elif optim_str.lower() == "adam":
        print("Using adam optimizer")
        if params["model_type"] == "edge":
            optim_args["momentum"] = params["momentum"]
            optim_args["lr"] = params["learning_rate"]
            optim_args["weight_decay"] = params["weight_decay"]
            optim_args["nesterov"] = not(params["no_nesterov"])
        else:
            # optim_args["nesterov"] = True
            optim_args["lr"] = params["cloud_learning_rate"]
            optim_args["weight_decay"] = params["cloud_weight_decay"]
            # optim_args["betas"][0] = params["adam_beta_1"]
            # optim_args["betas"][1] = params["adam_beta_2"]
            # optim_args["eps"] = params["adam_eps"]

        return optim.Adam, optim_args

    elif optim_str.lower() == "novograd":
        optim_args["weight_decay"] = params["weight_decay"]
        return NovoGrad, optim_args

    elif optim_str.lower() == "adabound":
        optim_args["weight_decay"] = params["weight_decay"]
        optim_args["amsbound"] = True
        optim_args["final_lr"] = 0.1
        return AdaBound, optim_args

    print("Requested optimizer not supported!")
    exit(1)


def get_scheduler(sched_str, params):
    sched_args = {}

    if sched_str.lower() == "steplr":
        sched_args["step_size"] = 50
        sched_args["gamma"] = 0.1
        return optim.lr_scheduler.StepLR, sched_args
    
    elif sched_str.lower() == "multisteplr":
        decay_steps = [
            int(0.33 * params["epochs"]),
            int(0.66 * params["epochs"]),
        ]
        print("Decreasing learning rates at epoch ", end="")
        for epoch in decay_steps:
            print(f"{epoch} ", end="")
        print("")
        sched_args["milestones"] = decay_steps
        sched_args["gamma"] = 0.1
        return optim.lr_scheduler.MultiStepLR, sched_args
    
    elif sched_str.lower() == "reducelronplateau":
        sched_args["patience"] = 3
        sched_args["gamma"] = 0.1
        sched_args["verbose"] = True
        return optim.lr_scheduler.ReduceLROnPlateau, sched_args

    elif sched_str.lower() == 'cosineannealinglr':
        print("Using cosine annealing lr")
        sched_args["T_max"] = len(params["train_loader"]) // params["batch_size"]
        # sched_args["T_max"] = params["epochs"]
        tmax = sched_args["T_max"]
        print(f"T_max: {tmax}")
        sched_args["verbose"] = True

        return optim.lr_scheduler.CosineAnnealingLR, sched_args

    elif sched_str.lower() == "constant":
        print("Using constant learning rate")
        # use a constant scheduler, i.e. no scheduler
        return DummyScheduler, sched_args
    print("Requested optimizer not supported!")
    exit(1)


class DummyScheduler():

    def __new__(*args, **kwargs):
        return None


class AdaBound(optim.Optimizer):
    """Implements AdaBound algorithm.
    It has been proposed in `Adaptive Gradient Methods with Dynamic Bound of Learning Rate`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): Adam learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        final_lr (float, optional): final (SGD) learning rate (default: 0.1)
        gamma (float, optional): convergence speed of the bound functions (default: 1e-3)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsbound (boolean, optional): whether to use the AMSBound variant of this algorithm
    .. Adaptive Gradient Methods with Dynamic Bound of Learning Rate:
        https://openreview.net/forum?id=Bkg3g2R9FX
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), final_lr=0.1, gamma=1e-3,
                 eps=1e-8, weight_decay=0, amsbound=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= final_lr:
            raise ValueError(
                "Invalid final learning rate: {}".format(final_lr))
        if not 0.0 <= gamma < 1.0:
            raise ValueError("Invalid gamma parameter: {}".format(gamma))
        defaults = dict(lr=lr, betas=betas, final_lr=final_lr, gamma=gamma, eps=eps,
                        weight_decay=weight_decay, amsbound=amsbound)
        super(AdaBound, self).__init__(params, defaults)

        self.base_lrs = list(map(lambda group: group['lr'], self.param_groups))

    def __setstate__(self, state):
        super(AdaBound, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsbound', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group, base_lr in zip(self.param_groups, self.base_lrs):
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead')
                amsbound = group['amsbound']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsbound:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsbound:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsbound:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * \
                    math.sqrt(bias_correction2) / bias_correction1

                # Applies bounds on actual learning rate
                # lr_scheduler cannot affect final_lr, this is a workaround to apply lr decay
                final_lr = group['final_lr'] * group['lr'] / base_lr
                lower_bound = final_lr * \
                    (1 - 1 / (group['gamma'] * state['step'] + 1))
                upper_bound = final_lr * \
                    (1 + 1 / (group['gamma'] * state['step']))
                step_size = torch.full_like(denom, step_size)
                step_size.div_(denom).clamp_(
                    lower_bound, upper_bound).mul_(exp_avg)

                p.data.add_(-step_size)

        return loss


class AdaBoundW(optim.Optimizer):
    """Implements AdaBound algorithm with Decoupled Weight Decay (arxiv.org/abs/1711.05101)
    It has been proposed in `Adaptive Gradient Methods with Dynamic Bound of Learning Rate`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): Adam learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        final_lr (float, optional): final (SGD) learning rate (default: 0.1)
        gamma (float, optional): convergence speed of the bound functions (default: 1e-3)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsbound (boolean, optional): whether to use the AMSBound variant of this algorithm
    .. Adaptive Gradient Methods with Dynamic Bound of Learning Rate:
        https://openreview.net/forum?id=Bkg3g2R9FX
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), final_lr=0.1, gamma=1e-3,
                 eps=1e-8, weight_decay=0, amsbound=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= final_lr:
            raise ValueError(
                "Invalid final learning rate: {}".format(final_lr))
        if not 0.0 <= gamma < 1.0:
            raise ValueError("Invalid gamma parameter: {}".format(gamma))
        defaults = dict(lr=lr, betas=betas, final_lr=final_lr, gamma=gamma, eps=eps,
                        weight_decay=weight_decay, amsbound=amsbound)
        super(AdaBoundW, self).__init__(params, defaults)

        self.base_lrs = list(map(lambda group: group['lr'], self.param_groups))

    def __setstate__(self, state):
        super(AdaBoundW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsbound', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group, base_lr in zip(self.param_groups, self.base_lrs):
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead')
                amsbound = group['amsbound']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsbound:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsbound:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsbound:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * \
                    math.sqrt(bias_correction2) / bias_correction1

                # Applies bounds on actual learning rate
                # lr_scheduler cannot affect final_lr, this is a workaround to apply lr decay
                final_lr = group['final_lr'] * group['lr'] / base_lr
                lower_bound = final_lr * \
                    (1 - 1 / (group['gamma'] * state['step'] + 1))
                upper_bound = final_lr * \
                    (1 + 1 / (group['gamma'] * state['step']))
                step_size = torch.full_like(denom, step_size)
                step_size.div_(denom).clamp_(
                    lower_bound, upper_bound).mul_(exp_avg)

                if group['weight_decay'] != 0:
                    decayed_weights = torch.mul(p.data, group['weight_decay'])
                    p.data.add_(-step_size)
                    p.data.sub_(decayed_weights)
                else:
                    p.data.add_(-step_size)

        return loss


class NovoGrad(optim.Optimizer):
    def __init__(self, params, grad_averaging=False, lr=0.1, betas=(0.95, 0.98), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(NovoGrad, self).__init__(params, defaults)
        self._lr = lr
        self._beta1 = betas[0]
        self._beta2 = betas[1]
        self._eps = eps
        self._wd = weight_decay
        self._grad_averaging = grad_averaging

        self._momentum_initialized = False

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        if not self._momentum_initialized:
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    grad = p.grad.data
                    if grad.is_sparse:
                        raise RuntimeError(
                            'NovoGrad does not support sparse gradients')

                    v = torch.norm(grad)**2
                    m = grad / (torch.sqrt(v) + self._eps) + self._wd * p.data
                    state['step'] = 0
                    state['v'] = v
                    state['m'] = m
                    state['grad_ema'] = None
            self._momentum_initialized = True

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                state['step'] += 1

                step, v, m = state['step'], state['v'], state['m']
                grad_ema = state['grad_ema']

                grad = p.grad.data
                g2 = torch.norm(grad)**2
                grad_ema = g2 if grad_ema is None else grad_ema * \
                    self._beta2 + g2 * (1. - self._beta2)
                grad *= 1.0 / (torch.sqrt(grad_ema) + self._eps)

                if self._grad_averaging:
                    grad *= (1. - self._beta1)

                g2 = torch.norm(grad)**2
                v = self._beta2 * v + (1. - self._beta2) * g2
                m = self._beta1 * m + \
                    (grad / (torch.sqrt(v) + self._eps) + self._wd * p.data)
                bias_correction1 = 1 - self._beta1 ** step
                bias_correction2 = 1 - self._beta2 ** step
                step_size = group['lr'] * \
                    math.sqrt(bias_correction2) / bias_correction1

                state['v'], state['m'] = v, m
                state['grad_ema'] = grad_ema
                p.data.add_(-step_size, m)
        return loss
