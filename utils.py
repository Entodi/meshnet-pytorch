import torch
import numpy as np
import random
import signal
from torch.autograd import Variable
from torch.nn.functional import log_softmax

def train(dataloader, net, optimizer, criterion):
    net.train()
    for i, data in enumerate(dataloader, 0):
        # get the inputs
        inputs, labels, _ = data
        # wrap them in Variable
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


def valid(dataloader, net, criterion):
    net.eval()
    loss = 0
    for i, data in enumerate(dataloader, 0):
        inputs, labels, _ = data
        inputs, labels = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(), requires_grad=False)
        outputs = net(inputs)
        loss += criterion(outputs, labels).cpu().detach().numpy()
        outputs = log_softmax(outputs, dim=1)
        _, predicted = torch.max(outputs.data, 1)
    loss /= len(dataloader)
    return loss


def load_net_weights(net, weights_filename):
    state_dict = torch.load(weights_filename)
    state = net.state_dict()
    state.update(state_dict)
    net.load_state_dict(state)
    return net


def worker_init_fn(x):
    seed = (int(torch.initial_seed()) + x) % (2**32-1)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def plot_evaluation(viz, results, model_name):
    def get_Y_legend(key, v_train, v_valid):
        Y = []
        legend = []

        Y.append(np.array(v_train))
        if v_valid is not None:
            Y.append(np.array(v_valid))
            legend.append('{} (train)'.format(key))
            legend.append('{} (test)'.format(key))
        else:
            legend.append(key)

        return Y, legend

    train_summary = results['train']
    valid_summary = results['valid']
    for k in train_summary.keys():
        v_train = train_summary[k]
        v_valid = valid_summary[k] if k in valid_summary.keys() else None
        if isinstance(v_train, dict):
            Y = []
            legend = []
            for k_ in v_train:
                vt = v_valid.get(k_) if v_valid is not None else None
                Y_, legend_ = get_Y_legend(k_, v_train[k_], vt)
                Y += Y_
                legend += legend_
        else:
            Y, legend = get_Y_legend(k, v_train, v_valid)

        opts = dict(
            xlabel='epochs',
            legend=legend,
            ylabel=k,
            title=k)

        if len(Y) == 1:
            Y = Y[0]
            X = np.arange(Y.shape[0])
        else:
            Y = np.column_stack(Y)
            X = np.column_stack([np.arange(Y.shape[0])] * Y.shape[1])

        viz.line(
            Y=Y,
            X=X,
            env=model_name,
            opts=opts,
            win='line_{}'.format(k))