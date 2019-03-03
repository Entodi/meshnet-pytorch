import time
import torch
import pandas as pd
import numpy as np
import random
import signal
from torch.autograd import Variable
from torch.nn.functional import log_softmax
from sklearn.metrics import f1_score

from nipy import save_image, load_image 
from nipy.core.api import Image


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


def convert_npy_to_nii(npy_data, numpy_filename, base_nifti_filename):
    bnifti = load_image(base_nifti_filename)
    img = Image.from_image(bnifti, data=npy_data.astype('uint8'))
    save_image(img, numpy_filename)
    print ('Saved {}..'.format(numpy_filename))


def predict(data, dataloader, net, n_subvolumes, n_classes, subvolume_shape):
    net.eval()
    runtime = np.zeros(len(dataloader) // n_subvolumes)
    segmentations = {}
    dataset = data.get_all_data()
    for i in range(len(dataset)):
        segmentations[i] = torch.zeros(tuple(np.insert(dataset[i].get_volume_shape(), 0, n_classes)), dtype=torch.uint8)
    for i, data in enumerate(dataloader, 0):
        subj_id = i // n_subvolumes
        inputs, _, coords = data
        inputs = Variable(inputs.cuda(), requires_grad=False)
        start = time.time()
        outputs = net(inputs)        
        end = time.time()
        runtime[subj_id] += end - start
        outputs = log_softmax(outputs, dim=1)
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.cpu()
        for j in range(predicted.shape[0]):
            c_j = coords[j]
            for c in range(n_classes):
                segmentations[subj_id][c, c_j[0, 0]:c_j[0, 1], 
                c_j[1, 0]:c_j[1, 1], c_j[2, 0]:c_j[2, 1]] += (predicted[j] == c)
    for i in segmentations.keys():
        segmentations[i] = torch.max(segmentations[i], 0)[1]
    return segmentations, runtime


def calculate_metrics(data, dataloader, net, n_subvolumes, n_classes, subvolume_shape, metrics=[(f1_score, 'dice')], save_prediction=False, model_name=''):
    net.eval()
    segmentations, runtime = predict(data, dataloader, net, n_subvolumes, n_classes, subvolume_shape)
    dataset = data.get_all_data()
    inputs_filenames = data.get_paths()
    columns = ['name']
    for m in metrics:
        columns += ['{}_{}'.format(m[1], i) for i in range(n_classes)]
        columns += ['{}_{}'.format('n_voxel_true', i) for i in range(n_classes)]
        columns += ['{}_{}'.format('n_voxel_pred', i) for i in range(n_classes)]
    results = pd.DataFrame(columns=columns)
    results['name'] = data.get_paths()
    results['time'] = runtime
    for i in segmentations.keys():
        # Back to original shape
        original = dataset[i].get_original()
        groundthruth = dataset[i].get_target()[original[0]:-original[0],
            original[1]:-original[1],
            original[2]:-original[2]]
        segmentation = segmentations[i][original[0]:-original[0],
            original[1]:-original[1],
            original[2]:-original[2]]
        for c in range(n_classes):
            for m in metrics:
                column = '{}_{}'.format(m[1], c)
                if m[1] == 'dice':
                    mask_groundthruth = (groundthruth == c).numpy().flatten()
                    mask_segmentation = (segmentation == c).numpy().flatten()
                elif m[1] == ['hd']:
                    mask_groundthruth = (groundthruth == c).numpy()
                    mask_segmentation = (segmentation == c).numpy()
                results[column].loc[i] = m[0](mask_groundthruth, mask_segmentation)
                results['n_voxel_true_{}'.format(c)].loc[i] = np.sum(
                    mask_groundthruth)
                results['n_voxel_pred_{}'.format(c)].loc[i] = np.sum(
                    mask_segmentation)
        if save_prediction:
            filename = '{}_{}_prediction.nii.gz'.format(
                inputs_filenames[i].split('.nii.gz')[0], model_name)
            convert_npy_to_nii(
                segmentation.numpy(), filename, inputs_filenames[i])
    return results