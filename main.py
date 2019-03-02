import argparse, random, time, os, visdom
from yaml import load, dump
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

import data
import utils
import models

parser = argparse.ArgumentParser(description='HCP tissue segmentation Training')
#-----------------------------------------------------------------------------
# Data arguments
parser.add_argument('--train_path', metavar='PATH', 
    required=True, help='Path to list with brains for training')
parser.add_argument('--validation_path', metavar='PATH', 
    required=True, help='Path to list with brains for validation')
parser.add_argument('--n_threads', '-j', default=4, type=int, 
    metavar='N', help='Number of data loading threads (default: 2)')
parser.add_argument('--n_subvolumes', default=100, type=int, 
    metavar='N', help='Number of total subvolumes to sample from one brain')
parser.add_argument('--sv_w', default=38, type=int, metavar='N', 
    help='Width of subvolumes')
parser.add_argument('--sv_h', default=38, type=int, metavar='N', 
    help='Height of subvolumes')
parser.add_argument('--sv_d', default=38, type=int, metavar='N', 
    help='Depth of subvolumes')
#-----------------------------------------------------------------------------
# Model arguments 
parser.add_argument('--model', default='', help='Model yml file')
parser.add_argument('--weight_init', default='kaiming_normal', 
    help='weight initilization')
parser.add_argument('--loss', default='CE', help='Loss function. Default: CE')
#-----------------------------------------------------------------------------
# Training arguments
parser.add_argument('--n_epochs', default=1000, type=int, metavar='N', 
    help='number of total epochs to run')
parser.add_argument('--batch_size', default=4, type=int, metavar='N', 
    help='size of batch')
parser.add_argument('--lr', type=float, default=1e-2, metavar='LR', 
	help='Base Learning rate (default: 0.01)')
parser.add_argument('--optimizer', default='Adam', help='Optimizer. Default: Adam')
#-----------------------------------------------------------------------------
# Misc arguments
parser.add_argument('--seed', default=0, type=int, metavar='N', 
    help='seed')
parser.add_argument('--no_visdom', action='store_true', 
    help='Turn off visdom monitoring')
parser.add_argument('--visdom_server', default='http://localhost', 
    help='Visdom Server URL')
parser.add_argument('--visdom_port', default=8097, type=int, 
    help='Visdom Server Port')
parser.add_argument('--add_info', default='', metavar='S', 
    help='Additional modificator to the experiment name')
args = parser.parse_args()
print ('{}'.format(args))
#-----------------------------------------------------------------------------
# Reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
# When running on the CuDNN backend, two further options must be set:
# Deterministic mode can have a performance impact, depending on your model.
# https://pytorch.org/docs/stable/notes/randomness.html
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False
#-----------------------------------------------------------------------------
# Create a model
model_info = load(open(args.model, 'r'))
model, params = models.create_model(args.model, 
    weight_initilization=args.weight_init)
#-----------------------------------------------------------------------------
# Naming the model and create a directory to save the experiment
current_time = time.strftime("%Y_%m_%d_%H_%M_%S",time.gmtime())
model_name = '{}_{}_{}_model_{}'.format(
    model_info['name'], args.loss, args.add_info, current_time)
modelPath = './models/{}/'.format(model_name)
try:
    os.mkdir(modelPath)
except:
    raise OSError("Can't create destination directory (%s)!" % (modelPath))  
#-----------------------------------------------------------------------------
# Setting visdom
if not args.no_visdom:
    try:
        viz = visdom.Visdom(server=args.visdom_server, 
            port=args.visdom_port, env=model_name)
        startup_sec = 1
        while not viz.check_connection() and startup_sec > 0:
            time.sleep(0.1)
            startup_sec -= 0.1
        assert viz.check_connection(), 'No connection could be formed quickly'
    except BaseException as e:
        print("The visdom experienced an exception while running: {}".format(repr(e)))
#-----------------------------------------------------------------------------
# Prepare dataset
subvolume_shape = np.array([args.sv_d, args.sv_h, args.sv_w])
train_dataset = data.VolumetricDataset(
    args.train_path, args.n_subvolumes, subvolume_shape)
train_dataset.build()
validation_dataset = data.VolumetricDataset(
    args.validation_path, args.n_subvolumes, subvolume_shape)
validation_dataset.build()
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, 
    shuffle=True, num_workers=args.n_threads, 
    worker_init_fn=lambda x: utils.worker_init_fn(x))
valid_loader = torch.utils.data.DataLoader(
    validation_dataset, batch_size=args.batch_size,
    shuffle=False, num_workers=args.n_threads, 
    worker_init_fn=lambda x: utils.worker_init_fn(x))
#-----------------------------------------------------------------------------
# Set a loss function
if args.loss == 'CE':
    criterion = nn.CrossEntropyLoss()
else:
    assert False, 'Loss {} isn\'t defined'.format(args.loss)
#-----------------------------------------------------------------------------
# Run on GPU
if torch.cuda.is_available():
    print("Using ", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
    model.cuda()
    criterion.cuda()
#-----------------------------------------------------------------------------
# Set an optimizer
if args.optimizer == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=False)
else:
    assert False, 'An optimizer {} isn\'t defined'.format(args.optimizer)
#-----------------------------------------------------------------------------
# Set a CosineAnnealingLR scheduler
T_max = 40
eta_min = 1e-10
T_mult = 2
scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
#-----------------------------------------------------------------------------
# Training
T_prev = 1
if not args.no_visdom:
    evaluation = {
    'train': {args.loss: np.array([]), 'Learning Rate': np.array([])}, 
    'valid': {args.loss: np.array([])}
    }
evaluation_df = pd.DataFrame(
    columns=['train_{}'.format(args.loss), 'valid_{}'.format(args.loss)])  
best_model = 'None'
best_valid_loss = 100000
best_gen_gap = 10000
for epoch in range(0, args.n_epochs):
    # Warm restart check
    if (epoch - T_prev) == T_max:
        T_prev += T_max
        T_max *= T_mult
        print ('Warm Restart...')
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=False)
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    scheduler.step()
    print ('Learning rate:', scheduler.get_lr())
    print('Training...')
    start = time.time()
    utils.train(train_loader, model, optimizer, criterion)
    end = time.time()
    print(end - start)
    print('Validating...')
    start = time.time()
    train_loss = utils.valid(train_loader, model, criterion)
    end = time.time()
    print(end - start)
    start = time.time()
    valid_loss = utils.valid(valid_loader, model, criterion)
    end = time.time()
    print(end - start)
    print ('Epoch {} Train CEL: {} Valid CEL: {}'.format(
    epoch, np.round(train_loss, 3), np.round(valid_loss, 3)))
    # sent results to Visdom
    if not args.no_visdom:
        evaluation['train']['Learning Rate'] = np.hstack(
            [evaluation['train']['Learning Rate'], scheduler.get_lr()])
        evaluation['train'][args.loss] = np.hstack(
            [evaluation['train'][args.loss], train_loss])
        evaluation['valid'][args.loss] = np.hstack(
            [evaluation['valid'][args.loss], valid_loss])
        utils.plot_evaluation(viz, evaluation, model_name)
    # save the model
    model_filename = modelPath + 'model_state_dict_' + str(epoch) + '.pt'
    torch.save(model.state_dict(), open(model_filename, 'wb'))
    # select current best model
    if valid_loss < best_valid_loss and np.abs(
        train_loss - valid_loss) < best_gen_gap:
            best_valid_loss = valid_loss
            best_gen_gap = np.abs(train_loss - valid_loss)
            best_model = model_filename
    # archive results to csv
    evaluation_df.at[epoch] = [train_loss, valid_loss]
    evaluation_df.to_csv(modelPath + 'evaluation.csv', index=False, sep=',')
    # save the model information
    model_info['run_args'] = '{}'.format(args)
    model_info['best_model'] = best_model
    model_info['valid_loss'] = best_valid_loss
    model_info['gen_gap'] = best_gen_gap
    model_info_filename =  modelPath + 'model_info.yml'
    dump(model_info, open(model_info_filename, 'w'))


