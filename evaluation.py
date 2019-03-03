import argparse, random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import data
import utils
import models

from yaml import load, dump

parser = argparse.ArgumentParser(description='Segmentation evaluation')
#-----------------------------------------------------------------------------
# Model arguments 
parser.add_argument('--models_file', help='A path to file with list of models')
#-----------------------------------------------------------------------------
# Data arguments
parser.add_argument('--evaluation_path', help='A path to file with dataset')
parser.add_argument('--batch_size', default=8, type=int, metavar='N', 
    help='size of batch')
parser.add_argument('--sv_w', default=38, type=int, 
    metavar='N', help='width of subvolumes')
parser.add_argument('--sv_h', default=38, type=int, 
    metavar='N', help='height of subvolumes')
parser.add_argument('--sv_d', default=38, type=int, 
    metavar='N', help='depth of subvolumes')
parser.add_argument('--n_subvolumes', default=1024, type=int, 
    metavar='N', help='number of total subvolumes per brain')
parser.add_argument('--n_threads', '-j', default=4, type=int, metavar='N', 
    help='number of data loading threads (default: 2)')
parser.add_argument('--save_prediction', action='store_true')
#-----------------------------------------------------------------------------
# Misc arguments
parser.add_argument('--seed', default=0, type=int, 
    metavar='N', help='seed')
parser.add_argument('--add_info', default='', 
    help='Additional modificator to the experiment name')
parser.add_argument('--name', default='', help='name to save results')
args = parser.parse_args()
print (args)
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
# Read models to evaluate
f = open(args.models_file, 'r')
models_list = f.read().splitlines()
# Read volumes to segment
f = open(args.evaluation_path, 'r')
files = f.read().splitlines()
#-----------------------------------------------------------------------------
# Evaluation
for m in models_list:
    #-----------------------------------------------------------------------------
    # Load models
    print ('Model: {}'.format(m))
    model_info = load(open(m, 'r'))
    best_model = model_info['best_model']
    model, params = models.create_model(m)
    if torch.cuda.device_count() > 0:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        model.cuda()
    model = utils.load_net_weights(model, best_model)
    #------------------------------------------------------------------------------
    # Run evaluation
    model.eval()
    results = pd.DataFrame()
    for f in files:
        print ('File: {}'.format(f))
        subvolume_shape = np.array([args.sv_d, args.sv_h, args.sv_w])
        dataset = data.VolumetricDataset([f], args.n_subvolumes, 
            subvolume_shape, extended=True, evaluation=True)
        dataset.build()
        dataset_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, 
            shuffle=False, num_workers=args.n_threads,
            worker_init_fn=lambda x: utils.worker_init_fn(x)
        )
        temp_results = utils.evaluate(
            dataset, dataset_loader, model,     
            save_prediction=args.save_prediction, 
            model_name=model_info['name']
        )
        results = pd.concat([results, temp_results])
    #-------------------------------------------------------------------------------
    # Save evaluation
    print (results.head())
    try:
        os.mkdir('./metrics/')
    except:
        pass
    if args.name == '':
        name = './metrics/{}_{}_{}_{}x{}x{}_{}.csv'.format(
            model_info['name'], args.batch_size, 
            args.n_subvolumes, args.sv_d, args.sv_h, args.sv_w, args.seed)
    else:
        name = args.name
    results.to_csv(name, index=False)