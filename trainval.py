from haven import haven_chk as hc
from haven import haven_results as hr
from haven import haven_utils as hu
import torch
import torchvision
import tqdm
import pandas as pd
import pprint
import itertools
import os
import pylab as plt
import exp_configs
import time
import numpy as np

from src import models
from src import datasets
from src import utils as ut


import argparse

from torch.utils.data import sampler
from torch.utils.data.sampler import RandomSampler
from torch.backends import cudnn
from torch.nn import functional as F
from torch.utils.data import DataLoader

cudnn.benchmark = True


def trainval(exp_dict, savedir_base, datadir, reset=False, num_workers=0):
    # bookkeepting stuff
    # ==================
    pprint.pprint(exp_dict)
    exp_id = hu.hash_dict(exp_dict) #tự tạo 1 cái key trong bảng hash
    os.makedirs(savedir_base, exist_ok=True)
    savedir = os.path.join(savedir_base, exp_id)
    if reset:
        hc.delete_and_backup_experiment(savedir)

    os.makedirs(savedir, exist_ok=True)
    hu.save_json(os.path.join(savedir, "exp_dict.json"), exp_dict)
    print("Experiment saved in %s" % savedir)

    # set seed
    # ==================
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print('-----done-torch-cuda-seed-----')

    # Dataset
    # ==================
    # train set
    train_set = datasets.get_dataset(dataset_dict=exp_dict["dataset"],
                                     split="train",
                                     datadir=datadir,
                                     exp_dict=exp_dict,
                                     dataset_size=exp_dict['dataset_size'])

    # val set
    val_set = datasets.get_dataset(dataset_dict=exp_dict["dataset"],
                                   split="val",
                                   datadir=datadir,
                                   exp_dict=exp_dict,
                                   dataset_size=exp_dict['dataset_size'])

    # test set
    test_set = datasets.get_dataset(dataset_dict=exp_dict["dataset"],
                                   split="test",
                                   datadir=datadir,
                                   exp_dict=exp_dict,
                                   dataset_size=exp_dict['dataset_size'])

    # val_sampler = torch.utils.data.SequentialSampler(val_set)
    val_loader = DataLoader(val_set,
                            # sampler=val_sampler,
                            batch_size=1,
                            collate_fn=ut.collate_fn,
                            num_workers=num_workers)

    test_loader = DataLoader(test_set,
                            # sampler=val_sampler,
                            batch_size=1,
                            collate_fn=ut.collate_fn,
                            num_workers=num_workers)

    print('-----done-loader-dts-----')                        

    # Model
    # ==================
    model = models.get_model(model_dict=exp_dict['model'],
                             exp_dict=exp_dict,
                             train_set=train_set).cuda()

    # model.opt = optimizers.get_optim(exp_dict['opt'], model)
    model_path = os.path.join(savedir, "model.pth")
    score_list_path = os.path.join(savedir, "score_list.pkl")

    if os.path.exists(score_list_path):
        # resume experiment
        model.load_state_dict(hu.torch_load(model_path))
        score_list = hu.load_pkl(score_list_path)
        s_epoch = score_list[-1]['epoch'] + 1
    else:
        # restart experiment
        score_list = []
        s_epoch = 0

    # Train & Val
    # ==================
    print("Starting experiment at epoch %d" % (s_epoch))
    model.waiting = 0
    model.val_score_best = -np.inf
    
    train_sampler = torch.utils.data.RandomSampler(
                                train_set, replacement=True, 
                                num_samples=len(val_set))

    train_loader = DataLoader(train_set,
                            sampler=train_sampler,
                            collate_fn=ut.collate_fn,
                            batch_size=exp_dict["batch_size"], 
                            drop_last=True, 
                            num_workers=num_workers)

    print('-----done-train-sampler-loader-----')                        
    
    for e in range(s_epoch, exp_dict['max_epoch']):
        # Validate only at the start of each cycle
        score_dict = {}
        # Train the model
        train_dict = model.train_on_loader(train_loader)
        # Validate the model
        val_dict = model.val_on_loader(val_loader, savedir_images=os.path.join(savedir, "images"), n_images=3)
        for k in val_dict:
            if "val_" in k:
                score_dict[k] = val_dict[k]

        # Get new score_dict
        score_dict.update(train_dict)
        score_dict["epoch"] = e
        score_dict["waiting"] = model.waiting

        model.waiting += 1

        # Add to score_list and save checkpoint
        score_list += [score_dict]

        # Save Best Checkpoint
        score_df = pd.DataFrame(score_list)
        if score_dict["val_score"] >= model.val_score_best:
            test_dict = model.val_on_loader(test_loader,
                                    savedir_images=os.path.join(savedir, "images"),
                                    n_images=3)  
            score_dict.update(test_dict)

            hu.save_pkl(os.path.join(savedir, "score_list_best.pkl"), score_list)
            score_df.to_csv(os.path.join(savedir, "score_best_df.csv"))
            hu.torch_save(os.path.join(savedir, "model_best.pth"),
                        model.get_state_dict())
            model.waiting = 0
            model.val_score_best = score_dict["val_score"]
            print("Saved Best: %s" % savedir)

        # Report & Save
        score_df = pd.DataFrame(score_list)
        # score_df.to_csv(os.path.join(savedir, "score_df.csv"))
        print("\n", score_df.tail(), "\n")
        hu.torch_save(model_path, model.get_state_dict())
        hu.save_pkl(score_list_path, score_list)
        print("Checkpoint Saved: %s" % savedir)

    print('Experiment completed et epoch %d' % e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_group_list', nargs="+")
    parser.add_argument('-sb', '--savedir_base', required=True)
    parser.add_argument('-d', '--datadir', default=None)
    parser.add_argument("-r", "--reset",  default=0, type=int)
    parser.add_argument("-ei", "--exp_id", default=None)
    parser.add_argument("-j", "--run_jobs", default=0, type=int)
    parser.add_argument("-nw", "--num_workers", type=int, default=0)

    args = parser.parse_args()

    # Collect experiments
    # ===================
    if args.exp_id is not None:
        # select one experiment
        savedir = os.path.join(args.savedir_base, args.exp_id)
        exp_dict = hu.load_json(os.path.join(savedir, "exp_dict.json"))

        exp_list = [exp_dict]

    else:
        # select exp group
        exp_list = []
        for exp_group_name in args.exp_group_list:
            exp_list += exp_configs.EXP_GROUPS[exp_group_name]

    # Run experiments
    # ===============
    if not args.run_jobs:
        for exp_dict in exp_list:
            # do trainval
            trainval(exp_dict=exp_dict,
                    savedir_base=args.savedir_base,
                    datadir=args.datadir,
                    reset=args.reset,
                    num_workers=args.num_workers)
    else:
        # launch jobs
        from haven import haven_jobs as hjb
        import job_configs as jc
        
        jm = hjb.JobManager(exp_list=exp_list, 
                    savedir_base=args.savedir_base, 
                    account_id=jc.ACCOUNT_ID,
                    workdir=os.path.dirname(os.path.realpath(__file__)),
                    job_config=jc.JOB_CONFIG,
                    )

        command = ('python trainval.py -ei <exp_id> -sb %s -d %s -nw 2' %  
                  (args.savedir_base, args.datadir))
        print(command)
        jm.launch_menu(command=command)
        
