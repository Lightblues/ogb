#%%
import pandas as pd
import shutil, os
import os.path as osp                   
import torch
import numpy as np
#%%
path = "/home/ec2-user/workspace/ogb/test/dataset/ogbn_mag/split/time/paper"
train_idx = pd.read_csv(osp.join(path, 'train.csv.gz'), compression='gzip', header = None).values.T[0]
# %%
path = "/home/ec2-user/workspace/ogb/test/dataset/ogbn_mag/raw/node-label/paper/node-label.csv.gz"
path_feat = "/home/ec2-user/workspace/ogb/test/dataset/ogbn_mag/raw/node-feat/paper/node-feat.csv.gz"
paper_df = pd.read_csv(osp.join(path), compression='gzip', header = None)
paper_df_feat = pd.read_csv(osp.join(path_feat), compression='gzip', header = None)
# %%
