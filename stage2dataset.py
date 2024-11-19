import argparse
import json
import os

import numpy as np
import torch

import trainUtils

with open(
    os.path.join("/home/tyfei/evoModel/checkpoints/backbone_lr/", "config.json"), "r"
) as f:
    configs = json.load(f)

pretrain_model = trainUtils.loadPretrainModel(configs)
model = trainUtils.buildModel(configs, pretrain_model, None)

a = torch.load("/home/tyfei/evoModel/checkpoints/backbone_lr/last.ckpt")
model.load_state_dict(a["state_dict"], strict=False)

model.only_embed = True
model.tf = -1.0

import pickle

with open("/data/tyfei/datasets/covid/S_aln.pkl", "rb") as f:
    S = pickle.load(f)
with open("/data/tyfei/datasets/covid/NSP5_aln.pkl", "rb") as f:
    NSP5 = pickle.load(f)
with open("/data/tyfei/datasets/covid/E_aln.pkl", "rb") as f:
    E = pickle.load(f)

import random

from tqdm import tqdm

random.seed(1509)
a = random.sample(range(len(S)), 80000)
res = []
with torch.no_grad():
    model = model.cuda(5)
    # for s, e, nsp5 in tqdm(zip(S, E, NSP5)):
    for i in tqdm(a):
        ret = {}
        t = NSP5[i]
        t["seq_t"] = torch.tensor(t["seq_t"]).cuda(5)
        ret["id"] = i
        ret["NSP5"] = model(t).squeeze().cpu().numpy()
        # e["seq_t"] = torch.tensor(e["seq_t"]).cuda(2)
        # ret["E"] = model(e).squeeze().cpu().numpy()
        # nsp5["seq_t"] = torch.tensor(nsp5["seq_t"]).cuda(2)
        # ret["NSP5"] = model(nsp5).squeeze().cpu().numpy()
        res.append(ret)

with open("/data/tyfei/datasets/covid/teststage2NSP5_lr_backbone_.pkl", "wb") as f:
    pickle.dump(res, f)
