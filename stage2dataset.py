import argparse
import json
import os

import numpy as np
import torch

import trainUtils

with open(
    os.path.join(
        "/home/tyfei/evoModel/checkpoints/full_genome_smallwc/", "config.json"
    ),
    "r",
) as f:
    configs = json.load(f)

pretrain_model = trainUtils.loadPretrainModel(configs)
model = trainUtils.buildModel(configs, pretrain_model, None)

a = torch.load("/home/tyfei/evoModel/checkpoints/full_genome_smallwc/last-v1.ckpt")
model.load_state_dict(a["state_dict"], strict=False)

model.only_embed = True
model.tf = -1.0

import pickle

seqs = {}
for i in os.listdir("/data/tyfei/datasets/covid/"):
    if i.endswith("_paired_sequences.pkl"):
        with open("/data/tyfei/datasets/covid/" + i, "rb") as f:
            seqs[i.split("_")[0]] = pickle.load(f)

import random

from tqdm import tqdm

random.seed(1509)
a = random.sample(range(len(seqs["s"])), 100000)
res = []
with torch.no_grad():
    model = model.cuda(6)

    for i in tqdm(a):
        ret = {}
        for j in seqs.keys():
            NSP5 = seqs[j]
            t = NSP5[i]
            ret["ori_" + j] = t["seq_t"]
            t["seq_t"] = torch.tensor(t["seq_t"]).cuda(6)
            ret["id"] = i
            ret[j] = model(t).squeeze().cpu().numpy()
            # e["seq_t"] = torch.tensor(e["seq_t"]).cuda(2)
            # ret["E"] = model(e).squeeze().cpu().numpy()
            # nsp5["seq_t"] = torch.tensor(nsp5["seq_t"]).cuda(2)
            # ret["NSP5"] = model(nsp5).squeeze().cpu().numpy()
            res.append(ret)

with open("/data/tyfei/datasets/covid/teststage2_paired100000.pkl", "wb") as f:
    pickle.dump(res, f)
