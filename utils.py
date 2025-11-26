import os
import pickle

import torch
from tqdm import tqdm

import modules


def loadesm3(configs):
    from esm.models.esm3 import ESM3

    model = ESM3.from_pretrained("esm3_sm_open_v1").cpu()
    model = modules.fixParameters(
        model, unfix=configs["pretrain_model"]["unfix_layers"]
    )
    model = modules.addlora(
        model,
        layers=configs["pretrain_model"]["add_lora"],
        ranks=configs["pretrain_model"]["rank"],
        alphas=configs["pretrain_model"]["alpha"],
    )
    return model


def loadPickle(path):
    with open(
        path,
        "rb",
    ) as f:

        name = os.path.basename(path)
        data = pickle.load(f)
        for j in data:
            j["origin"] = name
            if "strcture_t" in j:
                j["structure_t"] = j.pop("strcture_t")
    return data


def loadDatasetesm3(configs):
    import VirusDataset

    data = loadPickle(configs["dataset"]["path"])

    step_points = configs["augmentation"]["step_points"]
    crop = configs["augmentation"]["crop"]
    maskp = [
        (i, j)
        for i, j in zip(
            configs["augmentation"]["maskp"], configs["augmentation"]["maskpc"]
        )
    ]
    aug = VirusDataset.DataAugmentation(
        step_points, maskp, crop, [], tracks=configs["dataset"]["tracks"]
    )

    # ds1 = VirusDataset.ESM3MultiTrackDataset(
    #     data1, data2, label, augment=aug, tracks=configs["dataset"]["tracks"]
    # )
    ds1 = VirusDataset.ESM3MultiTrackDatasetTEST(
        data, tracks=configs["dataset"]["tracks"], augment=aug
    )

    ds = VirusDataset.ESM3datamoduleSingle(ds1)
    return ds
