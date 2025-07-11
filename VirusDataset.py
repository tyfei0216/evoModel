import abc
import os
import random

import numpy as np
import pytorch_lightning as L
import torch
from esm.utils.constants import esm3 as C
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler

SEQ_VOCAB = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
]


class MyDataLoader(DataLoader):
    def __init__(self, ds, step_ds, *args, **kwargs):
        super().__init__(ds, *args, **kwargs)
        self.ds = ds
        self.epoch = 0
        self.step_ds = step_ds

    def step(self):
        # print("step dataset")
        if self.step_ds:
            self.ds.step()

    def __iter__(self):
        self.epoch += 1
        self.ds.newEpoch()
        if self.step_ds:
            # self.ds.step()
            self.ds.ifaug = True
        else:
            self.ds.ifaug = False
        # print("now epoch ", self.epoch)
        return super().__iter__()


MIN_LENGTH = 50


class DataAugmentation:
    def __init__(
        self,
        step_points: list,
        maskp: list = None,
        maskpc: list = None,
        crop: list = None,
        croprange: list = None,
        mutate: list = None,
        mutatep: list = None,
        vocab: list = None,
    ) -> None:
        self.augs = {}
        if maskp is not None:
            assert len(step_points) == len(maskp)
            # self.maskp = maskp
            self.augs["maskp"] = maskp

        if maskpc is not None:
            assert len(step_points) == len(maskpc)
            self.augs["maskpc"] = maskpc

        if crop is not None:
            assert len(step_points) == len(crop)
            # self.crop = crop
            self.augs["crop"] = crop

        if mutate is not None:
            assert len(step_points) == len(mutate)
            assert mutatep is not None
            self.augs["mutate"] = mutate

        if mutatep is not None:
            assert mutate is not None
            assert len(step_points) == len(mutatep)
            # self.mutatep = mutatep
            self.augs["mutatep"] = mutatep

        if vocab is None:
            self.vocab = SEQ_VOCAB
        else:
            self.vocab = vocab

        self.step_points = step_points

        self.croprange = croprange

    def _getSettings(self, step):
        ret = {}

        for i in self.augs:
            ret[i] = -1.0
        for i in range(len(self.step_points)):
            if step > self.step_points[i]:
                for j in self.augs:
                    ret[j] = self.augs[j][i]

        return ret

    def getAugmentationParameters(self, seqlen, step):
        ret = self._getSettings(step)
        if "mutate" in ret and ret["mutate"] > 0:
            t = random.random()
            if t < ret["mutate"]:
                ret["mutate"] = 1.0
            else:
                ret["mutate"] = 0.0
        if "crop" in ret and ret["crop"] > 0:
            t = random.random()
            if t < ret["crop"]:
                if self.croprange is not None:
                    sampledlen = random.sample(self.croprange, 1)[0]
                    sampledlen = int(sampledlen * np.random.uniform(0.8, 1.2))
                    sampledlen = MIN_LENGTH if sampledlen < MIN_LENGTH else sampledlen
                    sampledlen = min(sampledlen, seqlen - 2)
                    ret["crop"] = sampledlen
                else:
                    sampledlen = int(seqlen * np.random.uniform(0.3, 0.8))
                    sampledlen = MIN_LENGTH if sampledlen < MIN_LENGTH else sampledlen
                    sampledlen = min(sampledlen, seqlen - 2)
                    ret["crop"] = sampledlen
                return ret

        ret["crop"] = -1
        return ret


# deprecated
class BaseDataset2(Dataset):
    def __init__(
        self,
        data,
        required_labels=[],
        seq=["NSP5", "E", "S"],
        return_mask=True,
        mask=[0.25, 0.5, 1.0],
        maskp=0.1,
    ):
        self.data = data
        self.required_labels = required_labels
        self.seq = seq
        self.return_mask = return_mask
        self.mask = mask
        self.maskp = maskp

    def __len__(self):
        return len(self.data)

    def generateMask(self, ret):
        mask = {}
        # print(ret)
        for i in ret:
            mask[i] = np.ones_like(ret[i])
        return mask

    def maskSequence(self, ret, mask):
        t = random.random()
        if t < self.mask[0]:
            return
        if t < self.mask[1]:
            for i in self.seq:
                t = ret[i]
                num = np.random.binomial(len(t), self.maskp)
                if num > 0:
                    a = np.array(random.sample(range(len(t)), num))
                    t[a] = 0
                    # print(mask)
                    mask[i][a] = MASKED_TOKEN
            return
        t = random.sample(self.seq, 1)[0]
        ret[t] = np.zeros_like(ret[t])
        mask[t] = np.zeros_like(mask[t])

    def __getitem__(self, index):
        t = self.data[index]
        ret = {}
        for i in self.seq:
            ret[i] = t[i].copy()

        if self.return_mask:
            mask = self.generateMask(ret)
            self.maskSequence(ret, mask)
        else:
            mask = None

        for i in self.seq:
            ret["ori_" + i] = t[i].copy()

        labels = []
        for i in self.required_labels:
            labels.append(t[i])

        if len(labels) == 0:
            labels = np.array([])

        return ret, mask, labels


IGNORE_TOKEN = 2
MASKED_TOKEN = 0
UNMASKED_TOKEN = 1


class BaseDataset(Dataset):
    def __init__(
        self,
        seq: list[str],
        aug: DataAugmentation = None,
        required_labels=[],
    ) -> None:
        self.step_cnt = 0
        self.aug = aug
        self.required_labels = required_labels
        assert len(seq) > 0
        self.seq = seq

    @abc.abstractmethod
    def getToken(self, token):
        pass

    def _maskSequence(self, sample, mask, p, method="point", avoid=None):
        while True:
            num = np.random.binomial(len(sample) - 2, p)
            if len(sample) > num + 5:
                break
        pos = self._generateMaskingPos(num, len(sample), method, avoid)
        if len(pos) > 0:
            sample[pos] = self.getToken("mask")
            mask[pos] &= 2

        return sample, mask

    def _generateMaskingPos(self, num, length, method="point", exclude=None):
        assert length > num + 5
        if method == "point":
            t = list(range(1, length - 1))
            if exclude is not None:
                t = list(set(t) - set(exclude))
            num = min(num, len(t) - 10)
            if num <= 0:
                return []
            a = np.array(random.sample(t, num))
            return a
        elif method == "block":
            s = random.randint(1, length - num)
            a = np.array(range(s, s + num))
            return a
        else:
            raise NotImplementedError

    def _cropSequence(self, sample, crop_length):
        if len(sample) < crop_length + 2 or crop_length < MIN_LENGTH:
            return sample
        s = random.randint(1, len(sample) - crop_length - 1)
        start = s
        end = s + crop_length
        t = torch.zeros((end - start + 2), dtype=torch.long)
        t[1:-1] = torch.tensor(sample[start:end])
        t[0] = self.getToken("start")
        t[-1] = self.getToken("end")
        return t

    def _mutateSample(self, sample, p, avoid):
        num = np.random.binomial(len(sample) - 2, p)
        pos = self._generateMaskingPos(num, len(sample), exclude=avoid)
        for i in pos:
            t = sample[i]
            while t == sample[i]:
                t = self.getToken(random.sample(self.aug.vocab, 1)[0])
            sample[i] = t

        return sample, pos

    @abc.abstractmethod
    def _generateMask(self, sample):
        pass

    def _augmentSample(self, sample, aug_parameters, aligned_seq=None):
        ret = {}

        ret["parameters"] = aug_parameters

        if aligned_seq is not None:
            ret["aligned"] = aligned_seq
            ret["mutation_label"] = (aligned_seq != sample).astype(np.int64)
        else:
            ret["mutation_label"] = np.zeros_like(sample, dtype=np.int64)

        sample = sample.copy()

        # do not use crop
        sample = self._cropSequence(sample, aug_parameters["crop"])

        mask, avoid = self._generateMask(sample)
        sample, pos = self._mutateSample(sample, aug_parameters["mutatep"], avoid)

        ret["mutate_pos"] = pos

        if len(pos) > 0:
            ret["mutation_label"][pos] = 2

        avoid = np.concatenate([avoid, pos])

        ret["ori_seq"] = sample.copy()

        if aug_parameters["maskp"] > 0:
            sample, mask = self._maskSequence(
                sample, mask, aug_parameters["maskp"], "point", avoid
            )

        if aug_parameters["maskpc"] > 0:
            sample, mask = self._maskSequence(
                sample, mask, aug_parameters["maskpc"], method="block"
            )

        ret["mask"] = mask
        ret["sample"] = sample

        return ret

    def processSeq(self, sample, aligned_seq=None):
        if self.aug is not None and self.ifaug:
            aug_parameters = self.aug.getAugmentationParameters(
                len(sample), self.step_cnt
            )
            ret = self._augmentSample(sample, aug_parameters, aligned_seq)
        else:
            ret = {}
            ret["parameters"] = {}
            ret["sample"] = sample.copy()
            ret["ori_seq"] = sample.copy()
            ret["mutate_pos"] = []
            if aligned_seq is not None:
                ret["aligned"] = aligned_seq
                ret["mutation_label"] = (aligned_seq != sample).astype(np.int64)
            else:
                # ret["aligned"] = None
                ret["mutation_label"] = np.zeros_like(sample, dtype=np.int64)
            ret["mask"], _ = self._generateMask(sample)
        return ret

    def prepareLabels(self, sample):
        # if not isinstance(labels, list):
        #     labels = [labels]
        labels = []
        for i in self.required_labels:
            if "classes" in sample and i in sample["sample"]["classes"]:
                labels.append(sample["classes"][i])
            else:
                labels.append(-1)
        return labels


class VESMDataset(BaseDataset):
    def __init__(
        self,
        data,
        stage,
        seq: list[str],
        aug: DataAugmentation = None,
        sample_list=None,
        required_labels=[],
        shuffle=False,
        update_pnt=True,
        stage_2_maskp=0.2,
        train_time_series=True,
        ignore_token=["X", "<unk>", "<pad>", "|", "."],
    ) -> None:
        super().__init__(
            seq,
            aug,
            required_labels,
        )
        assert stage in [
            "pretraining stage 1",
            "training stage 1",
            "training stage 2",
            "training stage 1 + stage 2",
            "inference",
        ]

        if "stage 1" not in stage:
            self.ifaug = False
        else:
            self.ifaug = True

        self.train_time_series = train_time_series
        self.stage = stage
        self.data = data
        self.sample_list = sample_list
        self.shuffle = shuffle
        self.update_pnt = update_pnt

        if sample_list is not None:
            self.sample_list = sample_list
            self._sample = True
            for i in range(len(self.sample_list)):
                if len(self.data[i]) < self.sample_list[i] or self.sample_list[i] < 0:
                    self.sample_list[i] = len(self.data[i])
        else:
            self.sample_list = []
            self._sample = False
            for i in self.data:
                self.sample_list.append(len(i))

        self.data_order = []
        for i in data:
            self.data_order.append(np.arange(len(i)))

        self.pnts = [0 for _ in data]

        self.stage_2_maskp = stage_2_maskp

        self.ignore_token = []
        for i in ignore_token:
            self.ignore_token.append(C.SEQUENCE_VOCAB.index(i))

    def _generateMask(self, seq):

        mask = np.ones_like(seq, dtype=np.int32)
        pos = np.where(np.isin(seq, self.ignore_token))[0]
        mask[pos] = IGNORE_TOKEN
        return mask, pos

    def getToken(self, token, track="seq_t"):
        # assert token in ["start", "end", "mask"]
        match token:
            case "start":
                match track:
                    case "seq_t":
                        return C.SEQUENCE_BOS_TOKEN
                    case "structure_t":
                        return C.STRUCTURE_BOS_TOKEN
                    case "sasa_t":
                        return 0
                    case "second_t":
                        return 0
                    case _:
                        raise ValueError
            case "end":
                match track:
                    case "seq_t":
                        return C.SEQUENCE_EOS_TOKEN
                    case "structure_t":
                        return C.STRUCTURE_EOS_TOKEN
                    case "sasa_t":
                        return 0
                    case "second_t":
                        return 0
                    case _:
                        raise ValueError
            case "mask":
                match track:
                    case "seq_t":
                        return C.SEQUENCE_MASK_TOKEN
                    case "structure_t":
                        return C.STRUCTURE_MASK_TOKEN
                    case "sasa_t":
                        return C.SASA_UNK_TOKEN
                    case "second_t":
                        return C.SS8_UNK_TOKEN
                    case _:
                        raise ValueError("mask of %s is not found" % track)
            case "pad":
                match track:
                    case "seq_t":
                        return C.SEQUENCE_PAD_TOKEN
                    case "structure_t":
                        return C.STRUCTURE_PAD_TOKEN
                    case "sasa_t":
                        return C.SASA_PAD_TOKEN
                    case "second_t":
                        return C.SS8_PAD_TOKEN
                    case _:
                        raise ValueError
            case _:
                assert track == "seq_t"
                assert token in C.SEQUENCE_VOCAB
                return C.SEQUENCE_VOCAB.index(token)

    def __len__(self):
        if self._sample:
            t = 0
            for i in self.sample_list:
                t += i
            return t
        else:
            t = 0
            for i in self.data:
                t += len(i)
                return t

    def shuffleIndex(self):
        for i in self.data_order:
            random.shuffle(i)

    def newEpoch(self):
        if self.shuffle:
            self.shuffleIndex()
        if self.update_pnt:
            for i in range(len(self.sample_list)):
                self.pnts[i] += self.sample_list[i]
                while self.pnts[i] >= len(self.data_order[i]):
                    self.pnts[i] -= len(self.data_order[i])

    def step(self):
        self.step_cnt += 1

    def getSample(self, data, input_idx):
        sample = {}
        sample["input"] = {}
        sample["aligned"] = {}
        sample["meta"] = {}
        if "stage 2" in self.stage:
            for i in self.seq:
                d1 = data[i]
                d2 = data["aligned_" + i]
                if len(d1) < len(d2):
                    d2 = d2[: len(d1)]
                else:
                    d1 = d1[: len(d2)]
                sample["input"][i] = d1
                sample["aligned"][i] = d2

        else:
            q = self.seq[input_idx % len(self.seq)]
            d1 = data[q]
            d2 = data["aligned_" + q]
            if len(d1) < len(d2):
                d2 = d2[: len(d1)]
            else:
                d1 = d1[: len(d2)]
            sample["input"][q] = d1
            sample["aligned"][q] = d2
        for i in ["id", "bin", "days"]:
            sample["meta"][i] = data[i]
        return sample

    def _getitemx1(self, idx):
        ori_idx = idx
        if self._sample:
            for i in range(len(self.sample_list)):
                if idx - self.sample_list[i] < 0:
                    # sample = self.getSample(
                    #     self.data[i],
                    #     self.data_order[i][idx % len(self.data_order[i])],
                    #     ori_idx,
                    # )
                    sample = self.data[i][
                        self.data_order[i][idx % len(self.data_order[i])]
                    ]
                    sample = self.getSample(sample, ori_idx)
                    if self.train_time_series:
                        t = random.choice(range(len(self.data[i])))
                        # q = self.getSample(self.data[i], t, ori_idx)
                        q = self.data[i][t]
                        q = self.getSample(q, ori_idx)
                        if sample["meta"]["days"] > q["meta"]["days"]:
                            return q, sample
                        return sample, q

                    return sample

                else:
                    idx -= self.sample_list[i]
        else:
            for i in self.data:
                if idx - len(i) < 0:
                    sample = i[idx]
                    sample = self.getSample(i, ori_idx)
                    if self.train_time_series:
                        t = random.choice(range(len(i)))
                        q = i[t]
                        q = self.getSample(i, ori_idx)
                        if sample["meta"]["days"] > q["meta"]["days"]:
                            return q, sample
                        return sample, q
                    return sample
                else:
                    idx -= len(i)
        raise KeyError

    def processSample(self, t):
        x1 = {}
        x1["input"] = {}
        x1["ori_seq"] = {}

        if "stage 1" in self.stage:
            x1["stage_1_masks"] = {}
            x1["label"] = {}
            x1["mutation_label"] = {}
            for i in t["input"]:
                seq = t["input"][i]
                align_seq = t["aligned"][i]
                ret = self.processSeq(seq, aligned_seq=align_seq)
                x1["input"][i] = ret["sample"]
                x1["input"]["aligned_" + i] = ret["aligned"]
                x1["ori_seq"][i] = ret["ori_seq"]
                x1["stage_1_masks"][i] = ret["mask"]
                # print(ret)
                x1["mutation_label"][i] = ret["mutation_label"]
                # x1["label"][i] = np.array(len(ret["mutate_pos"]) > 0, dtype=int)
                x1["label"][i] = self.prepareLabels(t)
        else:
            # x1["stage_1_masks"] = None
            # x1["label"] = None
            for i in t["input"]:
                seq = t["input"][i]
                x1["input"][i] = seq
                x1["ori_seq"][i] = seq

        if "stage 2" in self.stage:
            x1["stage_2_masks"] = []
            for i in self.seq:
                r = np.random.uniform(0.001, 0.999)
                if r < self.stage_2_maskp:
                    x1["stage_2_masks"].append(i)
        # else:
        # x1["stage_2_masks"] = None

        x1["meta"] = t["meta"]
        return x1

    def __getitem__(self, idx):
        if self.train_time_series:
            t1, t2 = self._getitemx1(idx)
            x1 = self.processSample(t1)
            x2 = self.processSample(t2)
            return x1, x2
        else:
            t1 = self._getitemx1(idx)
            x1 = self.processSample(t1)
            return x1


class VESMDataModule(L.LightningDataModule):
    def __init__(
        self,
        data1,
        seq,
        stage,
        batch_size=1,
        sample_train=None,
        sample_val=None,
        train_test_ratio=[0.85, 0.15],
        aug=None,
        seed=1509,
        stage_2_maskp=0.2,
        required_labels=[],
        train_time_series=True,
        ignore_token=["X", "<unk>", "<pad>", "|", "."],
    ):
        super().__init__()
        assert stage in [
            "pretraining stage 1",
            "training stage 1",
            "training stage 2",
            "training stage 1 + stage 2",
            "inference",
        ]
        self.stage = stage
        self.seq = seq
        self.value = 0
        self.batch_size = batch_size
        self.seed = seed

        from sklearn.model_selection import train_test_split

        self.traindata1 = []
        self.train_indices1 = []
        self.valdata1 = []
        self.val_indices1 = []

        L.seed_everything(self.seed)

        for i in data1:
            d1, v1, i1, i2 = train_test_split(
                i,
                range(len(i)),
                train_size=train_test_ratio[0],
                random_state=self.seed,
            )
            self.traindata1.append(d1)
            self.valdata1.append(v1)
            self.train_indices1.append(i1)
            self.val_indices1.append(i2)

        self.train_set = VESMDataset(
            self.traindata1,
            stage,
            seq,
            aug=aug,
            sample_list=sample_train,
            stage_2_maskp=stage_2_maskp,
            required_labels=required_labels,
            train_time_series=train_time_series,
            ignore_token=ignore_token,
        )

        self.val_set = VESMDataset(
            self.valdata1,
            stage,
            seq,
            aug=aug,
            sample_list=sample_val,
            stage_2_maskp=stage_2_maskp,
            required_labels=required_labels,
            train_time_series=train_time_series,
            ignore_token=ignore_token,
        )

    def train_dataloader(self):
        self.value += 1
        print("get train loader")
        # return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=4)
        return MyDataLoader(
            self.train_set,
            True,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=4,
        )

    def val_dataloader(self):
        self.value += 1
        print("get val loader")
        return MyDataLoader(
            self.val_set,
            False,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=4,
        )

    def test_dataloader(self):
        self.value += 1
        print("get test loader")
        self.val_set.ifaug = False
        self.val_set.train_time_series = False
        return DataLoader(
            self.val_set,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=4,
        )
