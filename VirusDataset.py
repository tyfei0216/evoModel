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
        tracks: dict = {"seq_t": 1, "structure_t": 1, "sasa_t": 1, "second_t": 1},
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

        self.tracks = tracks
        # if self.tracks == None:
        #     self.tracks = {"seq_t": 1, "structure_t": 1, "sasa_t": 1, "second_t": 1}
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
        rettrack = {}
        flag = 0
        for i in self.tracks:

            r = np.random.uniform(0.001, 0.999)
            if r < self.tracks[i]:
                flag = 1
                rettrack[i] = True
            else:
                rettrack[i] = False

        if flag == 0:
            rettrack["seq_t"] = True
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
                return ret, rettrack

        ret["crop"] = -1
        return ret, rettrack


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


IGNORE_TOKEN = 3
MASKED_TOKEN = 0
UNMASKED_TOKEN = 1


class BaseDataset(Dataset):
    def __init__(
        self,
        tracks=["seq_t"],
        return_mask=False,
        aug: DataAugmentation = None,
        required_labels=[],
    ) -> None:
        assert len(tracks) > 0
        self.tracks = tracks
        self.step_cnt = 0
        self.return_mask = return_mask
        self.aug = aug
        self.ifaug = False
        self.required_labels = required_labels

    @abc.abstractmethod
    def getToken(self, track, token):
        pass

    def _maskSequence(self, sample, pos):
        for i in sample:
            if not i.startswith("ori"):
                sample[i][pos] = self.getToken(i, "mask")

        return sample

    def _generateMaskingPos(self, num, length, method="point"):
        assert length > num + 5
        if method == "point":
            a = np.array(random.sample(range(length - 2), num)) + 1
            return a
        elif method == "block":
            s = random.randint(1, length - num)
            a = np.array(range(s, s + num))
            return a
        else:
            raise NotImplementedError

    def _cropSequence(self, sample, start, end):
        for i in sample:
            t = torch.zeros((end - start + 2), dtype=torch.long)
            t[1:-1] = torch.tensor(sample[i][start:end])
            t[0] = self.getToken(i, "start")
            t[-1] = self.getToken(i, "end")
            sample[i] = t
        return sample

    def _mutateSample(self, sample, pos):
        for i in pos:
            sample["seq_t"][i] = self.getToken(
                "seq_t", random.sample(self.aug.vocab, 1)[0]
            )

        return sample

    def _augmentSample(self, sample, aug_parameters, tracks=None):
        ret = {}
        ret["parameters"] = aug_parameters
        ret["prot"] = sample["prot"]

        if tracks is None:
            tracks = self.tracks

        retsample = {}
        for i in tracks:
            retsample[i] = sample[i].copy()

        samplelen = len(sample[self.tracks[0]])

        if aug_parameters["crop"] > MIN_LENGTH:
            s = random.randint(1, samplelen - aug_parameters["crop"] - 1)
            retsample = self._cropSequence(retsample, s, s + aug_parameters["crop"])
            samplelen = aug_parameters["crop"] + 2

        ret["parameters"]["mutated"] = 0.0
        if "seq_t" in tracks and aug_parameters["mutate"] > 0.5:
            num = np.random.binomial(samplelen - 2, aug_parameters["mutatep"])
            pos = self._generateMaskingPos(num, samplelen)
            if len(pos) > 0:
                retsample = self._mutateSample(retsample, pos)
                ret["parameters"]["mutated"] = 1.0

        for i in tracks:
            retsample["ori_" + i] = retsample[i].copy()

        if self.return_mask:
            mask = self.generateMask(retsample)
            # mask = np.ones_like(retsample[self.tracks[0]], dtype=np.float32)

        if aug_parameters["maskp"] > 0:
            num = np.random.binomial(samplelen - 2, aug_parameters["maskp"])
            pos = self._generateMaskingPos(num, samplelen)
            if len(pos) > 0:
                retsample = self._maskSequence(retsample, pos)
            if self.return_mask:
                mask[pos] &= 2

        if aug_parameters["maskpc"] > 0:
            num = np.random.binomial(samplelen - 2, aug_parameters["maskpc"])
            pos = self._generateMaskingPos(num, samplelen, "block")
            if len(pos) > 0:
                retsample = self._maskSequence(retsample, pos)
                if self.return_mask:
                    mask[pos] &= 2

        if tracks is not None:
            for i in tracks:
                if not tracks[i]:
                    sample.pop(i)

        retsample["prot"] = sample["prot"]
        ret["sample"] = retsample
        if self.return_mask:
            ret["mask"] = mask

        return ret

    def generateMask(self, x):
        for i in x:
            if i.endswith("_t"):
                return np.ones_like(x[i], dtype=np.int32)
        raise ValueError

    def processSample(self, sample):
        if self.aug is not None and self.ifaug:
            aug_parameters, tracks = self.aug.getAugmentationParameters(
                len(sample[self.tracks[0]]), self.step_cnt
            )
            ret = self._augmentSample(sample, aug_parameters, tracks)
        else:
            ret = {}
            x1 = {}
            for i in self.tracks:
                x1[i] = sample[i]
                x1["ori_" + i] = sample[i]

            x1["prot"] = sample["prot"]
            ret["parameters"] = {}
            ret["sample"] = x1

            if self.return_mask:
                ret["mask"] = self.generateMask(x1)

        return ret

    def prepareLabels(self, sample, label):
        labels = [label]
        for i in self.required_labels:
            if "classes" in sample and i in sample["sample"]["classes"]:
                labels.append(sample["classes"][i])
            else:
                labels.append(-1)
        return labels


class BalancedDataset(BaseDataset):
    def __init__(
        self,
        data,
        tracks=["seq_t"],
        return_mask=False,
        aug: DataAugmentation = None,
        sample_list=None,
        required_labels=[],
        shuffle=False,
        update_pnt=True,
    ) -> None:
        super().__init__(
            tracks,
            return_mask,
            aug,
            required_labels,
        )
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

    @property
    def sample(self):
        return self._sample

    @sample.setter
    def sample(self, v: bool):
        assert isinstance(v, bool)

        self._sample = v

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
        # print("called new epoch")
        if self.shuffle:
            self.shuffleIndex()
        if self.update_pnt:
            for i in range(len(self.sample_list)):
                self.pnts[i] += self.sample_list[i]
                while self.pnts[i] >= len(self.data_order[i]):
                    self.pnts[i] -= len(self.data_order[i])

    def step(self):
        self.step_cnt += 1

    def resetCnt(self):
        self.step_cnt = 0

    def _getitemx1(self, idx):
        if self.sample:
            for i in range(len(self.sample_list)):
                if idx - self.sample_list[i] < 0:
                    return self.data[i][
                        self.data_order[i][idx % len(self.data_order[i])]
                    ]

                else:
                    idx -= self.sample_list[i]
        else:
            for i in self.data:
                if idx - len(i) < 0:
                    return i[idx], 1
                else:
                    idx -= len(i)
        raise KeyError

    def __getitem__(self, idx):
        x1 = {}
        t1 = self._getitemx1(idx)

        x1 = self.processSample(t1)
        if "mutated" in x1["parameters"]:
            label = x1["parameters"]["mutated"]
        else:
            label = 0.0
        labels = self.prepareLabels(t1, label)

        s = x1["sample"]
        # s["prot"] = x1["prot"]
        s["mask"] = x1["mask"]

        return s, labels


class ESM3MultiTrackAutoEncoderDataset(BalancedDataset):
    def __init__(
        self,
        data,
        augment: DataAugmentation = None,
        sample_list=None,
        tracks=["seq_t", "structure_t", "sasa_t", "second_t"],
        return_mask=True,
        required_labels=[],
        shuffle=False,
        update_pnt=True,
        ignore_token=["X", "<unk>", "<pad>", "|", "."],
    ) -> None:
        super().__init__(
            data=data,
            tracks=tracks,
            return_mask=return_mask,
            aug=augment,
            sample_list=sample_list,
            required_labels=required_labels,
            shuffle=shuffle,
            update_pnt=update_pnt,
        )
        self.ignore_token = []
        for i in ignore_token:
            self.ignore_token.append(C.SEQUENCE_VOCAB.index(i))

    def generateMask(self, x):

        ret = super().generateMask(x)
        if "seq_t" in x:
            q = x["seq_t"]
            for i in range(len(q)):
                if q[i] in self.ignore_token:
                    ret[i] = IGNORE_TOKEN
        return ret

    def getToken(self, track, token):
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


class ESM3MultiTrackDataset(BaseDataset):

    def __init__(
        self,
        data1,
        data2,
        label,
        augment: DataAugmentation = None,
        tracks=["seq_t", "structure_t", "sasa_t", "second_t"],
        # origin = {0:""},
    ) -> None:
        super().__init__(tracks=tracks)
        self.data1 = data1
        self.data2 = data2
        self.label = label
        self.aug = augment
        self.iters = 0
        self.data2order = np.arange(len(data2))
        random.shuffle(self.data2order)
        self.ifaug = False
        # self.tracks = tracks

    def __len__(self):
        return len(self.data1)

    def newEpoch(self):
        random.shuffle(self.data2order)

    # def step(self):
    # random.shuffle(self.data2order)
    # super().step()

    def __getitem__(self, idx):
        x1 = {}
        x2 = {}
        for i in self.tracks:
            x1[i] = self.data1[idx][i]
            x2[i] = self.data2[self.data2order[idx % len(self.data2)]][i]
        if self.aug is not None and self.ifaug:
            maskp, crop = self.aug.getAugmentation(
                len(x1[self.tracks[0]]), self.step_cnt
            )
            x1 = self._augmentsample(x1, maskp, crop)
        return x1, torch.tensor([self.label[idx]]), x2


class ESM3BalancedDataModule(L.LightningDataModule):
    def __init__(
        self,
        data1,
        batch_size=1,
        sample_train=None,
        sample_val=None,
        train_test_ratio=[0.85, 0.15],
        aug=None,
        seed=1509,
        tracks=["seq_t", "structure_t", "sasa_t", "second_t"],
        required_labels=[],
    ):
        super().__init__()
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

        self.train_set = ESM3MultiTrackAutoEncoderDataset(
            self.traindata1,
            augment=aug,
            sample_list=sample_train,
            tracks=tracks,
            required_labels=required_labels,
        )

        self.val_set = ESM3MultiTrackAutoEncoderDataset(
            self.valdata1,
            augment=aug,
            sample_list=sample_val,
            tracks=tracks,
            required_labels=required_labels,
        )

    def train_dataloader(self):
        self.value += 1
        print("get train loader")
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


class Stage2DataModule(L.LightningDataModule):
    def __init__(
        self,
        data,
        batch_size=1,
        validate_size=1000,
        seed=1509,
        seq=["NSP5", "E", "S"],
        required_labels=[],
        mask=[0.25, 0.5, 1.0],
        maskp=0.1,
    ):
        super().__init__()
        # print(len(data))
        self.batch_size = batch_size
        self.seed = seed

        from sklearn.model_selection import train_test_split

        # self.traindata = []
        # self.train_indices = []
        # self.valdata = []
        # self.val_indices = []
        # L.seed_everything(self.seed)

        self.traindata, self.valdata, self.train_indices, self.val_indices = (
            train_test_split(
                data,
                range(len(data)),
                test_size=validate_size,
                random_state=self.seed,
            )
        )

        self.train_set = BaseDataset2(
            self.traindata,
            required_labels=required_labels,
            seq=seq,
            mask=mask,
            maskp=maskp,
        )

        self.val_set = BaseDataset2(
            self.valdata,
            required_labels=required_labels,
            seq=seq,
            mask=mask,
            maskp=maskp,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size, num_workers=4, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set, batch_size=self.batch_size, num_workers=4, shuffle=False
        )
