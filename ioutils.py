import re
from datetime import date

import pandas as pd

re1 = re.compile(r"[*\s]")
re2 = re.compile(r"[-* Xx.\s]")
re3 = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYacdefghiklmnpqrstvwyXx -*.]*$")
re4 = re.compile(r"[^ACDEFGHIKLMNPQRSTVWYacdefghiklmnpqrstvwy]")


def getDate(startdate=(2019, 12, 1)):
    def x(x):
        t = x.split("-")
        if len(t) < 3:
            return -1
        t = (int(t[0]), int(t[1]), int(t[2]))
        if t[1] == 0:
            t = (t[0], 1, t[2])
        if t[2] == 0:
            t = (t[0], t[1], 1)
        d = date(int(t[0]), int(t[1]), int(t[2]))
        d2 = date(startdate[0], startdate[1], startdate[2])
        return (d - d2).days

    return x


def parseline(line, remove="all"):
    try:
        if isinstance(line, bytes):
            line = line.decode("utf-8").strip()
    except:
        return None
    if line.startswith(">"):
        return line
    if remove == "all":
        line = re4.sub("", line)
    elif remove == "whitespaces":
        line = re1.sub("", line)
    elif remove == "none":
        pass
    else:
        raise NotImplementedError
    return line


# inspired by esm.data.readfasta
def readFasta(path, to_upper=True, remove="whitespaces", truclength=1500, checkseq=re3):

    if checkseq is not None:
        assert isinstance(checkseq, re.Pattern)

    with open(path, "rb") as f:
        line = None
        seq = []
        for t in f:
            t = parseline(t, remove=remove)
            if t is None:
                line = None
                continue

            if line is None:
                if t.startswith(">"):
                    line = t[1:]
                continue

            if t.startswith(">"):
                s = "".join(seq)
                seq = []
                if len(s) > 0:
                    if to_upper:
                        s = s.upper()
                    if truclength is not None:
                        s = s[:truclength]
                    yield line, s
                line = t[1:]
                continue

            if checkseq is not None:
                if not checkseq.match(t):
                    line = None
                    seq = []
                    continue

            seq.append(t)

        s = "".join(seq)
        seq = []
        if len(s) > 0:
            if to_upper:
                s = s.upper()
            if truclength is not None:
                s = s[:truclength]
            yield line, s


def readNCBICsv(
    path, to_upper=True, remove="whitespaces", truclength=1500, checkseq=re3
):
    if checkseq is not None:
        assert isinstance(checkseq, re.Pattern)

    entries = []
    df = pd.read_csv(path, header=0)
    for i, r in df.iterrows():
        if len(str(r["translation"])) > 5:
            s = parseline(str(r["translation"]), remove=remove)
            if checkseq is not None:
                if not checkseq.match(s):
                    continue
            if to_upper:
                s = s.upper()
            if truclength is not None:
                s = s[:truclength]
            if "definition" in df.columns:
                defination = r["definition"]
            else:
                defination = ""
            entries.append(((r["protein_id"], defination, r["organism"]), s))

    return entries
