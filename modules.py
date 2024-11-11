import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtune


def fixParameters(esm_model, unfix=["9", "10", "11"]):
    for i, j in esm_model.named_parameters():
        flag = 1
        for k in unfix:
            if k in i:
                flag = 0

        if flag == 1:
            j.requires_grad = False
        else:
            j.requires_grad = True

    return esm_model


class SelfAttention(nn.Module):
    def __init__(self, channels, n_head):
        super(SelfAttention, self).__init__()
        self.channels = channels
        # self.size = size
        self.n_head = n_head
        assert channels % n_head == 0
        self.rope = torchtune.modules.RotaryPositionalEmbeddings(channels // n_head)
        self.mha = nn.MultiheadAttention(channels, n_head, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        # x = x.swapaxes(1, 2)
        batch, length, channel = x.shape
        x = x.view(batch, length, self.n_head, self.channels // self.n_head)
        # print(x.shape)
        # print(self.rope(x).shape)
        x = self.rope(x)
        # print(x.shape)
        x = x.view(batch, length, self.channels)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value  # .swapaxes(2, 1)


class DecoderBlock(nn.Module):
    def __init__(self, channels, n_head, classes):
        super(DecoderBlock, self).__init__()
        self.channels = channels
        self.n_head = n_head
        self.classes = classes
        self.T1 = SelfAttention(channels, n_head)
        self.T2 = SelfAttention(channels, n_head)
        self.T3 = SelfAttention(channels, n_head)
        self.clf = nn.Linear(channels, classes)

    def forward(self, x):
        x = self.T1(x)
        x = self.T2(x)
        x = self.T3(x)
        x = self.clf(x)
        return x


class Linearcls(nn.Module):
    """simple linear classifier

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self, input_dim=1536, take_embed="first", dropout=-1, p0=None, output_dim=1
    ):
        super().__init__()

        assert take_embed in ["first", "mean", "max"]
        self.embed_dim = input_dim
        self.dropout = dropout
        self.take_embed = take_embed

        self.l1 = nn.Linear(self.embed_dim, self.embed_dim // 2)
        self.l2 = nn.Linear(self.embed_dim // 2, self.embed_dim // 4)
        self.l3 = nn.Linear(self.embed_dim // 4, output_dim)
        self.ln1 = nn.LayerNorm(self.embed_dim // 2)
        self.ln2 = nn.LayerNorm(self.embed_dim // 4)
        if p0 is None:
            self.p0 = None
        else:
            self.p0 = nn.Dropout(p0)
        if dropout > 0 and dropout < 1:
            self.dropout1 = nn.Dropout(p=self.dropout)
            self.dropout2 = nn.Dropout(p=self.dropout)
        else:
            self.dropout1 = None
            self.dropout2 = None

    def forward(self, x: torch.Tensor):

        if self.take_embed == "first":
            x = x[:, 0]
        elif self.take_embed == "mean":
            x = torch.mean(x, dim=1)
        elif self.take_embed == "max":
            x = x.transpose(1, 2)
            x = F.adaptive_max_pool1d(x, 1)

        if self.p0 is not None:
            x = self.p0(x)

        x = self.l1(x)
        x = self.ln1(x)
        if self.dropout1 is not None:
            x = self.dropout1(x)
        x = F.gelu(x)
        x = self.l2(x)
        x = self.ln2(x)
        if self.dropout2 is not None:
            x = self.dropout2(x)
        x = F.gelu(x)
        x = self.l3(x)
        # print("lin", x.shape)
        return x


class CrossGeneModel(L.LightningModule):
    def __init__(
        self,
        in_channels=256,
        transformer_layers=5,
        n_head=16,
        seq=["Mprot", "E", "Spike"],
        weight_decay=0.0,
        lr=1e-4,
    ):
        super().__init__()

        self.seq = seq
        self.weight_decay = weight_decay
        self.lr = lr

        self.encoder_blocks = nn.ModuleList(
            [SelfAttention(in_channels, n_head) for i in range(transformer_layers)]
        )

        self.decoder_block = nn.ModuleList(
            [SelfAttention(in_channels, n_head) for i in range(transformer_layers)]
        )

        self.cls = Linearcls(in_channels)

    def forward(self, x):
        inputs = []
        for i in self.seq:
            inputs.append(x[i])
        inputs.append(torch.zeros_like(inputs[-1]))

        inputs = torch.stack(inputs, dim=1)

        for block in self.encoder_blocks:
            inputs = block(inputs)

        clsres = self.cls(inputs)


class AutoEncoder(L.LightningModule):
    def __init__(
        self,
        esm_model,
        in_channels=1536,
        out_channels=256,
        n_head=16,
        lr=1e-4,
        only_embed=True,
        weight_decay=0.0,
        classes=33,
        masked_weight=0.1,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["esm_model"])
        self.esm_model = esm_model
        self.bottleneck = nn.Linear(in_channels, out_channels)

        self.decoder = DecoderBlock(out_channels, n_head, classes)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_head = n_head
        self.lr = lr
        self.weight_decay = weight_decay
        self.cri = nn.CrossEntropyLoss(reduction="none")
        self.only_embed = only_embed
        self.masked_weight = masked_weight

        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, input_dict):
        for i in ["seq_t", "structure_t", "ss8_t", "sasa_t"]:
            if i not in input_dict:
                input_dict[i] = None
            else:
                if len(input_dict[i].size()) == 1:
                    input_dict[i] = input_dict[i].unsqueeze(0)

        representations = self.esm_model(
            sequence_tokens=input_dict["seq_t"],
            structure_tokens=input_dict["structure_t"],
            ss8_tokens=input_dict["ss8_t"],
            sasa_tokens=input_dict["sasa_t"],
        )

        x = representations.embeddings

        batchsize, length, channels = x.shape

        embed = x[:, 0]

        embed = self.bottleneck(embed)

        if self.only_embed:
            return embed

        x = embed[:, None, :].repeat(1, length, 1)
        x = self.decoder(x)
        return embed, x

    def _common_training_step(self, input_dict, y, mask):
        self.only_embed = False
        _, x = self.forward(input_dict)
        x = x.view(-1, x.shape[-1])
        y = y.flatten()
        mask = mask.flatten()
        mask = 1 - mask
        mask += self.masked_weight
        loss = self.cri(x, y)
        # print(loss.shape, mask.shape)
        loss = loss * mask
        # print(loss.shape)
        loss = loss.sum() / mask.sum()
        # print(loss.shape, loss)
        # exit()
        return loss

    def training_step(self, batch, batch_idx):
        input_dict, y, mask = batch
        loss = self._common_training_step(input_dict, y, mask)
        self.training_step_outputs.append(loss.detach().cpu())
        self.log("train_loss:", loss, prog_bar=True)
        return loss

    def _common_epoch_end(self, outputs):

        if len(outputs) == 0:
            return 0
        loss = torch.stack(outputs).mean()

        outputs.clear()
        return loss

    def on_training_epoch_end(self):

        loss = self._common_epoch_end(self.training_step_outputs)

        print("finish training epoch, loss %f" % loss)
        self.log_dict(
            {
                "train_loss": loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.last_train_step = 0

    def on_validation_epoch_end(self):
        loss = self._common_epoch_end(self.validation_step_outputs)
        print("finish validating, loss %f" % (loss))
        self.log_dict(
            {
                "validate_loss": loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

    def validation_step(self, batch, batch_idx):
        input_dict, y, mask = batch
        loss = self._common_training_step(input_dict, y, mask)
        self.validation_step_outputs.append(loss.detach().cpu())
        return loss

    def on_save_checkpoint(self, checkpoint):
        backbones = []
        for i in checkpoint["state_dict"]:
            if "esm" in i and "lora" not in i:
                backbones.append(i)
        for i in backbones:
            del checkpoint["state_dict"][i]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return optimizer


class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim) * std_dev)
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x


class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x):
        return self.linear(x) + self.lora(x)


def _set_submodule(submodule, module_path, new_module):
    tokens = module_path.split(".")
    for token in tokens[:-1]:
        submodule = getattr(submodule, token)
    setattr(submodule, tokens[-1], new_module)


def addlora(esm_model, layers, ranks, alphas):
    # if layers is None:
    #     layers = [str(i) for i in range(12)]
    for i, j in esm_model.named_modules():
        if isinstance(j, nn.Linear):
            # print(i)
            # res = [False]
            # res.extend([t in i for t in layers])
            # res = reduce(lambda x, y: x or y, res)
            for layer, rank, alpha in zip(layers, ranks, alphas):
                if str(layer) in i:
                    _set_submodule(
                        esm_model,
                        i,
                        LinearWithLoRA(j, rank, alpha),
                    )
    return esm_model
