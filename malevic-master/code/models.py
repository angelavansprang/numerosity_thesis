import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F


class ProbingHead(pl.LightningModule):
    def __init__(self, D_in, D_out):
        super(ProbingHead, self).__init__()
        self.linear_layer = nn.Linear(D_in, D_out)

    def forward(self, x):
        return self.linear_layer(x.type(torch.float32))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        out = self.forward(x)
        loss = F.cross_entropy(out, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        out = self.forward(x)
        loss = F.cross_entropy(out, y)
        self.log("val_loss", loss)
        self.log("acc", torch.sum(torch.argmax(out, dim=1) == y).item() / x.shape[0])

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        out = self.forward(x)
        self.log("acc", torch.sum(torch.argmax(out, dim=1) == y).item() / x.shape[0])


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class MLP(pl.LightningModule):
    def __init__(self, D_out, width=768, output_dim=512, layernorm=False):
        """input:
        D_out (int): number of final classes
        width (int): input size of representations
        output_dim (int): output size of MLP

        NOTE: COPY OF ORIGINAL; a matrix of parameters works like a fully-connected layer, but no activation function and threshold value, so not really MLP?
        """
        super(MLP, self).__init__()
        # NOTE: LAYER NORMALIZATION IS ON
        self.layernorm = layernorm
        self.ln_post = LayerNorm(width)
        scale = width**-0.5
        self.proj = nn.Parameter(
            scale * torch.randn(width, output_dim)
        )  # original: a matrix of parameters works like a fully-connected layer, but no activation function and threshold value, so not really MLP?
        self.linear_layer = nn.Linear(output_dim, D_out)

    def forward(self, x):
        if self.layernorm:
            x = self.ln_post(x)
        z = x.type(torch.float32) @ self.proj
        out = self.linear_layer(z)
        # out = self.linear_layer(z.type(torch.float32))
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        out = self.forward(x)
        loss = F.cross_entropy(out, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        out = self.forward(x)
        loss = F.cross_entropy(out, y)
        self.log("val_loss", loss)
        self.log("acc", torch.sum(torch.argmax(out, dim=1) == y).item() / x.shape[0])

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        out = self.forward(x)
        self.log("acc", torch.sum(torch.argmax(out, dim=1) == y).item() / x.shape[0])


class MLP2(pl.LightningModule):
    def __init__(self, input_size, output_size, hidden_size=512, layernorm=False):
        """input:
        D_out (int): number of final classes
        width (int): input size of representations
        output_dim (int): output size of MLP
        """
        super(MLP2, self).__init__()
        self.layernorm = layernorm
        self.ln_post = LayerNorm(input_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        if self.layernorm:
            x = self.ln_post(x)
        z = self.fc1(x.type(torch.float32))
        z = self.gelu(z)
        out = self.fc2(z)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        out = self.forward(x)
        loss = F.cross_entropy(out, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        out = self.forward(x)
        loss = F.cross_entropy(out, y)
        self.log("val_loss", loss)
        self.log("acc", torch.sum(torch.argmax(out, dim=1) == y).item() / x.shape[0])

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        out = self.forward(x)
        self.log("acc", torch.sum(torch.argmax(out, dim=1) == y).item() / x.shape[0])
