import torch
import random
from utils import *
from torch.utils.data import DataLoader
from collections import defaultdict
import pickle
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


# Use this dictionary to find the number of layers necessary for the linear probe,
# (the input size of the linear probe depends on the size of the representations)
layer2size = {
    0: 50 * 768,
    1: 50 * 768,
    2: 50 * 768,
    3: 50 * 768,
    4: 50 * 768,
    5: 50 * 768,
    6: 50 * 768,
    7: 50 * 768,
    8: 50 * 768,
    9: 50 * 768,
    10: 50 * 768,
    11: 50 * 768,
    12: 50 * 768,
    13: 50 * 768,
    14: 50 * 768,
    15: 768,
    16: 512,
}


def build_dataloader(
    dataset, layer, split="train", balanced=True, objective="n_objects", batch_size=10
):
    """Return dataloaders with the (visual) CLIP representations as data and labels whether
    the text and image match. Only add data to the dataloaders if the maximum for
    that class is not yet reached.
    NOTE: REQUIRES THAT THE REPRESENTATIONS ARE ALREADY MADE

    Input:
    ids_val (list): contains ints of the IDs of the images that should be in the validation set.
    balanced (bool): whether the dataloaders should be made balanced (i.e. same number of instances per class)
    objective (string): either 'n_objects' or 'n_colors'

    TODO: IMPLEMENT BALANCED == TRUE. Current implementation not yet working, due to mismatch in shapes (in both MLP and MLP2)
    Try new approach using the representations already present, and selecting the correct ones based on selection function from utils.py
    """

    class2label, label2class = get_classlabel(dataset, split=split, objective=objective)
    labels = make_labels_dict(dataset, split=split)
    # if balanced:
    #     repr_path = f"../data/{dataset}/representations/{dataset}_{split}_balanced_{objective}_visual.pickle"
    # else:
    #     repr_path = f"../data/{dataset}/representations/{dataset}_{split}_visual.pickle"
    repr_path = f"../data/{dataset}/representations/{dataset}_{split}_visual.pickle"

    print(f"Will try to open representations of {dataset} of split {split}")
    print(f"Balanced is: {balanced}")
    with open(repr_path, "rb") as f:
        repr = pickle.load(f)

    inputs = []
    targets = []

    if balanced:
        balanced_labels, _ = make_balanced_data(labels, objective)

    for img_id, repr in repr.items():
        if balanced:
            if int(img_id) in balanced_labels.keys():
                input = repr[layer].flatten()  # flatten the representations!
                label = int(labels[int(img_id)][objective])
                label = label2class[label]
                inputs.append(input)
                targets.append(label)
        else:
            input = repr[layer].flatten()  # flatten the representations!
            label = int(labels[int(img_id)][objective])
            label = label2class[label]
            inputs.append(input)
            targets.append(label)

    dataset_train = list(zip(inputs, targets))
    print("len dataset: ", len(dataset_train))

    if split == "train":
        dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    else:
        dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)

    return dataloader, class2label


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
        scale = width ** -0.5
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

        NOTE: COPY OF ORIGINAL; a matrix of parameters works like a fully-connected layer, but no activation function and threshold value, so not really MLP?
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


def experiment_per_layer(
    objective, dataset, modelname, balanced=True, layernorm=False, save_models=False
):
    """#TODO: think about and change the option to save the trained models. This can be helpful
    for testing, but then you should also store the trainers? maybe just save them as pickle files?
    Then, need to include the path as argument?
    """
    if objective == "n_colors":
        D_out = 4
    elif objective == "n_objects":
        D_out = 5

    # if save_models:
    #     models = defaultdict(lambda: [])

    results = defaultdict(lambda: [])

    for layer, size in tqdm(layer2size.items()):
        print(f"Started with layer {layer} of size {size}")
        D_in = size

        loader_train, class2label = build_dataloader(
            dataset, layer, split="train", balanced=balanced, objective=objective
        )
        loader_val, class2label = build_dataloader(
            dataset, layer, split="val", balanced=balanced, objective=objective
        )
        loader_test, class2label = build_dataloader(
            dataset, layer, split="test", balanced=balanced, objective=objective
        )

        for i in range(5):
            if modelname == "linear_layer":
                model = ProbingHead(D_in, D_out)
            elif modelname == "MLP":
                model = MLP(D_out=D_out, width=D_in, layernorm=layernorm)
            elif modelname == "MLP2":
                model = MLP2(input_size=D_in, output_size=D_out, layernorm=layernorm)
            trainer = pl.Trainer(
                accelerator="gpu",
                callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
                enable_progress_bar=False,
            )
            train_info = trainer.fit(model, loader_train, loader_val)
            # performance = trainer.validate(model, loader_val)
            performance = trainer.test(dataloaders=loader_test)

            # performance = train_model(model, loader_train, loader_val)
            results[layer].append(performance[0]["acc"])
            if save_models:
                save_models_path = f'../models/{modelname}_{i}_{dataset}_{objective}_{"balanced" if balanced else "unbalanced"}_{"layernorm" if layernorm else "no_layernorm"}.pt'
                torch.save(model.state_dict(), save_models_path)

    return results


def train_model(model, loader_train, loader_val):
    trainer = pl.Trainer(
        accelerator="gpu",
        callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
        enable_progress_bar=False,
    )
    train_info = trainer.fit(model, loader_train, loader_val)
    performance = trainer.validate(model, loader_val)
    return performance


def test_model(model, loader_test):
    trainer = pl.Trainer(
        accelerator="gpu",
        callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
        enable_progress_bar=False,
    )
    performance = trainer.test(model, dataloaders=loader_test)
    return performance


def test_experiment_per_layer(objective, dataset, models, balanced):
    results = defaultdict(lambda: [])
    for layer, _ in tqdm(layer2size.items()):
        loader_test, _ = build_dataloader(
            dataset, layer, split="test", balanced=balanced, objective=objective
        )
        for model in models[layer]:
            performance = test_model(model, loader_test)
            results[layer].append(performance[0]["acc"])
    return results


if __name__ == "__main__":

    # parameters
    dataset = "sup1"
    objective = "n_colors"
    balanced = False
    modelname = "MLP"
    layernorm = False

    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = experiment_per_layer(
        objective,
        dataset,
        modelname=modelname,
        balanced=balanced,
        layernorm=layernorm,
        save_models=True,
    )

    # results = test_experiment_per_layer(objective, dataset, models, balanced)

    # print(results)
    # TODO: check how results look like after testing, and change it so that I can make a confusion matrix

    results_path = "../results/"

    results_tosave = dict(results)
    with open(
        results_path
        + f'test_results_{modelname}_{dataset}_{objective}_{"balanced" if balanced else "unbalanced"}_{"layernorm" if layernorm else "no_layernorm"}.pickle',
        "wb",
    ) as f:
        pickle.dump(results_tosave, f)
