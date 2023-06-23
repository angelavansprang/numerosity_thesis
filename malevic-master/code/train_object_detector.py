import torch
import random
import utils
import models
from collections import defaultdict
import pickle
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import argparse
from torch.utils.data import DataLoader


def build_dataloader(dataset, layer, split="train", batch_size=10, patch_per_img=1):
    inputs = []
    targets = []

    repr_path = f"../data/{dataset}/representations/{dataset}_{split}_visual.pickle"

    print(f"Will try to open representations of {dataset} of split {split}")
    with open(repr_path, "rb") as f:
        repr = pickle.load(f)

    file_name_patches = (
        f"../data/{dataset}/representations/{dataset}_{split}_patches.pickle"
    )
    with open(file_name_patches, "rb") as f:
        objectpatches = pickle.load(f)

    for img_id, repr in repr.items():
        patches = objectpatches[img_id]
        nodes = patches.keys()

        black_patches = list(range(-1, 49))
        for key in nodes:
            black_patches.remove(key)
        black_patch_ids = random.sample(black_patches, patch_per_img)
        input = utils.filter_repr(
            layer, black_patch_ids, repr, single_patch=True, padding_up_to=None
        )
        for patch in input:
            inputs.append(patch)
            targets.append(0)
        # for black_id in black_patch_ids:
        #     input = utils.filter_repr(layer, [black_id], repr, single_patch=True)
        #     inputs.append(input)
        #     targets.append(0)

        object_patches = random.sample(nodes, patch_per_img)
        input = utils.filter_repr(
            layer, object_patches, repr, single_patch=True, padding_up_to=None
        )
        for patch in input:
            inputs.append(patch)
            targets.append(1)
        # for object_id in object_patches:
        #     input = utils.filter_repr(layer, [object_id], repr, single_patch=True)
        #     inputs.append(input)
        #     targets.append(1)

    dataset_train = list(zip(inputs, targets))
    # print("len dataset: ", len(dataset_train))

    if split == "train":
        dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    else:
        dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)

    return dataloader


def experiment_per_layer(
    dataset,
    modelname,
    patch_per_img,
    layernorm=False,
    save_models=False,
):
    D_out = 2
    results = defaultdict(lambda: [])

    for layer, size in utils.layer2size(padding_up_to=None, single_patch=True).items():
        print(f"Started with layer {layer} of size {size}")
        D_in = size

        loader_train = build_dataloader(
            dataset, layer, split="train", batch_size=10, patch_per_img=patch_per_img
        )
        loader_val = build_dataloader(
            dataset, layer, split="val", batch_size=10, patch_per_img=patch_per_img
        )
        loader_test = build_dataloader(
            dataset, layer, split="test", batch_size=10, patch_per_img=patch_per_img
        )

        model = utils.open_model(D_in, D_out, layernorm, modelname)
        trainer = pl.Trainer(
            accelerator="gpu",
            callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
            enable_progress_bar=False,
            log_every_n_steps=100,
        )
        train_info = trainer.fit(model, loader_train, loader_val)
        performance = trainer.test(dataloaders=loader_test)

        results[layer].append(performance[0]["acc"])
        if save_models:
            save_models_path = f'../models/{modelname}_layer{layer}_{dataset}_object_det{"_layernorm" if layernorm else ""}.pt'
            torch.save(model.state_dict(), save_models_path)
            print(f"Model layer {layer} saved")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a probe on a patch representation of ViT to detect whether it contains an object"
    )
    parser.add_argument("--dataset", choices=["sup1", "pos", "posmo", "sup1mo"])
    parser.add_argument("--modelname", choices=["linear_layer", "MLP", "MLP2"])
    parser.add_argument("--patch_per_img", type=int, default=1)
    parser.add_argument("--layernorm", action="store_true")
    parser.add_argument("--save_models", action="store_true")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = experiment_per_layer(
        args.dataset,
        args.modelname,
        args.patch_per_img,
        args.layernorm,
        args.save_models,
    )
    results_file = f'test_results_{args.modelname}_{args.dataset}_object_det{"_layernorm" if args.layernorm else ""}.pickle'

    results_path = "../results/"
    results_tosave = dict(results)
    with open(
        results_path + results_file,
        "wb",
    ) as f:
        pickle.dump(results_tosave, f)
