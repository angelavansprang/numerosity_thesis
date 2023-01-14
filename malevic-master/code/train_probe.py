import torch
import random
import utils
import models
from collections import defaultdict
import pickle
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import argparse


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

    results = defaultdict(lambda: [])

    for layer, size in utils.layer2size.items():
        print(f"Started with layer {layer} of size {size}")
        D_in = size

        loader_train, class2label = utils.build_dataloader(
            dataset, layer, split="train", balanced=balanced, objective=objective
        )
        loader_val, class2label = utils.build_dataloader(
            dataset, layer, split="val", balanced=balanced, objective=objective
        )
        loader_test, class2label = utils.build_dataloader(
            dataset, layer, split="test", balanced=balanced, objective=objective
        )

        for i in range(5):
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
                save_models_path = f'../models/{modelname}_layer{layer}_{i}_{dataset}_{objective}_{"balanced" if balanced else "unbalanced"}_{"layernorm" if layernorm else "no_layernorm"}.pt'
                torch.save(model.state_dict(), save_models_path)

    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a probe on representations ViT")
    parser.add_argument("--dataset", choices=["sup1", "pos"], required=True)
    parser.add_argument("--objective", choices=["n_colors", "n_objects"], required=True)
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--modelname", choices=["linear_layer", "MLP", "MLP2"])
    parser.add_argument("--layernorm", action="store_true")
    parser.add_argument("--save_models", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = experiment_per_layer(
        args.objective,
        args.dataset,
        modelname=args.modelname,
        balanced=args.balanced,
        layernorm=args.layernorm,
        save_models=args.save_models,
    )

    results_path = "../results/"

    results_tosave = dict(results)
    with open(
        results_path
        + f'test_results_{args.modelname}_{args.dataset}_{args.objective}_{"balanced" if args.balanced else "unbalanced"}_{"layernorm" if args.layernorm else "no_layernorm"}.pickle',
        "wb",
    ) as f:
        pickle.dump(results_tosave, f)
