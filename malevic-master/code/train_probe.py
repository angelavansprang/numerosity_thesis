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
    objective,
    dataset,
    modelname,
    balanced=True,
    layernorm=False,
    save_models=False,
    padding_up_to=None,
    single_patch=False,
    amnesic_obj=None,
    single_experiment=False,
):
    """#TODO: think about and change the option to save the trained models. This can be helpful
    for testing, but then you should also store the trainers? maybe just save them as pickle files?
    Then, need to include the path as argument?
    """
    if single_experiment:
        n_runs = 1
    else:
        n_runs = 5

    if objective == "n_colors":
        D_out = 4
        # D_out = 5
    elif objective == "n_objects":
        D_out = 5
    elif objective == "shape":
        # D_out = 5
        D_out = 4
        single_patch = True
    elif objective == "color":
        D_out = 5
        single_patch = True

    results = defaultdict(lambda: [])

    for layer, size in utils.layer2size(padding_up_to, single_patch).items():
        print(f"Started with layer {layer} of size {size}")
        D_in = size

        if objective == "n_colors" or objective == "n_objects":
            loader_train, class2label = utils.build_dataloader(
                dataset,
                layer,
                split="train",
                balanced=balanced,
                objective=objective,
                padding_up_to=padding_up_to,
                single_patch=single_patch,
            )
            loader_val, class2label = utils.build_dataloader(
                dataset,
                layer,
                split="val",
                balanced=balanced,
                objective=objective,
                padding_up_to=padding_up_to,
                single_patch=single_patch,
            )
            loader_test, class2label = utils.build_dataloader(
                dataset,
                layer,
                split="test",
                balanced=balanced,
                objective=objective,
                padding_up_to=padding_up_to,
                single_patch=single_patch,
            )
        elif objective == "shape" or objective == "color":
            loader_train, class2label = utils.build_dataloader_patchbased(
                dataset,
                layer,
                objective,
                split="train",
                balanced=balanced,
                threshold=padding_up_to,
                amnesic_obj=amnesic_obj,
            )
            loader_val, class2label = utils.build_dataloader_patchbased(
                dataset,
                layer,
                objective,
                split="val",
                balanced=balanced,
                threshold=padding_up_to,
                amnesic_obj=amnesic_obj,
            )
            loader_test, class2label = utils.build_dataloader_patchbased(
                dataset,
                layer,
                objective,
                split="test",
                balanced=balanced,
                threshold=padding_up_to,
                amnesic_obj=amnesic_obj,
            )

        for i in range(n_runs):
            print("D_in: ", D_in)
            print("D_out: ", D_out)
            for batch, labels in loader_train:
                print(batch[0].shape)
                break
            model = utils.open_model(D_in, D_out, layernorm, modelname)
            trainer = pl.Trainer(
                accelerator="gpu",
                callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
                enable_progress_bar=False,
                log_every_n_steps=100,
            )
            print("NO PROBLEM BEFORE TRAINING")
            train_info = trainer.fit(model, loader_train, loader_val)
            performance = trainer.test(dataloaders=loader_test)

            results[layer].append(performance[0]["acc"])
            if save_models:
                save_models_path = f'../models/{modelname}_layer{layer}_{i}_{dataset}_{objective}_{"balanced" if balanced else "unbalanced"}_{"filtered" if filter else "unfiltered"}{"_single_patch" if single_patch else ""}_{"layernorm" if layernorm else "no_layernorm"}.pt'
                torch.save(model.state_dict(), save_models_path)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a probe on representations ViT")
    parser.add_argument("--dataset", choices=["sup1", "pos"], required=True)
    parser.add_argument(
        "--objective",
        choices=["n_colors", "n_objects", "shape", "color"],
        required=True,
    )
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--modelname", choices=["linear_layer", "MLP", "MLP2"])
    parser.add_argument("--layernorm", action="store_true")
    parser.add_argument("--save_models", action="store_true")
    parser.add_argument("--padding_up_to", type=int, default=None)
    parser.add_argument("--single_patch", action="store_true")
    parser.add_argument("--amnesic_obj", choices=["shape", "color"], default=None)
    parser.add_argument("--single_experiment", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = experiment_per_layer(
        args.objective,
        args.dataset,
        modelname=args.modelname,
        balanced=args.balanced,
        layernorm=args.layernorm,
        save_models=args.save_models,
        padding_up_to=args.padding_up_to,
        single_patch=args.single_patch,
        amnesic_obj=args.amnesic_obj,
        single_experiment=args.single_experiment,
    )

    results_path = "../results/"

    results_tosave = dict(results)
    with open(
        results_path
        + f'test_results_{args.modelname}_{args.dataset}_{args.objective}_{"balanced" if args.balanced else "unbalanced"}_{"filtered_" + str({args.padding_up_to}) if args.padding_up_to is not None else "unfiltered"}{"_single_patch" if args.single_patch else ""}_{"layernorm" if args.layernorm else "no_layernorm"}{"_amnesic" + str(args.amnesic_obj) if args.amnesic_obj is not None else ""}.pickle',
        "wb",
    ) as f:
        pickle.dump(results_tosave, f)
