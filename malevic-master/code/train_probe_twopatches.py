import torch
import random
import utils
import models
from collections import defaultdict
import pickle
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import argparse


layer2size = {
    0: 2 * 768,
    1: 2 * 768,
    2: 2 * 768,
    3: 2 * 768,
    4: 2 * 768,
    5: 2 * 768,
    6: 2 * 768,
    7: 2 * 768,
    8: 2 * 768,
    9: 2 * 768,
    10: 2 * 768,
    11: 2 * 768,
    12: 2 * 768,
    13: 2 * 768,
    14: 2 * 768,
}


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
):
    """#TODO: think about and change the option to save the trained models. This can be helpful
    for testing, but then you should also store the trainers? maybe just save them as pickle files?
    Then, need to include the path as argument?
    """
    if objective == "binding_problem":
        D_out = 2
    else:
        raise ValueError

    single_patch = True
    results = defaultdict(lambda: [])

    for layer, size in layer2size.items():
        print(f"Started with layer {layer} of size {size}")
        D_in = size  # TODO: CHANGE

        loader_train = utils.build_dataloader_twopatches(
            dataset,
            layer,
            split="train",
            threshold=padding_up_to,
            amnesic_obj=amnesic_obj,
        )
        loader_val = utils.build_dataloader_twopatches(
            dataset,
            layer,
            split="val",
            threshold=padding_up_to,
            amnesic_obj=amnesic_obj,
        )
        loader_test = utils.build_dataloader_twopatches(
            dataset,
            layer,
            split="test",
            threshold=padding_up_to,
            amnesic_obj=amnesic_obj,
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
                save_models_path = f'../models/{modelname}_layer{layer}_{i}_{dataset}_{objective}_{"balanced" if balanced else "unbalanced"}_{"filtered" if filter else "unfiltered"}{"_single_patch" if single_patch else ""}_{"layernorm" if layernorm else "no_layernorm"}{"_amnesic" + str({args.amnesic_obj}) if args.amnesic_obj is not None else ""}.pt'
                torch.save(model.state_dict(), save_models_path)

    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a probe on representations ViT")
    parser.add_argument("--dataset", choices=["sup1", "pos"], required=True)
    parser.add_argument(
        "--objective",
        choices=["binding_problem"],
        required=True,
    )
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--modelname", choices=["linear_layer", "MLP", "MLP2"])
    parser.add_argument("--layernorm", action="store_true")
    parser.add_argument("--save_models", action="store_true")
    parser.add_argument("--padding_up_to", type=int, default=None)
    parser.add_argument("--single_patch", action="store_true")
    parser.add_argument("--amnesic_obj", choices=["shape", "color"])
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
    )

    results_path = "../results/"

    results_tosave = dict(results)
    with open(
        results_path
        + f'test_results_{args.modelname}_{args.dataset}_{args.objective}_{"balanced" if args.balanced else "unbalanced"}_{"filtered_" + str({args.padding_up_to}) if args.padding_up_to is not None else "unfiltered"}{"_single_patch" if args.single_patch else ""}_{"layernorm" if args.layernorm else "no_layernorm"}{"_amnesic" + str({args.amnesic_obj}) if args.amnesic_obj is not None else ""}.pickle',
        "wb",
    ) as f:
        pickle.dump(results_tosave, f)
