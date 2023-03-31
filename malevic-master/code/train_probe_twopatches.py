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
    layernorm=False,
    save_models=False,
    padding_up_to=None,
    amnesic_obj=None,
    first_projection_only=False,
    single_experiment=True,
    mode="normal",
):
    """#TODO: think about and change the option to save the trained models. This can be helpful
    for testing, but then you should also store the trainers? maybe just save them as pickle files?
    Then, need to include the path as argument?

    mode: one of the following: {None, "same_color", "same_shape"}
    """
    if objective == "binding_problem":
        D_out = 2
    else:
        raise ValueError

    single_patch = True
    results = defaultdict(lambda: [])

    if single_experiment:
        max_iter = 1
    else:
        max_iter = 5

    for layer, size in layer2size.items():
        print(f"Started with layer {layer} of size {size}")
        D_in = size  # TODO: CHANGE

        loader_train = utils.build_dataloader_twopatches(
            dataset,
            layer,
            split="train",
            threshold=padding_up_to,
            amnesic_obj=amnesic_obj,
            first_projection_only=first_projection_only,
            mode=mode,
        )
        loader_val = utils.build_dataloader_twopatches(
            dataset,
            layer,
            split="val",
            threshold=padding_up_to,
            amnesic_obj=amnesic_obj,
            first_projection_only=first_projection_only,
            mode=mode,
        )
        loader_test = utils.build_dataloader_twopatches(
            dataset,
            layer,
            split="test",
            threshold=padding_up_to,
            amnesic_obj=amnesic_obj,
            first_projection_only=first_projection_only,
            mode=mode,
        )

        for i in range(max_iter):
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
                save_models_path = f'../models/{modelname}_layer{layer}_{i}_{dataset}_{objective}_{"filtered" if filter else "unfiltered"}_{"layernorm" if layernorm else "no_layernorm"}{"_amnesic" + str({args.amnesic_obj}) if args.amnesic_obj is not None else ""}{"_firstprojectiononly" if first_projection_only else ""}{"_normalmode" if mode is None else "_mode{mode}"}.pt'
                torch.save(model.state_dict(), save_models_path)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a probe on representations ViT")
    parser.add_argument("--dataset", choices=["sup1", "pos", "posmo"], required=True)
    parser.add_argument(
        "--objective",
        choices=["binding_problem"],
        required=True,
    )
    parser.add_argument("--modelname", choices=["linear_layer", "MLP", "MLP2"])
    parser.add_argument("--layernorm", action="store_true")
    parser.add_argument("--save_models", action="store_true")
    parser.add_argument("--padding_up_to", type=int, default=None)
    parser.add_argument("--amnesic_obj", choices=["shape", "color"])
    parser.add_argument("--first_projection_only", action="store_true")
    parser.add_argument(
        "--mode", choices=["normal", "same_color", "same_shape"], default="normal"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = experiment_per_layer(
        args.objective,
        args.dataset,
        modelname=args.modelname,
        layernorm=args.layernorm,
        save_models=args.save_models,
        padding_up_to=args.padding_up_to,
        amnesic_obj=args.amnesic_obj,
        first_projection_only=args.first_projection_only,
        mode=args.mode,
    )

    results_path = "../results/"

    results_tosave = dict(results)
    with open(
        results_path
        + f'test_results_{args.modelname}_{args.dataset}_{args.objective}_{"filtered_" + str({args.padding_up_to}) if args.padding_up_to is not None else "unfiltered"}_{"layernorm" if args.layernorm else "no_layernorm"}{"_amnesic" + str({args.amnesic_obj}) if args.amnesic_obj is not None else ""}{"_firstprojectiononly" if args.first_projection_only else ""}{"_normalmode" if args.mode is None else "_mode{args.mode}"}.pickle',
        "wb",
    ) as f:
        pickle.dump(results_tosave, f)
