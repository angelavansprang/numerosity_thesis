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
        if layer == 7:
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
                    save_models_path = f'../models/{modelname}_layer{layer}_{i}_{dataset}_{objective}_{"filtered_" + str({padding_up_to}) if padding_up_to is not None else "unfiltered"}_{"layernorm" if layernorm else "no_layernorm"}{"_amnesic" + str({args.amnesic_obj}) if args.amnesic_obj is not None else ""}{"_firstprojectiononly" if first_projection_only else ""}{"_normalmode" if mode is None else f"_mode:{mode}"}.pt'
                    torch.save(model.state_dict(), save_models_path)
                    print(f"Model layer {layer} - {i} saved")

    return results


def evaluate_experiment(
    objective,
    dataset_totrain,
    dataset_totest,
    modelname,
    layernorm=False,
    padding_up_to=None,
    amnesic_obj=None,
    first_projection_only=False,
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

    for layer, size in layer2size.items():
        print(f"Started with layer {layer} of size {size}")
        D_in = size  # TODO: CHANGE

        loader_test = utils.build_dataloader_twopatches(
            dataset_totest,
            layer,
            split="test",
            threshold=padding_up_to,
            amnesic_obj=amnesic_obj,
            first_projection_only=first_projection_only,
            mode=mode,
        )

        model_path = f'../models/{modelname}_layer{layer}_0_{dataset_totrain}_{objective}_{"filtered_" + str({padding_up_to}) if padding_up_to is not None else "unfiltered"}_{"layernorm" if layernorm else "no_layernorm"}{"_amnesic" + str({args.amnesic_obj}) if args.amnesic_obj is not None else ""}{"_firstprojectiononly" if first_projection_only else ""}{"_normalmode" if mode is None else f"_mode:{mode}"}.pt'
        model = utils.open_model(D_in, D_out, layernorm, modelname)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        trainer = pl.Trainer(
            accelerator="gpu",
            callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
            enable_progress_bar=False,
            log_every_n_steps=100,
        )
        performance = trainer.test(model=model, dataloaders=loader_test)

        results[layer].append(performance[0]["acc"])

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a probe on representations ViT")
    parser.add_argument("--dataset", choices=["sup1", "pos", "posmo", "sup1mo"])
    parser.add_argument(
        "--objective",
        choices=["binding_problem"],
        required=True,
    )
    parser.add_argument(
        "--modelname", choices=["linear_layer", "MLP", "MLP2", "MLP2_large"]
    )
    parser.add_argument("--layernorm", action="store_true")
    parser.add_argument("--save_models", action="store_true")
    parser.add_argument("--padding_up_to", type=int, default=None)
    parser.add_argument("--amnesic_obj", choices=["shape", "color"])
    parser.add_argument("--first_projection_only", action="store_true")
    parser.add_argument(
        "--mode",
        choices=[
            "normal",
            "same_color",
            "same_shape",
            "normal_with_black",
            "balanced",
            "original",
        ],
        default="normal",
    )
    parser.add_argument("--evaluate_only", action="store_true")
    parser.add_argument("--dataset_totrain", choices=["sup1", "pos", "posmo", "sup1mo"])
    parser.add_argument("--dataset_totest", choices=["sup1", "pos", "posmo", "sup1mo"])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.evaluate_only:
        results = evaluate_experiment(
            args.objective,
            args.dataset_totrain,
            args.dataset_totest,
            modelname=args.modelname,
            layernorm=args.layernorm,
            padding_up_to=args.padding_up_to,
            amnesic_obj=args.amnesic_obj,
            first_projection_only=args.first_projection_only,
            mode=args.mode,
        )
        results_file = f'test_results_{args.modelname}_trainedon{args.dataset_totrain}_testedon{args.dataset_totest}_{args.objective}_{"filtered_" + str({args.padding_up_to}) if args.padding_up_to is not None else "unfiltered"}_{"layernorm" if args.layernorm else "no_layernorm"}{"_amnesic" + str({args.amnesic_obj}) if args.amnesic_obj is not None else ""}{"_firstprojectiononly" if args.first_projection_only else ""}{"_normalmode" if args.mode is None else f"_mode:{args.mode}"}.pickle'
    else:
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
        results_file = f'test_results_{args.modelname}_{args.dataset}_{args.objective}_{"filtered_" + str({args.padding_up_to}) if args.padding_up_to is not None else "unfiltered"}_{"layernorm" if args.layernorm else "no_layernorm"}{"_amnesic" + str({args.amnesic_obj}) if args.amnesic_obj is not None else ""}{"_firstprojectiononly" if args.first_projection_only else ""}{"_normalmode" if args.mode is None else f"_mode_{args.mode}"}.pickle'

    results_path = "../results/"
    results_tosave = dict(results)
    with open(
        results_path + results_file,
        "wb",
    ) as f:
        pickle.dump(results_tosave, f)
