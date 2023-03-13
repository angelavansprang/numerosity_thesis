# This file should be used to train probes on amnesic data, to check if it worked.
# Amnesic probing can either be linear of with kernels
# Similar to the file train_probe.py, but then only amnesic probing

import amnesic_probing
import utils
import models
import utils_amnesic_probing
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pickle
import argparse
import run_kernels
from collections import defaultdict


# # below is the original
# gammas = [0.05, 0.1, 0.15]
# alphas = [0.8, 1, 1.2]
# degrees = [2, 3]
gammas = [0.1]
alphas = [1]
degrees = [2]

kernel2params = {
    "poly": {"gammas": gammas, "degrees": degrees, "alphas": alphas},
}


def get_kernelized_data(
    dataset,
    layer,
    objective,
    gamma,
    degree,
    alpha,
    threshold=30,
    run_id=1,
    kernel_type="poly",
    d=1024,
):
    params_str = "kernel-type={}_d={}_gamma={}_degree={}_alpha={}".format(
        kernel_type, d, str(gamma), str(degree), str(alpha)
    )

    X_train_path = f"../kernel_removal/{kernel_type}/{dataset}/{objective}/layer{layer}/{params_str}/preimage/Z_train.{params_str}.pickle"
    X_val_path = f"../kernel_removal/{kernel_type}/{dataset}/{objective}/layer{layer}/{params_str}/preimage/Z_val.{params_str}.pickle"
    X_test_path = f"../kernel_removal/{kernel_type}/{dataset}/{objective}/layer{layer}/{params_str}/preimage/Z_test.{params_str}.pickle"
    # X_train_path = f"../kernel_removal/interim/malevic{run_id}/kernel/projected/X_{dataset}_layer{layer}_{objective}.proj.{params_str}.pickle"
    # X_val_path = f"../kernel_removal/interim/malevic{run_id}/kernel/projected/X_dev_{dataset}_layer{layer}_{objective}.proj.{params_str}.pickle"
    # X_test_path = f"../kernel_removal/interim/malevic{run_id}/kernel/projected/X_test_{dataset}_layer{layer}_{objective}.proj.{params_str}.pickle"
    with open(X_train_path, "rb") as f:
        X_train, _, _, _ = pickle.load(f)
    with open(X_val_path, "rb") as f:
        X_val, _, _, _ = pickle.load(f)
    with open(X_test_path, "rb") as f:
        X_test, _, _, _ = pickle.load(f)

    _, y_train, _, y_val, _, y_test = run_kernels.load_ViTpatches(
        dataset, objective, layer, threshold, normalize=True
    )

    NN = 50000
    y_train, y_val, y_test = y_train[:NN], y_val[:NN], y_test[:NN]

    return X_train, y_train, X_val, y_val, X_test, y_test


def check_projections_ViT(
    dataset, objective, balanced, threshold=30, kernelize=False, probe_type="lin"
):
    """
    Perform projection P in all layers of the ViT, and train a classifier on the objective to check whether the projection worked
    """
    if objective == "color":
        D_out = 5
    elif objective == "shape":
        D_out = 4

    #TODO: remove 13 again
    for layer in range(13, 15):  # There are 14 layers with patches
        print(f"Start with layer {layer}")
        if not kernelize:  # perform amnesic probing linearly
            results = {}
            (
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                y_test,
                class2label,
            ) = amnesic_probing.get_alldata_perlayer(
                dataset, layer, objective, balanced, threshold
            )

            X_train = np.asarray(X_train)
            X_val = np.asarray(X_val)
            X_test = np.asarray(X_test)

            P = utils_amnesic_probing.open_intersection_nullspaces(
                dataset, objective, layer
            )

            # project
            X_train = X_train.dot(P)
            X_val = X_val.dot(P)

            D_in = X_train.shape[1]

            loader_train = utils_amnesic_probing.build_dataloader(
                X_train, y_train, "train", batch_size
            )
            loader_val = utils_amnesic_probing.build_dataloader(
                X_val, y_val, "val", batch_size
            )
            loader_test = utils_amnesic_probing.build_dataloader(
                X_test, y_test, "test", batch_size
            )

            acc, classifier = amnesic_probing.train_classifier(
                D_in, D_out, loader_train, loader_val, loader_test, probe_type
            )
            results[layer] = acc
        else:  # use kernelize concept erasure
            results = defaultdict(lambda: [])
            for i, gamma in enumerate(kernel2params["poly"]["gammas"]):
                for j, degree in enumerate(kernel2params["poly"]["degrees"]):
                    for k, alpha in enumerate(kernel2params["poly"]["alphas"]):
                        (
                            X_train,
                            y_train,
                            X_val,
                            y_val,
                            X_test,
                            y_test,
                        ) = get_kernelized_data(
                            dataset,
                            layer,
                            objective,
                            gamma,
                            degree,
                            alpha,
                            threshold,
                            run_id=1,
                            kernel_type="poly",
                            d=1024,
                        )

                        loader_train = utils_amnesic_probing.build_dataloader(
                            X_train, y_train, "train", batch_size
                        )
                        loader_val = utils_amnesic_probing.build_dataloader(
                            X_val, y_val, "val", batch_size
                        )
                        loader_test = utils_amnesic_probing.build_dataloader(
                            X_test, y_test, "test", batch_size
                        )

                        D_in = X_train.shape[1]

                        acc, classifier = amnesic_probing.train_classifier(
                            D_in,
                            D_out,
                            loader_train,
                            loader_val,
                            loader_test,
                            probe_type,
                        )

                        results[layer].append(acc)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform amnesic probing on representations ViT"
    )
    parser.add_argument("--dataset", choices=["sup1", "pos"], required=True)
    parser.add_argument(
        "--amnesic_objective",
        choices=["shape", "color"],
        required=True,
    )
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--kernelize", action="store_true")
    parser.add_argument("--probe_type", choices=["lin", "MLP"], required=True)
    args = parser.parse_args()

    batch_size = 10
    num_classifiers = 100
    threshold = 30

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    outcomes = check_projections_ViT(
        args.dataset,
        args.amnesic_objective,
        balanced=False,
        threshold=threshold,
        kernelize=args.kernelize,
        probe_type=args.probe_type,
    )
    print(dict(outcomes))

    results_path = "../results/"

    results_tosave = dict(outcomes)
    with open(
        results_path
        + f'test_results_amnprbng_{"kernelized" if args.kernelize is not None else "linrmvl"}_{args.probe_type}_{args.dataset}_{args.amnesic_objective}.pickle',
        "wb",
    ) as f:
        pickle.dump(results_tosave, f)
