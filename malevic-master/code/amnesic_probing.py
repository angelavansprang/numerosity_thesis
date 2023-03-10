# ┌─────────────────────────────┐
# │ attempt for amnesic probing │
# └─────────────────────────────┘
# Use this file to obtain the projections for (linear) amnesic probing

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


def get_debiasing_projection(
    num_classifiers,
    dataset,
    layer,
    objective,
    balanced,
    threshold,
    D_in=768,
    to_save=False,
    filename=None,
):
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        class2label,
    ) = get_alldata_perlayer(dataset, layer, objective, balanced, threshold)
    X_train = np.asarray(X_train)
    X_val = np.asarray(X_val)
    X_test = np.asarray(X_test)
    X_train_cp = X_train.copy()
    X_val_cp = X_val.copy()
    rowspace_projections = []
    Ws = []
    all_projections = []
    prev_acc = -99
    iters_no_change = 0
    best_projection = None
    iters_under_threshold = 0

    if objective == "color":
        D_out = 5
    elif objective == "shape":
        D_out = 4

    random_acc = 1 / D_out
    margin = 0.01

    for i in range(num_classifiers):
        if iters_under_threshold >= 3:
            print("3 iterations under the minimum accuracy.. stopping the process")
            break

        loader_train = utils_amnesic_probing.build_dataloader(
            X_train_cp, y_train, "train", batch_size
        )
        loader_val = utils_amnesic_probing.build_dataloader(
            X_val_cp, y_val, "val", batch_size
        )
        loader_test = utils_amnesic_probing.build_dataloader(
            X_test, y_test, "test", batch_size
        )

        acc, classifier = train_classifier(
            D_in, D_out, loader_train, loader_val, loader_test
        )
        print(f"Iteration {i} accuracy: {acc}")

        if prev_acc == acc:
            iters_no_change += 1
        else:
            iters_no_change = 0

        if iters_no_change >= 3:
            print("3 iterations with no accuracy change.. stopping the process")
            break
        prev_acc = acc

        if acc <= random_acc + margin and best_projection is not None:
            iters_under_threshold += 1
            continue

        W = classifier.linear_layer.parameters()
        W = next(W)
        Ws.append(W)
        P_rowspace_wi = utils_amnesic_probing.get_rowspace_projection(W)
        rowspace_projections.append(P_rowspace_wi)

        P = utils_amnesic_probing.get_projection_to_intersection_of_nullspaces(
            rowspace_projections, input_dim=768
        )
        all_projections.append(P)

        # project
        X_train_cp = X_train.dot(P)
        X_val_cp = X_val.dot(P)

        # the first iteration that gets closest performance (or less) to majority
        if acc <= random_acc + margin and best_projection is None:
            print("projection saved timestamp: {}".format(i))
            best_projection = (P, i + 1)

    P = utils_amnesic_probing.get_projection_to_intersection_of_nullspaces(
        rowspace_projections, D_in
    )

    if best_projection is None:
        print("projection saved timestamp: {}".format(num_classifiers))
        print("using all of the iterations as the final projection")
        best_projection = (P, num_classifiers)

    if to_save:
        dataset = filename["dataset"]
        objective = filename["objective"]
        layer = filename["layer"]
        balanced = filename["balanced"]
        file_path = f"../data/{dataset}/representations/"

        file_name = f"best_projection_{objective}_layer{layer}.pickle"
        with open(file_path + file_name, "wb") as f:
            pickle.dump(best_projection, f)

        file_name = f"P_{objective}_layer{layer}.pickle"
        with open(file_path + file_name, "wb") as f:
            pickle.dump(P, f)

        file_name = f"rowspace_projections_{objective}_layer{layer}.pickle"
        with open(file_path + file_name, "wb") as f:
            pickle.dump(rowspace_projections, f)

        file_name = f"all_projections_{objective}_layer{layer}.pickle"
        with open(file_path + file_name, "wb") as f:
            pickle.dump(all_projections, f)

        file_name = f"Ws_{objective}_layer{layer}.pickle"
        with open(file_path + file_name, "wb") as f:
            pickle.dump(Ws, f)

    return P, rowspace_projections, Ws, all_projections, best_projection


def train_classifier(D_in, D_out, loader_train, loader_val, loader_test, probe_type="lin"):
    if probe_type == "lin":
        classifier = models.ProbingHead(
            D_in=D_in, D_out=D_out
        )  # this is a linear classifier
    elif probe_type == "MLP":
        classifier = models.MLP2(input_size=D_in, output_size=D_out)

    # 3. Train a linear classifier to predict Z (the property to remove)

    print(classifier)

    trainer = pl.Trainer(
        # accelerator="gpu",
        callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
        enable_progress_bar=False,
        log_every_n_steps=100,
    )
    train_info = trainer.fit(classifier, loader_train, loader_val)
    performance = trainer.test(dataloaders=loader_test)
    return performance[0]["acc"], classifier


def get_alldata_perlayer(dataset, layer, objective, balanced, threshold=30):
    X_train, y_train, class2label = utils_amnesic_probing.get_data_patchbased(
        dataset,
        layer,
        objective,
        split="train",
        balanced=balanced,
        threshold=threshold,
    )
    X_val, y_val, class2label = utils_amnesic_probing.get_data_patchbased(
        dataset,
        layer,
        objective,
        split="val",
        balanced=balanced,
        threshold=threshold,
    )
    X_test, y_test, class2label = utils_amnesic_probing.get_data_patchbased(
        dataset,
        layer,
        objective,
        split="test",
        balanced=balanced,
        threshold=threshold,
    )
    return X_train, y_train, X_val, y_val, X_test, y_test, class2label


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
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--kernelize", action="store_true")
    args = parser.parse_args()

    batch_size = 10
    num_classifiers = 100
    threshold = 30

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    filename = {
        "objective": args.amnesic_objective,
        "dataset": args.dataset,
        "layer": args.layer,
        "balanced": args.balanced,
    }

    # outcomes = check_projections_ViT(
    #     args.dataset,
    #     args.amnesic_objective,
    #     balanced=False,
    #     threshold=threshold,
    #     kernelize=args.kernelize,
    # )
    # print(dict(outcomes))

    (
        P,
        rowspace_projections,
        Ws,
        all_projections,
        best_projection,
    ) = get_debiasing_projection(
        num_classifiers,
        args.dataset,
        args.layer,
        args.amnesic_objective,
        args.balanced,
        threshold,
        D_in=768,
        to_save=True,
        filename=filename,
    )
