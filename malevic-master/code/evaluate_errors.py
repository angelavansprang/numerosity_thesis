import torch
import random
import utils
import models
from collections import defaultdict
import pickle
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import argparse
import matplotlib.pyplot as plt
from sklearn import metrics


def obtain_predictions(
    objective, dataset, modelname, layer: int, balanced=True, layernorm=False
):
    """#TODO: each layer individually, is that nice?"""
    if objective == "n_colors":
        D_out = 4
    elif objective == "n_objects":
        D_out = 5

    results = defaultdict(lambda: [])

    D_in = utils.layer2size()[layer]

    loader_test, class2label = utils.build_dataloader(
        dataset, layer, split="test", balanced=balanced, objective=objective
    )

    actual = []
    predicted = []
    for i in range(5):
        model_path = f'../models/{modelname}_layer{layer}_{i}_{dataset}_{objective}_{"balanced" if balanced else "unbalanced"}_{"layernorm" if layernorm else "no_layernorm"}.pt'
        model = utils.open_model(D_in, D_out, layernorm, modelname)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        for x, y in loader_test:
            out = model.forward(x)
            actual += y.detach().cpu().numpy().tolist()
            predicted += torch.argmax(out, dim=1).detach().cpu().numpy().tolist()

    return actual, predicted


def obtain_predictions_bp(
    dataset_totrain,
    dataset_totest,
    modelname,
    layer: int,
    layernorm=False,
    amnesic_obj=None,
    first_projection_only=False,
    padding_up_to=30,
    mode="normal",
):
    """#TODO: each layer individually, is that nice?
    if dataset_totrain is None, then dataset is the dataset that is trained and tested on,
    else, the model that is used is trained on dataset_totrain and tested on dataset
    """
    results = defaultdict(lambda: [])

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

    D_in = layer2size[layer]
    D_out = 2

    loader_test = utils.build_dataloader_twopatches(
        dataset_totest,
        layer,
        # split="test",
        split="train",  # this is a bigger set
        threshold=padding_up_to,
        amnesic_obj=amnesic_obj,
        first_projection_only=first_projection_only,
        mode=mode,
    )

    actual = []
    predicted = []

    model_path = f'../models/{modelname}_layer{layer}_0_{dataset_totrain}_binding_problem_{"filtered_" + str({padding_up_to}) if padding_up_to is not None else "unfiltered"}_{"layernorm" if layernorm else "no_layernorm"}{"_amnesic" + str({amnesic_obj}) if amnesic_obj is not None else ""}{"_firstprojectiononly" if first_projection_only else ""}{"_normalmode" if mode is None else "_mode{mode}"}.pt'

    model = utils.open_model(D_in, D_out, layernorm, modelname)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    for x, y in loader_test:
        out = model.forward(x)
        actual += y.detach().cpu().numpy().tolist()
        predicted += torch.argmax(out, dim=1).detach().cpu().numpy().tolist()

    return actual, predicted


def get_confusion_matrix(actual, predicted, class2label, img_name):
    confusion_matrix = metrics.confusion_matrix(actual, predicted)
    if class2label is not None:
        print(class2label.values())
        disp = metrics.ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix,
            # display_labels=list(class2label.values()).sort(),
            display_labels=class2label.values(),
        )
    else:
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    disp.plot()

    plt.title("Confusion matrix " + img_name)
    img_name = "../plots/cm_" + img_name + ".png"
    img_name.replace(" ", "_")
    plt.savefig(img_name, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate errors of trained probes by making a confusion matrix"
    )
    parser.add_argument("--dataset", choices=["sup1", "pos"])
    parser.add_argument(
        "--objective",
        choices=["n_colors", "n_objects", "binding_problem"],
        required=True,
    )
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--modelname", choices=["linear_layer", "MLP", "MLP2"])
    parser.add_argument("--layernorm", action="store_true")
    parser.add_argument("--layer", choices=range(17), required=True, type=int)
    parser.add_argument("--dataset_totrain", choices=["sup1", "pos", "posmo", "sup1mo"])
    parser.add_argument("--dataset_totest", choices=["sup1", "pos", "posmo", "sup1mo"])
    parser.add_argument(
        "--mode",
        choices=["normal", "same_color", "same_shape", "distance"],
        default="normal",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.objective == "binding_problem":
        actual, predicted = obtain_predictions_bp(
            args.dataset_totrain,
            args.dataset_totest,
            args.modelname,
            args.layer,
            layernorm=False,
            amnesic_obj=None,
            first_projection_only=False,
            padding_up_to=30,
            mode=args.mode,
        )
        img_info = f'(layer {args.layer}, {args.objective}, {args.modelname}: trained on {args.dataset_totrain}, tested on {args.dataset_totest}{", layernorm" if args.layernorm else ""}{f", mode:{args.mode}" if args.mode != "normal" else ""})'
        class2label = None
    else:
        actual, predicted = obtain_predictions(
            args.objective,
            args.dataset,
            modelname=args.modelname,
            layer=args.layer,
            balanced=args.balanced,
            layernorm=args.layernorm,
        )

        class2label, _ = utils.get_classlabel(args.dataset, objective=args.objective)
        print(class2label)

        img_info = f'(layer {args.layer}, {args.objective}, {args.modelname}, {args.dataset}, {"balanced" if args.balanced else "unbalanced"}, {"layernorm" if args.layernorm else "no layernorm"})'

    get_confusion_matrix(actual, predicted, class2label, img_info)
