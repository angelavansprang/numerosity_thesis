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

    D_in = utils.layer2size[layer]

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


def get_confusion_matrix(actual, predicted, class2label, img_name):

    confusion_matrix = metrics.confusion_matrix(actual, predicted)
    print(class2label.values())
    disp = metrics.ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix,
        # display_labels=list(class2label.values()).sort(),
        display_labels=class2label.values(),
    )
    disp.plot()

    plt.title("Confusion matrix " + img_name)
    img_name = "../plots/cm_" + img_name + ".png"
    img_name.replace(" ", "_")
    plt.savefig(img_name, bbox_inches="tight")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Evaluate errors of trained probes by making a confusion matrix"
    )
    parser.add_argument("--dataset", choices=["sup1", "pos"], required=True)
    parser.add_argument("--objective", choices=["n_colors", "n_objects"], required=True)
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--modelname", choices=["linear_layer", "MLP", "MLP2"])
    parser.add_argument("--layernorm", action="store_true")
    parser.add_argument("--layer", choices=range(17), required=True, type=int)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

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
