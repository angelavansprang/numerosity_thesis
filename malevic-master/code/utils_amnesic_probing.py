# ┌─────────────────────────────────────────┐
# │ copies of functions for amnesic probing │
# └─────────────────────────────────────────┘

import torch
import numpy as np
import tqdm
import scipy
import utils
import pickle
from torch.utils.data import DataLoader


def get_data_patchbased(
    dataset,
    layer,
    objective,
    split="train",
    balanced=True,
    threshold=30,
):
    """Return data with the (visual) CLIP representations of one patch as data and the objective as label.

    Input:
    ids_val (list): contains ints of the IDs of the images that should be in the validation set.
    balanced (bool): whether the dataloaders should be made balanced (i.e. same number of instances per class)
    objective (string): ['n_objects', 'n_colors', 'color', 'shape']
    threshold (int): do not include images with more object patches than the threshold
    """
    if objective == "color" or objective == "shape":
        class2label, label2class = utils.get_class_colorshape(objective)
    elif objective == "n_colors" or objective == "n_objects":
        labels = utils.get_classlabel(dataset, split, objective)
        class2label, label2class = utils.get_classlabel(
            dataset=dataset, split=split, objective=objective
        )

    if balanced:
        repr_path = f"../data/{dataset}/representations/{dataset}_{split}_balanced_{objective}_visual.pickle"
    else:
        repr_path = f"../data/{dataset}/representations/{dataset}_{split}_visual.pickle"
    # repr_path = f"../data/{dataset}/representations/{dataset}_{split}_visual.pickle"

    print(f"Will try to open representations of {dataset} of split {split}")
    print(f"Balanced is: {balanced}")
    with open(repr_path, "rb") as f:
        repr = pickle.load(f)

    inputs = []
    targets = []

    # if balanced:
    #     balanced_labels, _ = make_balanced_data(labels, objective)

    file_name_patches = (
        f"../data/{dataset}/representations/{dataset}_{split}_patches.pickle"
    )

    with open(file_name_patches, "rb") as f:
        objectpatches = pickle.load(f)

    for img_id, repr in repr.items():
        patches = objectpatches[img_id]
        # patches = transformer_patches.get_all_patches_with_objects(
        #     f"{img_id}.png", dataset, split
        # )  # TODO: MAYBE JUST STORE THESE?
        nodes = patches.keys()

        if len(nodes) <= threshold:
            input = utils.filter_repr(
                layer,
                nodes,
                repr,
                single_patch=True,
                padding_up_to=threshold,
            )
            for i, patch_id in enumerate(nodes):
                patch = input[i]
                boxes = patches[
                    patch_id
                ]  # one patch could contain multiple object boxes, but skip these because label uncertain
                if len(boxes) > 1:
                    break
                box = boxes[0]
                # print(box)
                # print(box[objective])
                if objective == "color" or objective == "shape":
                    label = label2class[box[objective]]
                elif objective == "n_colors" or objective == "n_objects":
                    label = int(labels[int(img_id)][objective])
                    label = label2class[label]
                inputs.append(patch)
                targets.append(label)

    return inputs, targets, class2label


def build_dataloader(inputs, targets, split, batch_size=10):
    dataset_train = list(zip(inputs, targets))

    if split == "train":
        dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    else:
        dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)

    return dataloader


def open_best_projection(dataset, objective, layer):
    # Open best projection
    path = f"../data/{dataset}/representations/"
    file_name = f"projection_{objective}_layer{layer}.pickle"
    with open(path + file_name, "rb") as f:
        P, num_classifiers = pickle.load(f)
    return P, num_classifiers


def open_intersection_nullspaces(dataset, objective, layer):
    path = f"../data/{dataset}/representations/"
    file_name = f"rowspace_projections_{objective}_layer{layer}.pickle"
    with open(path + file_name, "rb") as f:
        rowspace_projections = pickle.load(f)
    P = get_projection_to_intersection_of_nullspaces(
                rowspace_projections, input_dim=768
            )
    return P

def get_rowspace_projection(W: np.ndarray) -> np.ndarray:
    """
    :param W: the matrix over its nullspace to project
    :return: the projection matrix over the rowspace
    """
    if not isinstance(W, np.ndarray):
        W = W.detach().numpy()
    if np.allclose(W, 0):
        w_basis = np.zeros_like(W.T)
    else:
        w_basis = scipy.linalg.orth(W.T)  # orthogonal basis

    w_basis * np.sign(w_basis[0][0])  # handle sign ambiguity
    P_W = w_basis.dot(w_basis.T)  # orthogonal projection on W's rowspace

    return P_W


def get_projection_to_intersection_of_nullspaces(
    rowspace_projection_matrices: list[np.ndarray], input_dim: int
):
    """
    Given a list of rowspace projection matrices P_R(w_1), ..., P_R(w_n),
    this function calculates the projection to the intersection of all nullspasces of the matrices w_1, ..., w_n.
    uses the intersection-projection formula of Ben-Israel 2013 http://benisrael.net/BEN-ISRAEL-NOV-30-13.pdf:
    N(w1)∩ N(w2) ∩ ... ∩ N(wn) = N(P_R(w1) + P_R(w2) + ... + P_R(wn))
    :param rowspace_projection_matrices: List[np.array], a list of rowspace projections
    :param input_dim: input dim
    """

    I = np.eye(input_dim)
    Q = np.sum(rowspace_projection_matrices, axis=0)
    P = I - get_rowspace_projection(Q)

    return P


def debias_by_specific_directions(directions: list[np.ndarray], input_dim: int):
    """
    the goal of this function is to perform INLP on a set of user-provided directions
    (instead of learning those directions).
    :param directions: list of vectors, as numpy arrays.
    :param input_dim: dimensionality of the vectors.
    """

    rowspace_projections = []

    for v in directions:
        P_v = get_rowspace_projection(v)
        rowspace_projections.append(P_v)

    P = get_projection_to_intersection_of_nullspaces(rowspace_projections, input_dim)

    return P
