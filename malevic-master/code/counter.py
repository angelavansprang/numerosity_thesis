"""
This file is meant to use a bounding probe as a counter
This is also the latest version in the notebook (v3) that had an accuracy of 57% on the training dataset of pos
"""

import utils
import torch
import numpy as np
import copy
import pickle
import argparse


def stack_reprs_2patches(patch1, patch2, repr, layer):
    input1 = utils.filter_repr(
        layer, [patch1], repr, single_patch=True, padding_up_to=None
    )
    input2 = utils.filter_repr(
        layer, [patch2], repr, single_patch=True, padding_up_to=None
    )
    z = torch.stack([torch.from_numpy(input1[0]), torch.from_numpy(input2[0])])
    z = z.flatten()
    return z


def get_object_patches(model, reprs, layer):
    patches = []
    for patch_id in range(49):
        x = utils.filter_repr(
            layer, [patch_id], reprs, single_patch=True, padding_up_to=None
        )[0]
        x = torch.from_numpy(x)
        out = model.forward(x)
        y = torch.argmax(out)
        if y == 1:
            patches.append(patch_id)
    return patches


def same_object(patch_id1, patch_id2, model, reprs, layer):
    # Check whether patch_id1 and patch_id2 are the same object twice (in both directions)
    z1 = stack_reprs_2patches(patch_id1, patch_id2, reprs, layer)
    z2 = stack_reprs_2patches(patch_id2, patch_id1, reprs, layer)

    input = torch.stack((z1, z2))

    out = model.forward(input)
    y1, y2 = torch.softmax(out, dim=1)
    prob = np.mean(
        (y1[1].item(), y2[1].item())
    )  # weighted average of probabilities in both directions

    return 1 if prob >= 0.5 else 0


def get_prob_same_object(patch_id1, patch_id2, model, reprs, layer):
    # Check whether patch_id1 and patch_id2 are the same object twice (in both directions)
    z1 = stack_reprs_2patches(patch_id1, patch_id2, reprs, layer)
    z2 = stack_reprs_2patches(patch_id2, patch_id1, reprs, layer)

    input = torch.stack((z1, z2))

    out = model.forward(input)
    y1, y2 = torch.softmax(out, dim=1)
    prob = np.mean(
        (y1[1].item(), y2[1].item())
    )  # weighted average of probabilities in both directions

    return prob


import random


def get_object_set(patch0, patches, model, reprs, layer):
    object_set = set()
    neighbors = utils.get_neighboring_patches(patch0)

    object_set.add(patch0)
    patches_to_explore = []
    history = copy.deepcopy(neighbors)
    for neighbor in neighbors:
        if neighbor in patches:
            if same_object(patch0, neighbor, model, reprs, layer):
                object_set.add(neighbor)
                patches_to_explore.append((neighbor))

    while len(patches_to_explore) != 0:
        edge = patches_to_explore.pop()
        history.append(edge)
        neighbors = utils.get_neighboring_patches(edge)
        for neighbor in neighbors:
            if neighbor in patches:
                if same_object(patch0, neighbor, model, reprs, layer):
                    object_set.add(neighbor)
                    if neighbor not in history:
                        patches_to_explore.append(neighbor)
                elif same_object(edge, neighbor, model, reprs, layer):
                    if random.random() < get_prob_same_object(
                        patch0, neighbor, model, reprs, layer
                    ):
                        # elif (
                        #     same_object(edge, neighbor, model, reprs, layer)
                        #     and get_prob_same_object(patch0, neighbor, model, reprs, layer)
                        #     > 0.3
                        # ):
                        object_set.add(neighbor)
                        if neighbor not in history:
                            patches_to_explore.append(neighbor)

    return object_set


# def get_object_set(patch0, patches, model, reprs, layer):
#     object_set = set()
#     object_set.add(patch0)
#     patches_to_explore = utils.get_neighboring_patches(patch0)
#     history = []

#     while len(patches_to_explore) != 0:
#         edge = patches_to_explore.pop()
#         if edge in patches:
#             if random.random() < get_prob_same_object(
#                 patch0, edge, model, reprs, layer
#             ):
#                 object_set.add(edge)
#                 neighbors = utils.get_neighboring_patches(edge)
#                 for neighbor in neighbors:
#                     if neighbor not in history:
#                         patches_to_explore.append(neighbor)
#         history.append(edge)

#     return object_set


def get_object_sets(reprs, layer, binding_probe, detect_probe):
    object_patches = get_object_patches(detect_probe, reprs, layer)
    object_patches_to_explore = copy.deepcopy(object_patches)
    object_sets = set()
    for patch in object_patches:
        if patch in object_patches_to_explore:
            object_set = get_object_set(
                patch, copy.deepcopy(object_patches), binding_probe, reprs, layer
            )
            for patch in object_set:
                if patch in object_patches_to_explore:
                    object_patches_to_explore.remove(patch)
            object_sets.add(frozenset(object_set))

    return object_sets


def get_error(predictions, labels):
    errors = [np.abs(predictions[i] - labels[i]) for i in range(len(predictions))]
    return np.mean(errors)


def get_accuracy(predictions, labels):
    sames = [1 if predictions[i] == labels[i] else 0 for i in range(len(predictions))]
    accuracy = sum(sames) / len(sames)
    return accuracy


def experiment(dataset, modelname, split="test"):
    repr_file = f"{dataset}_{split}_visual.pickle"
    with open(f"../data/{dataset}/representations/" + repr_file, "rb") as f:
        reprs = pickle.load(f)

    annotation = utils.get_annotation(dataset, split)

    layernorm = False
    padding_up_to = None
    results = {}

    for layer in range(15):
        # for layer in range(7, 8):
        predictions = []
        labels = []

        detect_path = f"../models/{modelname}_layer{layer}_{dataset}_object_det.pt"
        detect_probe = utils.open_model(768, 2, layernorm, modelname)
        detect_probe.load_state_dict(torch.load(detect_path))
        detect_probe.eval()

        amnesic_obj = None
        first_projection_only = False
        mode = "normal"
        # binding_path = f'../models/{modelname}_layer{layer}_0_{dataset}_binding_problem_{"filtered_" + str({padding_up_to}) if padding_up_to is not None else "unfiltered"}_{"layernorm" if layernorm else "no_layernorm"}{"_amnesic" + str({amnesic_obj}) if amnesic_obj is not None else ""}{"_firstprojectiononly" if first_projection_only else ""}{"_normalmode" if mode is None else f"_mode:{mode}"}.pt'
        binding_path = f"../models/MLP2_layer{layer}_0_pos_binding_problem_unfiltered_no_layernorm_mode:normal.pt"
        binding_probe = utils.open_model(1536, 2, layernorm, modelname)
        binding_probe.load_state_dict(torch.load(binding_path))
        binding_probe.eval()

        i = 0
        for img_id, repr in reprs.items():
            object_sets = get_object_sets(repr, layer, binding_probe, detect_probe)
            predictions.append(len(object_sets))
            labels.append(int(annotation[img_id][0]["n_objects"]))

            if i % 100 == 0:
                print(
                    f"{i}/{len(reprs)}: {img_id}.png, {len(object_sets)} vs. {int(annotation[img_id][0]['n_objects'])}, pred vs. true"
                )
            i += 1

        error = get_error(predictions, labels)
        accuracy = get_accuracy(predictions, labels)

        results[layer] = {"error": error, "accuracy": accuracy}

    return results


if __name__ == "__main__":
    # by definition, no layernorm, since this does not seem necessary for the binding probe

    parser = argparse.ArgumentParser(
        description="Perform a counting experiment with a detector and bounding probe on MALeViC data"
    )
    parser.add_argument(
        "--dataset", choices=["sup1", "pos", "posmo", "sup1mo"], required=True
    )
    parser.add_argument(
        "--modelname", choices=["linear_layer", "MLP", "MLP2"], required=True
    )
    args = parser.parse_args()

    results = experiment(args.dataset, args.modelname, split="test")
    print(results)

    results_path = "../results/"
    results_file = f"results_counter_{args.dataset}_bindingprobe_{args.modelname}_test_check.pickle"
    results_tosave = dict(results)
    with open(
        results_path + results_file,
        "wb",
    ) as f:
        pickle.dump(results_tosave, f)
