import json
import torch
import sys
import copy
import random
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import pickle
from torch.utils.data import DataLoader

import models
import transformer_patches
import utils_amnesic_probing

sys.path.append("../../")
import Transformer_MM_Explainability.CLIP.clip as clip

# GLOBAL VARIABLES
global_path = ".."
padding_up_to = 30

# Use this dictionary to find the number of layers necessary for the linear probe,
# (the input size of the linear probe depends on the size of the representations)


def layer2size(padding_up_to=None, single_patch=False):
    if single_patch:
        layer2size = {
            0: 768,
            1: 768,
            2: 768,
            3: 768,
            4: 768,
            5: 768,
            6: 768,
            7: 768,
            8: 768,
            9: 768,
            10: 768,
            11: 768,
            12: 768,
            13: 768,
            14: 768,
        }
        return layer2size
    if padding_up_to is not None:
        layer2size = {
            0: padding_up_to * 768,
            1: padding_up_to * 768,
            2: padding_up_to * 768,
            3: padding_up_to * 768,
            4: padding_up_to * 768,
            5: padding_up_to * 768,
            6: padding_up_to * 768,
            7: padding_up_to * 768,
            8: padding_up_to * 768,
            9: padding_up_to * 768,
            10: padding_up_to * 768,
            11: padding_up_to * 768,
            12: padding_up_to * 768,
            13: padding_up_to * 768,
            14: padding_up_to * 768,
        }
    else:
        layer2size = {
            0: 50 * 768,
            1: 50 * 768,
            2: 50 * 768,
            3: 50 * 768,
            4: 50 * 768,
            5: 50 * 768,
            6: 50 * 768,
            7: 50 * 768,
            8: 50 * 768,
            9: 50 * 768,
            10: 50 * 768,
            11: 50 * 768,
            12: 50 * 768,
            13: 50 * 768,
            14: 50 * 768,
            15: 768,
            16: 512,
        }
    return layer2size


def get_annotation(dataset, split="train"):
    path = f"{global_path}/data/{dataset}/annotation/"
    with open(path + split + "_annotation.json") as f1:
        annotation = json.load(f1)
        return annotation


def get_desc(img_id, annotation=defaultdict(None), split="train"):
    """Get description of an image as given in the annotations"""
    if annotation[str(img_id)] == None:
        annotation = get_annotation(split)
    return annotation[str(img_id)]


def get_model_preprocess(device, model_type="ViT-B/32"):
    """
    ViT-L/14 seems to break things; because the transformer has 24 layers instead of 12
    """
    clip.clip._MODELS = {
        "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
        "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
        "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    }
    model, preprocess = clip.load(model_type, device=device, jit=False)
    return model, preprocess


def open_model(D_in, D_out, layernorm, modelname):
    if modelname == "linear_layer":
        model = models.ProbingHead(D_in, D_out)
    elif modelname == "MLP":
        model = models.MLP(D_out=D_out, width=D_in, layernorm=layernorm)
    elif modelname == "MLP2":
        model = models.MLP2(input_size=D_in, output_size=D_out, layernorm=layernorm)
    return model


def get_repr(img_path, device, model, preprocess):
    """input:
    model (CLIP instance)
    """
    reprs = []
    img = preprocess(Image.open(img_path)).unsqueeze(0).to(device)

    # First step: make patches by a 2d convolution with a kernel_size and stride of 32 (the patch size)
    # here, we get 768 patches of 7x7 pixels
    z = model.visual.conv1(
        img.type(model.visual.conv1.weight.dtype)
    )  # shape = [*, width, grid, grid]

    # Second step: concatenate embeddings
    z = z.reshape(z.shape[0], z.shape[1], -1)  # shape = [*, width, grid ** 2]
    z = z.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

    z = torch.cat(
        [
            model.visual.class_embedding.to(z.dtype)
            + torch.zeros(z.shape[0], 1, z.shape[-1], dtype=z.dtype, device=z.device),
            z,
        ],
        dim=1,
    )  # shape = [*, grid ** 2 + 1, width]
    reprs.append(z)

    # Third step: add positional embeddings
    z = z + model.visual.positional_embedding.to(z.dtype)
    reprs.append(z)

    # Fourth step: layer normalization
    z = model.visual.ln_pre(z)

    # Fifth step: through the transformer; maybe exploit this further?
    # !! Info, there are 12 layers in here
    z = z.permute(1, 0, 2)  # NLD -> LND
    reprs.append(z)

    for i, block in enumerate(
        model.visual.transformer.resblocks
    ):  # deze loop vervangt:       z = model.visual.transformer(z)
        z = block(z)
        reprs.append(z)

    z = z.permute(1, 0, 2)  # LND -> NLD

    # Sixth step: another layer normalization
    z = model.visual.ln_post(z[:, 0, :])
    reprs.append(z)

    # Seventh step: project back
    if model.visual.proj is not None:
        z = z @ model.visual.proj
        reprs.append(z)

    return reprs


def filter_repr(layer, nodes, reprs, single_patch=False, padding_up_to=30):
    """
    layer (int): specifies layer in transformer, in range(0, 14)
    nodes (list): the nodes to filter the representation on. Only these are kept, the rest is not.

    0: torch.Size([1, 50, 768])
    1: torch.Size([1, 50, 768])
    2: torch.Size([50, 1, 768])
    3: torch.Size([50, 1, 768])
    4: torch.Size([50, 1, 768])
    5: torch.Size([50, 1, 768])
    6: torch.Size([50, 1, 768])
    7: torch.Size([50, 1, 768])
    8: torch.Size([50, 1, 768])
    9: torch.Size([50, 1, 768])
    10: torch.Size([50, 1, 768])
    11: torch.Size([50, 1, 768])
    12: torch.Size([50, 1, 768])
    13: torch.Size([50, 1, 768])
    14: torch.Size([50, 1, 768])
    15: torch.Size([1, 768])
    16: torch.Size([1, 512])
    """
    if layer not in range(15):
        raise ValueError(
            "Wrong layer; can only filter nodes in layers 0 up to (and including) 14"
        )

    # device = "cuda" if torch.cuda.is_available() else "cpu"

    padding_necessary = len(nodes) != padding_up_to and not single_patch

    if padding_necessary:
        padding = torch.stack(
            [torch.zeros(768) for i in range(padding_up_to - len(nodes))]
        )
        # padding = torch.stack(
        #     [torch.zeros(768) for i in range(padding_up_to - len(nodes))]
        # ).to(device)

    if layer == 0 or layer == 1:
        repr_patches = [
            # torch.from_numpy((reprs[layer][0][node + 1]))
            reprs[layer][0][node + 1]
            for node in nodes  # +1 to account for class embedding at beginning
        ]
        if not single_patch:
            z = torch.stack([torch.from_numpy(patch) for patch in repr_patches])
            if padding_necessary:
                z = torch.cat((z, padding))
            z = z.unsqueeze(0)
        else:
            z = repr_patches
    else:
        repr_patches = [
            # torch.from_numpy(reprs[layer][node + 1][0])
            reprs[layer][node + 1][0]
            for node in nodes  # +1 to account for class embedding at beginning
        ]
        if not single_patch:
            z = torch.stack([torch.from_numpy(patch) for patch in repr_patches])
            if padding_necessary:
                z = torch.cat((z, padding))
            z = z.unsqueeze(1)
        else:
            z = repr_patches
    return z


def make_labels_dict(dataset, split="train"):
    """Make a dictionary with the labels of the images in the split

    input:
    split (string): train/ val/ test
    """
    annotation = get_annotation(dataset, split)
    labels = {}
    for img in os.listdir(f"{global_path}/data/{dataset}/images/{split}/"):
        img_id = int(img.replace(".png", ""))
        desc = get_desc(img_id, annotation, split)
        labels[img_id] = {
            "n_colors": desc[0]["n_colors"],
            "n_objects": desc[0]["n_objects"],
            "objects": desc[1]["objects"],
        }
    return labels


def get_freqs_labels(labels, objective):
    """input:
    labels (dict): as returned by make_labels_dict()
    objective (string): "n_objects" or "n_colors"

    returns: (dict) e.g. {4: 907, 3: 731, 5: 259, 2: 103}
    """

    count = defaultdict(lambda: 0)
    for _, data in labels.items():
        count[int(data[objective])] += 1
    return count


def make_balanced_data(labels, objective, remove_key=None):
    """input:
    labels (dict): as returned by make_labels_dict()
    objective (string): "n_objects"  or "n_colors"
    remove_key (int or None): key to remove from count dict (i.e. value in labels)

    returns:
    balanced_labels: balanced version of labels (i.e. shortened)
    objective: copy of input

    """
    count = get_freqs_labels(labels, objective)
    if remove_key is not None:
        if remove_key in count.keys():
            del count[remove_key]
    max = min(count.values())  # min frequency is our max frequency after balancing

    balanced_labels = {}
    counter = defaultdict(lambda: 0)
    for key, value in labels.items():
        # print("remove_key: ", remove_key)
        # print("value[objective]: ", value[objective])
        if int(value[objective]) != remove_key:
            label = value[objective]
            if counter[label] < max:
                balanced_labels[key] = value
                counter[label] += 1
        # else:
        #     print(f"Removed value {value}")
        #     break
    print(f"Made balanced data for {objective}; {max} per class; {dict(count)}")
    return balanced_labels, objective


def get_classlabel(dataset, split="train", objective="n_colors", remove_key=None):
    """
    objective (string): n_colors/ n_objects
    remove_key (int): objective to remove

    returns:
    class2label (dict): from learning class to actual label (e.g. n_colors=5)
    label2class (dict): from actual label (e.g. n_colors=5) to class (e.g. 3)
    """
    labels = make_labels_dict(dataset, split)

    if remove_key is not None:
        labels_cp = copy.deepcopy(labels)
        for id, value in labels.items():
            if int(value[objective]) == remove_key:
                del labels_cp[id]
        labels = labels_cp
    count = get_freqs_labels(labels, objective)

    class2label = {}
    label2class = {}

    i = 0
    for number in sorted(count.keys()):
        label2class[number] = number - min(count.keys())
        class2label[i] = number
        i += 1

    # print(class2label)
    # print(label2class)

    return class2label, label2class


def get_class_colorshape(objective):
    """
    objective (string): "color" or "shape"
    """
    color2class = {
        "white": 0,
        "blue": 1,
        "red": 2,
        "yellow": 3,
        "green": 4,
    }
    class2color = {
        0: "white",
        1: "blue",
        2: "red",
        3: "yellow",
        4: "green",
    }
    shape2class = {"square": 0, "rectangle": 1, "circle": 2, "triangle": 3}
    class2shape = {0: "square", 1: "rectangle", 2: "circle", 3: "triangle"}

    if objective == "color":
        return class2color, color2class
    elif objective == "shape":
        return class2shape, shape2class


def get_freq_plot(objective="n_colors", split="train", save=True):
    """input:
    objective (string): 'n_colors' or 'n_objects'
    """
    labels = make_labels_dict(split)
    count = defaultdict(lambda: 0)
    for _, data in labels.items():
        count[int(data[objective])] += 1

    plt.bar(x=count.keys(), height=count.values())
    plt.title(
        "Frequency distribution of " + objective + f"(min: {min(count.values())})"
    )
    plt.xticks([int(key) for key in count.keys()])

    if save:
        plt.savefig(f"freq_{objective}_{split}.png", bbox_inches="tight")
    plt.show()


def build_dataloader(
    dataset,
    layer,
    split="train",
    balanced=True,
    objective="n_objects",
    batch_size=10,
    padding_up_to=None,
    single_patch=False
    # filter=False,  # filter out the nodes/ patches without object
):
    """Return dataloaders with the (visual) CLIP representations as data and labels whether
    the text and image match. Only add data to the dataloaders if the maximum for
    that class is not yet reached.
    NOTE: REQUIRES THAT THE REPRESENTATIONS ARE ALREADY MADE

    Input:
    ids_val (list): contains ints of the IDs of the images that should be in the validation set.
    balanced (bool): whether the dataloaders should be made balanced (i.e. same number of instances per class)
    objective (string): either 'n_objects' or 'n_colors'
    NOTE: if single_patch, then every single patch from an image with max. padding_up_to object patches is returned as input

    TODO: IMPLEMENT BALANCED == TRUE. Current implementation not yet working, due to mismatch in shapes (in both MLP and MLP2)
    Try new approach using the representations already present, and selecting the correct ones based on selection function from utils.py
    """
    if dataset == "pos" and objective == "n_colors":
        remove_key = 1
    else:
        remove_key = None

    class2label, label2class = get_classlabel(
        dataset, split=split, objective=objective, remove_key=remove_key
    )
    labels = make_labels_dict(dataset, split=split)
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
    filter = padding_up_to is not None

    file_name_patches = (
        f"../data/{dataset}/representations/{dataset}_{split}_patches.pickle"
    )

    with open(file_name_patches, "rb") as f:
        objectpatches = pickle.load(f)

    for img_id, repr in repr.items():
        if filter:
            patches = objectpatches[img_id]
            # patches = transformer_patches.get_all_patches_with_objects(
            #     f"{img_id}.png", dataset, split
            # )  # TODO: MAYBE JUST STORE THESE?
            nodes = patches.keys()

            if len(nodes) <= padding_up_to:
                input = filter_repr(
                    layer,
                    nodes,
                    repr,
                    single_patch=single_patch,
                    padding_up_to=padding_up_to,
                )
                label = int(labels[int(img_id)][objective])
                # print("label: ", label)
                # if label == 1:
                #     print("Found the problem")
                label = label2class[label]
                if not single_patch:
                    input = filter_repr(
                        layer, nodes, repr, padding_up_to=padding_up_to
                    ).flatten()
                    inputs.append(input)
                    targets.append(label)
                else:
                    for patch in input:
                        inputs.append(patch)
                        targets.append(label)
        else:
            input = repr[layer].flatten()  # flatten the representations!
            label = int(labels[int(img_id)][objective])
            label = label2class[label]
            inputs.append(input)
            targets.append(label)

    # print("len(inputs): ", len(inputs))
    # print("len(targets): ", len(targets))
    dataset_train = list(zip(inputs, targets))
    # print("len dataset: ", len(dataset_train))

    if split == "train":
        dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    else:
        dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)

    return dataloader, class2label


def build_dataloader_patchbased(
    dataset,
    layer,
    objective,
    split="train",
    balanced=True,
    batch_size=10,
    threshold=30,
):
    """Return dataloaders with the (visual) CLIP representations of one patch as data and the objective as label.

    Input:
    ids_val (list): contains ints of the IDs of the images that should be in the validation set.
    balanced (bool): whether the dataloaders should be made balanced (i.e. same number of instances per class)
    objective (string): ['n_objects', 'n_colors', 'color', 'shape']
    threshold (int): do not include images with more object patches than the threshold
    """
    if objective == "color" or objective == "shape":
        class2label, label2class = get_class_colorshape(objective)
    elif objective == "n_colors" or objective == "n_objects":
        labels = get_classlabel(dataset, split, objective)
        class2label, label2class = get_classlabel(
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
            input = filter_repr(
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

    dataset_train = list(zip(inputs, targets))
    print("len dataset: ", len(dataset_train))

    if split == "train":
        dataloader = DataLoader(
            dataset_train, batch_size=batch_size, shuffle=True, num_workers=72
        )
    else:
        dataloader = DataLoader(
            dataset_train, batch_size=batch_size, shuffle=False, num_workers=72
        )

    return dataloader, class2label


def get_neighboring_patches(patch_id):
    if patch_id == 0:
        neighbors = [1, 7, 8]
    elif patch_id == 6:
        neighbors = [5, 12, 13]
    elif patch_id == 42:
        neighbors = [35, 36, 43]
    elif patch_id == 48:
        neighbors = [40, 41, 47]
    elif patch_id < 7:
        # patch is in upper row
        neighbors = [
            patch_id + i
            for i in [-1, 1, 6, 7, 8]
            if patch_id + i >= 0 and patch_id < 49
        ]
    elif patch_id > 41:
        # patch is in lower row
        neighbors = [
            patch_id + i
            for i in [-8, -7, -6, -1, 1]
            if patch_id + i >= 0 and patch_id < 49
        ]
    elif patch_id % 7 == 0:
        # patch is in left column
        neighbors = [
            patch_id + i
            for i in [-7, -6, 1, 7, 8]
            if patch_id + i >= 0 and patch_id < 49
        ]
    elif patch_id % 7 == 6:
        # patch is in left column
        neighbors = [
            patch_id + i
            for i in [-8, -7, -1, 6, 7]
            if patch_id + i >= 0 and patch_id < 49
        ]
    else:
        neighbors = [
            patch_id + i
            for i in [-8, -7, -6, -1, 1, 6, 7, 8]
            if patch_id + i >= 0 and patch_id < 49
        ]
    return neighbors


def find_hard_positives_twopatches(patches):
    """input:
    patches (dict): keys = patch numbers ([0, 49]), values = boxes on patch

    returns:
    (int) patch_id, (int) patch_id, (0, 1) binary label indicating same object
    """
    boxes_dict = defaultdict(lambda: [])
    for patch_id, boxes in patches.items():
        for box in boxes:
            boxes_dict[box["object_id"]].append(patch_id)

    b_keys = list(boxes_dict.keys())
    random.shuffle(b_keys)
    for box in b_keys:
        patch_ids = boxes_dict[box]
        for patch_id in patch_ids:
            patches_left = copy.deepcopy(patch_ids)
            for neighbor in get_neighboring_patches(patch_id):
                if neighbor in patches_left:
                    patches_left.remove(neighbor)
            if len(patches_left) > 0:
                return patch_id, patches_left[0], 1

    # if nothing is returned by now, there are no patches of the same object that are not neighbors.
    # then, just return -1, -1
    return -1, -1, 0


def get_hard_negatives_twopatches(patches):
    keys = list(patches.keys())
    random.shuffle(keys)
    for patch_id in keys:
        boxes = patches[patch_id]
        for box in boxes:
            box_color = box["color"]
            box_shape = box["shape"]
            box_id = box["object_id"]
            for (
                patch_id2,
                boxes2,
            ) in (
                patches.items()
            ):  # Second loop to find another patch with same color and shape
                for box2 in boxes2:
                    if box2["color"] == box_color and box2["shape"] == box_shape:
                        if box2["object_id"] != box_id:
                            return patch_id, patch_id2, 0
    return -1, -1, 1


def get_randoms_twopatches(patches):
    keys = list(patches.keys())
    patch_id1, patch_id2 = random.sample(keys, 2)
    boxes1 = patches[patch_id1]
    boxes2 = patches[patch_id2]
    for box1 in boxes1:
        for box2 in boxes2:
            if box1["object_id"] == box2["object_id"]:
                return patch_id1, patch_id2, 1
    return patch_id1, patch_id2, 0


def build_dataloader_twopatches(
    dataset,
    layer,
    split="train",
    batch_size=10,
    threshold=30,
    amnesic_obj=None,
):
    """Return dataloaders with the (visual) CLIP representations of one patch as data and the objective as label.

    Input:
    ids_val (list): contains ints of the IDs of the images that should be in the validation set.
    balanced (bool): whether the dataloaders should be made balanced (i.e. same number of instances per class)
    objective (string): either 'color' or 'shape'
    threshold (int): do not include images with more object patches than the threshold
    """

    def stack_reprs_2patches(patch1, patch2, label, repr):
        if patch1 == -1 and patch2 == -1:
            patch1, patch2, label = get_randoms_twopatches(patches)
        input1 = filter_repr(
            layer, [patch1], repr, single_patch=True, padding_up_to=threshold
        )
        input2 = filter_repr(
            layer, [patch2], repr, single_patch=True, padding_up_to=threshold
        )
        z = torch.stack(
            [torch.from_numpy(input1[0]), torch.from_numpy(input2[0])]
        )  # TODO: Check if this works
        z = z.flatten()
        return z, label

    # class2label, label2class = get_class_colorshape(objective)

    repr_path = f"../data/{dataset}/representations/{dataset}_{split}_visual.pickle"

    print(f"Will try to open representations of {dataset} of split {split}")
    with open(repr_path, "rb") as f:
        repr = pickle.load(f)

    inputs = []
    targets = []

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

            if split == "train" or split == "val":
                # find 3 repr patch duo's: hard positives, hard negatives, random
                patch1, patch2, label = find_hard_positives_twopatches(patches)
                z, label = stack_reprs_2patches(patch1, patch2, label, repr)
                inputs.append(z)
                targets.append(label)

                patch1, patch2, label = get_hard_negatives_twopatches(patches)
                z, label = stack_reprs_2patches(patch1, patch2, label, repr)
                inputs.append(z)
                targets.append(label)

                patch1, patch2, label = get_randoms_twopatches(patches)
                z, label = stack_reprs_2patches(patch1, patch2, label, repr)
                inputs.append(z)
                targets.append(label)
            elif split == "test":
                nodes = list(nodes)
                for patch1 in nodes:
                    for patch2 in nodes:
                        if patch2 > patch1:
                            box1 = patches[patch1][0]
                            box2 = patches[patch2][0]
                            label = box1["object_id"] == box2["object_id"]
                            z, label = stack_reprs_2patches(patch1, patch2, label, repr)
                            inputs.append(z)
                            targets.append(label)

    if amnesic_obj is not None:  # TODO: Check if this works
        amnesic_inputs = []
        P = utils_amnesic_probing.open_intersection_nullspaces(
            dataset, amnesic_obj, layer
        )
        P = torch.from_numpy(P)
        for z in inputs:
            z1 = z[:768]
            z2 = z[768:]
            z1 = torch.unsqueeze(z1, 0)
            z2 = torch.unsqueeze(z2, 0)
            z1 = z1.to(torch.float32)
            z2 = z2.to(torch.float32)
            P = P.to(torch.float32)
            z1 = z1 @ P
            z2 = z2 @ P
            amnesic_z = torch.stack([z1, z2])
            amnesic_z = amnesic_z.flatten()
            amnesic_inputs.append(amnesic_z)
        inputs = amnesic_inputs

    dataset_train = list(zip(inputs, targets))
    print("len dataset: ", len(dataset_train))

    if split == "train":
        dataloader = DataLoader(
            dataset_train, batch_size=batch_size, shuffle=True, num_workers=72
        )
    else:
        dataloader = DataLoader(
            dataset_train, batch_size=batch_size, shuffle=False, num_workers=72
        )

    return dataloader


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = get_model_preprocess(device, model_type="ViT-B/32")

    labels_train = make_labels_dict(dataset="pos", split="train")
    labels_val = make_labels_dict(dataset="pos", split="val")
    labels_test = make_labels_dict(dataset="pos", split="test")
    count_train = get_freqs_labels(labels_train, "n_colors")
    count_val = get_freqs_labels(labels_val, "n_colors")
    count_test = get_freqs_labels(labels_test, "n_colors")
    print("train: ", count_train)
    print("val: ", count_val)
    print("test: ", count_test)

    balanced_labels, _ = make_balanced_data(labels_train, objective="n_colors")
    print("len balanced labels train: ", len(balanced_labels))
    balanced_labels, _ = make_balanced_data(labels_val, objective="n_colors")
    print("len balanced labels val: ", len(balanced_labels))
    balanced_labels, _ = make_balanced_data(labels_test, objective="n_colors")
    print("len balanced labels test: ", len(balanced_labels))

    # img_filename = "0.png"
    # dataset = "sup1"
    # split = "test"
    # img_path = f"../data/{dataset}/images/{split}/{img_filename}"
    # reprs = get_repr(img_path, device, model, preprocess)
    # print(len(reprs), reprs[0].size())
    # nodes = transformer_patches.get_all_patches_with_objects(
    #     img_filename, dataset, split
    # )
    # print("nodes: ", nodes)
    # print("amount of nodes: ", len(nodes))
    # filt_repr = filter_repr(0, nodes, reprs, single_patch=True)
    # print("len filt_repr: ", len(filt_repr))
    # print("filt_repr[0]: ", filt_repr[0])
    # print("filt_repr[0].shape: ", filt_repr[0].shape)

    # img_filename = "1.png"
    # dataset = "sup1"
    # split = "test"
    # img_path = f"../data/{dataset}/images/{split}/{img_filename}"
    # reprs = get_repr(img_path, device, model, preprocess)
    # print(len(reprs), reprs[8].size())
    # nodes = transformer_patches.get_all_patches_with_objects(
    #     img_filename, dataset, split
    # )
    # print("nodes: ", nodes)
    # print("amount of nodes: ", len(nodes))
    # filt_repr = filter_repr(8, nodes, reprs, single_patch=True)
    # print("len filt_repr: ", len(filt_repr))
    # print("filt_repr[0]: ", filt_repr[0])
    # print("filt_repr[0].shape: ", filt_repr[0].shape)
