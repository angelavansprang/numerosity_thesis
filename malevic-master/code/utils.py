import json
import torch
import sys
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import pickle
from torch.utils.data import DataLoader
import models
import transformer_patches

sys.path.append("../../")
import Transformer_MM_Explainability.CLIP.clip as clip

# GLOBAL VARIABLES
global_path = ".."
# padding_up_to = 30

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


def make_balanced_data(labels, objective):
    """input:
    labels (dict): as returned by make_labels_dict()
    objective (string): "n_objects"  or "n_colors"

    returns:
    balanced_labels: balanced version of labels (i.e. shortened)
    objective: copy of input

    """
    count = get_freqs_labels(labels, objective)
    max = min(count.values())  # min frequency is our max frequency after balancing

    balanced_labels = {}
    counter = defaultdict(lambda: 0)
    for key, value in labels.items():
        label = value[objective]
        if counter[label] < max:
            balanced_labels[key] = value
            counter[label] += 1
    print(f"Made balanced data for {objective}; {max} per class; {dict(count)}")
    return balanced_labels, objective


def get_classlabel(dataset, split="train", objective="n_colors"):
    """
    objective (string): n_colors/ n_objects

    returns:
    class2label (dict): from learning class to actual label (e.g. n_colors=5)
    label2class (dict): from actual label (e.g. n_colors=5) to class (e.g. 3)
    """
    labels = make_labels_dict(dataset, split)
    count = get_freqs_labels(labels, objective)

    class2label = {}
    label2class = {}

    i = 0
    for number in sorted(count.keys()):
        label2class[number] = number - min(count.keys())
        class2label[i] = number
        i += 1

    return class2label, label2class


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

    class2label, label2class = get_classlabel(dataset, split=split, objective=objective)
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

    for img_id, repr in repr.items():
        if filter:
            nodes = transformer_patches.get_all_patches_with_objects(
                f"{img_id}.png", dataset, split
            )  # TODO: MAYBE JUST STORE THESE?

            if len(nodes) <= padding_up_to:
                input = filter_repr(
                    layer,
                    nodes,
                    repr,
                    single_patch=single_patch,
                    padding_up_to=padding_up_to,
                )
                label = int(labels[int(img_id)][objective])
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

    dataset_train = list(zip(inputs, targets))
    print("len dataset: ", len(dataset_train))

    if split == "train":
        dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    else:
        dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)

    return dataloader, class2label


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = get_model_preprocess(device, model_type="ViT-B/32")

    # labels = make_labels_dict(dataset="sup1", split="test")
    # print("len labels: ", len(labels))
    # balanced_labels, _ = make_balanced_data(labels, objective="n_objects")
    # print("len balanced labels: ", len(balanced_labels))

    # TODO: check

    img_filename = "0.png"
    dataset = "sup1"
    split = "test"
    img_path = f"../data/{dataset}/images/{split}/{img_filename}"
    reprs = get_repr(img_path, device, model, preprocess)
    print(len(reprs), reprs[0].size())
    nodes = transformer_patches.get_all_patches_with_objects(
        img_filename, dataset, split
    )
    print("nodes: ", nodes)
    print("amount of nodes: ", len(nodes))
    filt_repr = filter_repr(0, nodes, reprs, single_patch=True)
    print("len filt_repr: ", len(filt_repr))
    print("filt_repr[0]: ", filt_repr[0])
    print("filt_repr[0].shape: ", filt_repr[0].shape)

    img_filename = "1.png"
    dataset = "sup1"
    split = "test"
    img_path = f"../data/{dataset}/images/{split}/{img_filename}"
    reprs = get_repr(img_path, device, model, preprocess)
    print(len(reprs), reprs[8].size())
    nodes = transformer_patches.get_all_patches_with_objects(
        img_filename, dataset, split
    )
    print("nodes: ", nodes)
    print("amount of nodes: ", len(nodes))
    filt_repr = filter_repr(8, nodes, reprs, single_patch=True)
    print("len filt_repr: ", len(filt_repr))
    print("filt_repr[0]: ", filt_repr[0])
    print("filt_repr[0].shape: ", filt_repr[0].shape)
