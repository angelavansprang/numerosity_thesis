import json
import torch
import sys
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import pickle

sys.path.append("../../")
import Transformer_MM_Explainability.CLIP.clip as clip

# GLOBAL VARIABLES
global_path = ".."


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


def get_model_preprocess(device, model_type="ViT-L/14"):
    clip.clip._MODELS = {
        "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
        "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
        "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    }
    model, preprocess = clip.load(model_type, device=device, jit=False)
    return model, preprocess


# def get_repr(img_path, device, model, preprocess):
#     img = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
#     image_features = model.encode_image(img)
#     return image_features


def get_repr(img_path, device, model, preprocess):
    """input:
    model (CLIP instance)
    """
    reprs = []
    img = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    # image_features = model.encode_image(img)

    # First step: make patches by a 2d convolution with a kernel_size and stride of 32 (the patch size)
    # here, we get 768 patches of 7x7 pixels
    z = model.visual.conv1(
        img.type(model.visual.conv1.weight.dtype)
    )  # shape = [*, width, grid, grid]

    # Second step: concatenate embeddings
    # !! Info in embedding?
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
    print(count)
    max = min(count.values())  # min frequency is our max frequency after balancing

    balanced_labels = {}
    counter = defaultdict(lambda: 0)
    for key, value in labels.items():
        label = value[objective]
        if counter[label] < max:
            balanced_labels[key] = value
            counter[label] += 1
    print(f"Made balanced data for {objective}; {max} per class")
    return balanced_labels, objective


def get_classlabel(dataset, split="train", objective="n_colors"):
    """
    objective (string): n_colors/ n_objects
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


# def make_representations(
#     device, model, preprocess, dataset, split="train", to_store=False
# ):
#     """Gets visual representations of the images using CLIP (ViT)

#     Input:
#     img_ids (list): img_ids as integer with '.png'

#     Stores:
#     clip_data_vis (dict): img_id is key, value is the tensor (of last layer)
#     """
#     path = f"{global_path}/data/{dataset}/images/{split}/"

#     img_ids = os.listdir(path)
#     visual_repr_name = f"final_layer_{split}.pickle"

#     clip_data_vis = {}

#     for img_name in tqdm(img_ids):
#         img_path = path + img_name
#         img_id = img_name.replace(".png", "")

#         image_features = get_repr(img_path, device, model, preprocess)

#         clip_data_vis[img_id] = image_features.cpu().detach().numpy()[0]
#         del image_features

#     if to_store:
#         file_path = f"{global_path}/data/{dataset}/representations/"
#         with open(file_path + visual_repr_name, "wb") as f:
#             pickle.dump(clip_data_vis, f)


def make_representations_visual(
    device,
    model,
    preprocess,
    dataset="sup1",
    file_path="../data/sup1/representations/",
    to_store=False,
    balance_objective=None,
):
    """Makes visual representation from the VALSE images

    Input:
    balanced_subset (dict): labels as returned by make_labels_dict()
    objective (string): if balanced_subset is not None, this says whether it is balanced for 'n_objects' or 'n_colors'

    Stores:
    clip_data_vis (dict): img_id is key, value is the tensor (of last layer)
    """
    if dataset == "pos":
        splits = ["test"]
    elif dataset == "sup1":
        splits = ["train", "test", "val"]
    data_location = [f"../data/{dataset}/images/{split}/" for split in splits]

    for loc in data_location:
        split = loc.replace(f"../data/{dataset}/images/", "")
        split = split.replace("/", "")
        visual_repr_name = (
            f"{dataset}_{split}"
            + f'{"_balanced_" + balance_objective if balance_objective is not None else ""}'
            # + f'{objective if balanced_subset is not None else ""}'
            + "_visual.pickle"
        )
        img_ids = os.listdir(loc)

        clip_data_vis = {}

        if balance_objective is not None:
            labels = make_labels_dict(dataset, split)
            balanced_subset, _ = make_balanced_data(labels, balance_objective)

        for img_name in tqdm(img_ids):
            img_path = loc + img_name
            img_id = img_name.replace(".png", "")

            if balanced_subset is not None:
                if int(img_id) in balanced_subset.keys():
                    image_features = get_repr(img_path, device, model, preprocess)
                    clip_data_vis[img_id] = [
                        repr.detach().cpu().numpy() for repr in image_features
                    ]
                    del image_features
            else:
                image_features = get_repr(img_path, device, model, preprocess)
                clip_data_vis[img_id] = [
                    repr.detach().cpu().numpy() for repr in image_features
                ]
                del image_features

        # print(f"{split}: \n", clip_data_vis)
        if to_store:
            with open(file_path + visual_repr_name, "wb") as f:
                pickle.dump(clip_data_vis, f)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = get_model_preprocess(device, model_type="ViT-L/14")

    make_representations_visual(
        device,
        model,
        preprocess,
        dataset="sup1",
        to_store=True,
        balance_objective="n_colors",
    )
