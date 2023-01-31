import utils
import pickle
import os
import torch


def make_representations_visual(
    device,
    model,
    preprocess,
    dataset="sup1",
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
    splits = ["train", "test", "val"]
    data_location = [f"../data/{dataset}/images/{split}/" for split in splits]
    file_path = f"../data/{dataset}/representations/"

    for loc in data_location:
        split = loc.replace(f"../data/{dataset}/images/", "")
        split = split.replace("/", "")
        visual_repr_name = (
            f"{dataset}_{split}"
            + f'{"_balanced_" + balance_objective if balance_objective is not None else ""}'
            + "_visual.pickle"
        )
        img_ids = os.listdir(loc)

        clip_data_vis = {}

        if balance_objective is not None:
            labels = utils.make_labels_dict(dataset, split)
            balanced_subset, _ = utils.make_balanced_data(labels, balance_objective)

        for img_name in img_ids:
            img_path = loc + img_name
            img_id = img_name.replace(".png", "")

            if balance_objective is not None:
                if int(img_id) in balanced_subset.keys():
                    image_features = utils.get_repr(img_path, device, model, preprocess)
                    clip_data_vis[img_id] = [
                        repr.detach().cpu().numpy() for repr in image_features
                    ]
                    del image_features
            else:
                image_features = utils.get_repr(img_path, device, model, preprocess)
                clip_data_vis[img_id] = [
                    repr.detach().cpu().numpy() for repr in image_features
                ]
                del image_features

        if to_store:
            with open(file_path + visual_repr_name, "wb") as f:
                pickle.dump(clip_data_vis, f)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = utils.get_model_preprocess(device, model_type="ViT-B/32")

    make_representations_visual(
        device,
        model,
        preprocess,
        dataset="pos",
        to_store=True,
        balance_objective="n_objects",
    )
