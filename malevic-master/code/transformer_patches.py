import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from PIL import Image
import bounding_boxes
import os
import utils
import pickle
from collections import defaultdict


def open_image_withpatches(imgname, dataset, split, to_save=False):
    print(f"../data/{dataset}/images/{split}/{imgname}")
    image = Image.open(
        f"../data/{dataset}/images/{split}/{imgname}"
        # f"../examples/{dataset}/30patchimages/{imgname}"
    )
    my_dpi = 300

    # Set up figure
    print(image.size)
    fig = plt.figure(
        figsize=(float(image.size[0]) / my_dpi, float(image.size[1]) / my_dpi),
        dpi=my_dpi,
    )
    ax = fig.add_subplot(111)

    # Remove whitespace from around the image
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # Set the gridding interval: here we use the major tick interval
    myInterval = image.size[0] / 7
    loc = plticker.MultipleLocator(base=myInterval)
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)

    # Add the grid
    ax.grid(which="major", axis="both", linestyle="-")

    # Add the image
    ax.imshow(image)

    # Find number of gridsquares in x and y direction
    nx = abs(int(float(ax.get_xlim()[1] - ax.get_xlim()[0]) / float(myInterval)))
    ny = abs(int(float(ax.get_ylim()[1] - ax.get_ylim()[0]) / float(myInterval)))

    # Add some labels to the gridsquares
    for j in range(ny):
        y = myInterval / 2 + j * myInterval
        for i in range(nx):
            x = myInterval / 2.0 + float(i) * myInterval
            ax.text(
                # x, y, "{:d}".format(i + j * nx + 1), color="w", ha="center", va="center"
                x,
                y,
                "{:d}".format(i + j * nx),
                color="w",
                ha="center",
                va="center",
            )

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    if to_save:
        plt.savefig(
            # f"../examples/{dataset}/patches_{imgname}",
            f"../examples/{dataset}/30patchimages/patches_{imgname}",
            bbox_inches="tight",
            pad_inches=0,
        )

    # plt.show()


def get_all_patches_with_objects(img_filename, dataset, split):
    """New version: returns dictionary of patch numbers with box"""
    img_path = f"../data/{dataset}/images/{split}/{img_filename}"
    boxes = bounding_boxes.get_boxes(img_filename, dataset, split)

    img = Image.open(img_path)
    height, width = img.size
    step = int(height / 7)

    patches = defaultdict(lambda: [])
    for j, y in enumerate(range(0, step * 7, step)):
        for i, x in enumerate(range(0, step * 7, step)):
            number = i + j * 7
            patch = (x, y, x + int(width / 7), y + int(height / 7))
            for box in boxes:
                if check_box_in_patch(patch, box["box"]):
                    patches[number].append(box)
    return patches


def store_patches_dataset(dataset):
    splits = ["train", "test", "val"]
    data_location = [f"../data/{dataset}/images/{split}/" for split in splits]
    file_path = f"../data/{dataset}/representations/"

    for loc in data_location:
        split = loc.replace(f"../data/{dataset}/images/", "")
        split = split.replace("/", "")
        repr_name = f"{dataset}_{split}" + "_patches.pickle"
        img_ids = os.listdir(loc)

        patches = {}

        for i, img_name in enumerate(img_ids):
            img_id = img_name.replace(".png", "")
            img_patches = get_all_patches_with_objects(img_name, dataset, split)
            patches[img_id] = dict(img_patches)
            if i % 100 == 0:
                print(f"{i}/ {len(img_ids)}")

        with open(file_path + repr_name, "wb") as f:
            pickle.dump(patches, f)

        print(f"Split {split} done")


# def get_all_patches_with_objects(img_filename, dataset, split):
#     img_path = f"../data/{dataset}/images/{split}/{img_filename}"
#     boxes = bounding_boxes.get_boxes(img_filename, dataset, split)

#     img = Image.open(img_path)
#     height, width = img.size
#     step = int(height / 7)

#     patches = []
#     for j, y in enumerate(range(0, step * 7, step)):
#         for i, x in enumerate(range(0, step * 7, step)):
#             number = i + j * 7
#             patch = (x, y, x + int(width / 7), y + int(height / 7))
#             for box in boxes:
#                 if check_box_in_patch(patch, box):
#                     patches.append(number)
#                     break
#     return patches


def check_box_in_patch(patch, box):
    """
    patch: format (top-left x, top-left y, bottom-right x, bottom-right y) e.g. [608, 969, 902, 1117]
    box (list): format (top-left x, top-left y, bottom-right x, bottom-right y) e.g. [608, 969, 902, 1117]
    """
    tlx_patch, tly_patch, brx_patch, bry_patch = patch
    tlx, tly, brx, bry = box
    if tlx >= brx_patch or brx <= tlx_patch or tly_patch >= bry or bry_patch <= tly:
        return False
    else:
        return True


def analyze_amount_objectpatches(dataset, split, balance_objective=None, to_save=False):
    import os

    results = {}
    img_ids = os.listdir(f"../data/{dataset}/images/{split}/")

    if balance_objective == None:
        for img_name in img_ids:
            patches = get_all_patches_with_objects(img_name, dataset, split)
            results[img_name] = len(patches)
    else:
        labels = utils.make_labels_dict(dataset, split)
        balanced_subset, _ = utils.make_balanced_data(labels, balance_objective)
        for img_name in img_ids:
            img_id = int(img_name.replace(".png", ""))
            if img_id in balanced_subset.keys():
                patches = get_all_patches_with_objects(img_name, dataset, split)
                results[img_name] = len(patches)
    if to_save:
        file_name = (
            f"../examples/{dataset}/objectpatches_{split}"
            + f'{"_balanced_" + balance_objective if balance_objective is not None else ""}'
            + ".pickle"
        )
        with open(file_name, "wb") as f:
            pickle.dump(results, f)


def visualize_N_objectpatches(dataset, split, balance_objective=None, to_save=False):
    objectpatches = open_N_objectpatches(dataset, split, balance_objective)
    file_name = (
        f"../examples/{dataset}/objectpatches_{split}"
        + f'{"_balanced_" + balance_objective if balance_objective is not None else ""}'
        + ".pickle"
    )
    with open(file_name, "rb") as f:
        objectpatches = pickle.load(f)

    results = defaultdict(lambda: 0)
    for N_patches in objectpatches.values():
        results[N_patches] += 1

    plt.bar(list(results.keys()), list(results.values()))
    plt.title(
        f"Frequencies number of object patches ({dataset}, {split}"
        + f'{", " + balance_objective if balance_objective is not None else ""}'
        + ")"
    )
    plt.xlabel("Number of object patches")
    plt.ylabel("Frequency")

    if to_save:
        file_name = (
            f"../examples/{dataset}/frequencies_{dataset}_{split}"
            + f'{"_balanced_" + balance_objective if balance_objective is not None else ""}'
            + ".png"
        )
        plt.savefig(
            file_name,
            bbox_inches="tight",
            pad_inches=0,
        )


def open_N_objectpatches(dataset, split, balance_objective):
    file_name = (
        f"../examples/{dataset}/objectpatches_{split}"
        + f'{"_balanced_" + balance_objective if balance_objective is not None else ""}'
        + ".pickle"
    )
    with open(file_name, "rb") as f:
        objectpatches = pickle.load(f)
    return objectpatches


if __name__ == "__main__":
    # open_image_withpatches("bb_10295.png", "sup1", "train", to_save=True)
    # print(get_all_patches_with_objects("10295.png", "sup1", "train"))

    # import shutil

    # objectpatches = open_N_objectpatches("sup1", "train", balance_objective="n_colors")
    # thirty_ids = [id for id, value in objectpatches.items() if value == 30]
    # print(len(thirty_ids))

    # for id in thirty_ids:
    #     src_path = f"../data/sup1/images/train/{id}"
    #     dst_path = f"../examples/sup1/30patchimages/{id}"
    #     shutil.copy(src_path, dst_path)

    # import os

    # for img_name in os.listdir("../examples/sup1/30patchimages"):
    #     if img_name[0] == "b" and img_name[1] == "b":
    #         open_image_withpatches(img_name, "sup1", "train", to_save=True)

    store_patches_dataset("sup1mo")
