import numpy as np
import json
from PIL import Image
from math import ceil
import utils
import matplotlib.pyplot as plt


def convert_coordinate(p, resolution=1500):
    return ceil(ceil(float(p)) / 1024 * resolution)


def get_rotation(obj, img, resolution):
    rr = convert_coordinate(obj["rr"], resolution)
    cc = convert_coordinate(obj["cc"], resolution)
    r = convert_coordinate(obj["radius"], resolution)
    if obj["shape"] == "triangle":
        if cc + 3 >= resolution:
            return "vert"
        if rr >= resolution:
            return "horiz"
        if img[cc + 3, rr] == img[cc - 3, rr]:
            return "horiz"
        elif cc + r >= resolution:
            return "vert"
        elif (
            img[cc - r, rr - 2 * r - 1] == img[cc + r, rr - 2 * r - 1]
            and img[cc - r, rr - 2 * r - 1] != (0, 0, 0, 255)
            and img[cc - r, rr - 2 * r - 1] == img[cc, rr - 4]
        ):
            print("Triggered here")
            return "horiz"
        else:
            return "vert"
    if obj["shape"] == "rectangle":
        if cc + ceil(r / 2) + 2 >= resolution:
            return "vert"
        elif img[cc, rr] == img[cc + ceil(r / 2) + 2, rr]:
            return "horiz"
        else:
            return "vert"
    if obj["shape"] == "circle" or obj["shape"] == "square":
        return "N/A"


def get_box(obj, resolution):
    rr = convert_coordinate(obj["rr"], resolution)
    cc = convert_coordinate(obj["cc"], resolution)
    r = convert_coordinate(obj["radius"], resolution)
    box = []

    if obj["shape"] == "circle" or obj["shape"] == "square":
        # tl = [cc - r - 1, rr - r - 1]
        # br = [cc + r + 1, rr + r + 1]
        tl = [cc - r, rr - r]
        br = [cc + r, rr + r]
        box.extend(tl)
        box.extend(br)
        return box

    if obj["shape"] == "rectangle":
        if obj["rotation"] == "vert":
            tl = [cc - ceil(r / 2) - 1, rr - 2 * r - 1]
            br = [cc + ceil(r / 2) + 1, rr + 2 * r + 1]
            box.extend(tl)
            box.extend(br)
            return box
        if obj["rotation"] == "horiz":
            # tl = [cc - 2 * r - 1, rr - ceil(r / 2) - 1]
            # br = [cc + 2 * r + 1, rr + ceil(r / 2) + 1]
            tl = [cc - 2 * r, rr - ceil(r / 2)]
            br = [cc + 2 * r, rr + ceil(r / 2)]
            box.extend(tl)
            box.extend(br)
            return box

    if obj["shape"] == "triangle":
        if obj["rotation"] == "vert":
            tl = [cc - 2 * r, rr - 2 * r]
            bl = [cc - 2 * r, rr + 2 * r]
            rm = [cc, rr]
            box.extend(tl)
            box.extend(bl)
            box.extend(rm)
            return box
        if obj["rotation"] == "horiz":
            tl = [cc - 2 * r, rr - 2 * r]
            tr = [cc + 2 * r, rr - 2 * r]
            bm = [cc, rr]
            box.extend(tl)
            box.extend(tr)
            box.extend(bm)
            return box


def bbox_to_rect(bbox, color):
    """Convert bounding box to matplotlib format."""
    # Convert the bounding box (top-left x, top-left y, bottom-right x,
    # bottom-right y) format to matplotlib format: ((upper-left x,
    # upper-left y), width, height)
    if bbox["shape"] != "triangle":
        bbox = bbox["box"]
        return plt.Rectangle(
            xy=(bbox[0], bbox[1]),
            width=bbox[2] - bbox[0],
            height=bbox[3] - bbox[1],
            fill=False,
            edgecolor=color,
            linewidth=2,
        )
    else:
        bbox = bbox["box"]
        xy1 = (bbox[0], bbox[1])
        xy2 = (bbox[2], bbox[3])
        xy3 = (bbox[4], bbox[5])
        return plt.Polygon(
            xy=(xy1, xy2, xy3),
            closed=True,
            fill=False,
            edgecolor=color,
            linewidth=2,
        )


def get_boxes(img_filename, dataset="sup1", split="train", global_path=".."):
    img_path = f"{global_path}/data/{dataset}/images/{split}/{img_filename}"
    annotation = utils.get_annotation(dataset=dataset, split=split)
    img = Image.open(img_path)
    img_size = img.size
    assert img_size[0] == img_size[1], "width should equal height"
    img = img.load()

    objects = annotation[img_filename.split(".")[0]][1]["objects"]
    for obj in objects:
        obj["rotation"] = get_rotation(obj, img, resolution=img_size[0])
    for obj in objects:
        obj["box"] = get_box(obj, resolution=img_size[0])

    boxes = []
    for obj in objects:
        boxes.append(obj)

    return boxes


def make_img_boxes(
    img_filename, dataset="sup1", split="train", to_save=False, global_path=".."
):
    img_path = f"{global_path}/data/{dataset}/images/{split}/{img_filename}"
    boxes = get_boxes(img_filename, dataset, split)

    plt.rcParams["figure.figsize"] = (10, 10)
    fig, ax = plt.subplots()
    img = np.array(Image.open(img_path))
    plt.imshow(img)

    for bbox in boxes:
        print(bbox)
        ax.add_patch(bbox_to_rect(bbox, "cyan"))

    if to_save:
        plt.axis("off")
        plt.savefig(
            f"../plots/bb_{dataset}_{split}_{img_filename}",
            bbox_inches="tight",
            pad_inches=0,
        )
    else:
        plt.show()


if __name__ == "__main__":
    import os

    make_img_boxes(img_filename="275.png", dataset="sup1", split="val", to_save=True)
