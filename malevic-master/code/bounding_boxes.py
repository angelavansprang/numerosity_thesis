import numpy as np
import json
from PIL import Image
from math import ceil
import utils


def convert_coordinate(p, resolution=350):
    return ceil(ceil(float(p)) / 1024 * resolution)


# obj is annotation of an object, img -- Pillow RGB image


def get_rotation(obj, img):
    rr = convert_coordinate(obj["rr"])
    cc = convert_coordinate(obj["cc"])
    r = convert_coordinate(obj["radius"])
    if obj["shape"] == "triangle":
        if cc + 2 > 349:
            return "vert"
        if rr > 349:
            return "horiz"
        if img[cc + 2, rr] == img[cc - 2, rr]:
            return "horiz"
        else:
            return "vert"
    if obj["shape"] == "rectangle":
        if cc + ceil(r / 2) + 2 > 349:
            return "vert"
        elif img[cc, rr] == img[cc + ceil(r / 2) + 2, rr]:
            return "horiz"
        else:
            return "vert"
    if obj["shape"] == "circle" or obj["shape"] == "square":
        return "N/A"


def get_box(obj):

    rr = convert_coordinate(obj["rr"])
    cc = convert_coordinate(obj["cc"])
    r = convert_coordinate(obj["radius"])
    box = []

    if obj["shape"] == "circle" or obj["shape"] == "square":
        tl = [cc - r - 1, rr - r - 1]
        br = [cc + r + 1, rr + r + 1]
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
            tl = [cc - 2 * r - 1, rr - ceil(r / 2) - 1]
            br = [cc + 2 * r + 1, rr + ceil(r / 2) + 1]
            box.extend(tl)
            box.extend(br)
            return box

    if obj["shape"] == "triangle":
        if obj["rotation"] == "vert":
            tl = [cc - 2 * r - 1, rr - 2 * r - 1]
            br = [cc + 1, rr + 2 * r + 1]
            box.extend(tl)
            box.extend(br)
            return box
        if obj["rotation"] == "horiz":
            tl = [cc - 2 * r - 1, rr - 2 * r - 1]
            br = [cc + 2 * r + 1, rr + 1]
            box.extend(tl)
            box.extend(br)
            return box


if __name__ == "__main__":
    annotation = utils.get_annotation(dataset="sup1", split="test")
