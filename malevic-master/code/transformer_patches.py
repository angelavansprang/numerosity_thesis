import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from PIL import Image
import bounding_boxes


def open_image_withpatches(imgname, dataset, split, to_save=False):

    image = Image.open(
        f"../data/{dataset}/images/{split}/{imgname}"
        # f"../examples/{dataset}/{imgname}"
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
                x, y, "{:d}".format(i + j * nx), color="w", ha="center", va="center"
            )

    # ax.set_xticklabels([])
    # ax.set_yticklabels([])

    if to_save:
        plt.savefig(
            f"../examples/{dataset}/patches_{imgname}",
            bbox_inches="tight",
            pad_inches=0,
        )

    plt.show()


def get_all_patches_with_objects(img_filename, dataset, split):
    img_path = f"../data/{dataset}/images/{split}/{img_filename}"
    boxes = bounding_boxes.get_boxes(img_filename, dataset, split)

    img = Image.open(img_path)
    height, width = img.size
    step = int(height / 7)

    patches = []
    for j, y in enumerate(range(0, step * 7, step)):
        for i, x in enumerate(range(0, step * 7, step)):
            number = i + j * 7
            patch = (x, y, x + int(width / 7), y + int(height / 7))
            for box in boxes:
                if check_box_in_patch(patch, box):
                    patches.append(number)
                    break
    return patches


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


if __name__ == "__main__":
    # open_image_withpatches("0.png", "sup1", "test", to_save=True)
    print(get_all_patches_with_objects("0.png", "sup1", "test"))
