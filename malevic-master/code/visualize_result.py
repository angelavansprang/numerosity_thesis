import numpy as np
import matplotlib.pyplot as plt
import pickle


layer2name = {
    0: "embed",
    1: "pos",
    2: "l_norm",
    3: "1",
    4: "2",
    5: "3",
    6: "4",
    7: "5",
    8: "6",
    9: "7",
    10: "8",
    11: "9",
    12: "10",
    13: "11",
    14: "12",
    15: "l_norm",
    16: "output",
}


def open_results(filename):
    with open(filename, "rb") as handle:
        y = pickle.load(handle)
    return y


def plot_accuracy_probes(config):
    """input:
    config (dict): contains the keys, "filenames", "labels", "info_fig", "save", "only_transformer"
    """
    results = []
    xs = []
    ys = []
    yerrors = []
    N_probes = len(config["filenames"])
    info_fig = config["info_fig"]

    for name in config["filenames"]:
        results.append(open_results(name))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for result in results:
        x = [key for key in result.keys()]
        while len(x) < 17:
            x.append(np.nan)
        # y = [np.mean(result[key]) for key in x if key < len(result) else 0]
        y = [np.mean(result[key]) for key in result.keys()]
        while len(y) < 17:
            y.append(np.nan)
        yerr = [np.std(result[key]) for key in result.keys()]
        while len(yerr) < 17:
            yerr.append(np.nan)
        xs.append(x)
        ys.append(y)
        yerrors.append(yerr)

    if not config["only_transformer"]:
        for i in range(N_probes):
            ax.errorbar(
                xs[i],
                ys[i],
                yerrors[i],
                linestyle="--",
                marker="o",
                label=config["labels"][i],
            )
        plt.xticks(range(17), labels=layer2name.values(), rotation=45)
        plt.suptitle(f"Accuracy probes from representations {info_fig}")
    else:
        labels = [label.replace(" - transformer", "") for label in layer2name.values()]
        for i in range(N_probes):
            ax.errorbar(
                x,
                ys[i][3:15],
                yerrors[i][3:15],
                linestyle="--",
                marker="o",
                label=config["labels"][i],
            )
        plt.xticks(x[3:15], labels=labels[3:15])
        plt.suptitle(
            f"Accuracy probes from representations transformer only {info_fig}"
        )
    ax.legend()

    if config["save"]:
        if N_probes == 4:
            imgname = config["filenames"][1].replace("../results/test_results_", "")
            imgname = imgname.replace(
                ".pickle", f'{"_transformeronly" if config["only_transformer"] else ""}'
            )
            imgname = (
                config["filenames"][0].replace(".pickle", "")
                + "_"
                + imgname
                + "_"
                + config["labels"][2]
                + config["labels"][3]
                + ".png"
            )
        elif N_probes == 3:
            imgname = config["filenames"][1].replace("../results/test_results_", "")
            imgname = imgname.replace(
                ".pickle", f'{"_transformeronly" if config["only_transformer"] else ""}'
            )
            imgname = (
                config["filenames"][0].replace(".pickle", "")
                + "_"
                + imgname
                + "_"
                + config["labels"][2]
                + ".png"
            )
        elif N_probes == 2:
            imgname = config["filenames"][1].replace("../results/test_results_", "")
            imgname = imgname.replace(
                ".pickle",
                f'{"_transformeronly.png" if config["only_transformer"] else ".png"}',
            )
            imgname = config["filenames"][0].replace(".pickle", "") + imgname

        elif N_probes == 1:
            imgname = config["filenames"][0].replace("../results/test_results_", "")
            imgname = imgname.replace(
                ".pickle",
                f'{"_transformeronly.png" if config["only_transformer"] else ".png"}',
            )

        imgname = imgname.replace("../results", "../plots")
        plt.ylabel("Accuracy")
        plt.xlabel("Layer")
        if config["entire_y_axis"]:
            plt.ylim([0, 1.0])
        plt.savefig(imgname, bbox_inches="tight")

    plt.show()

    return results


if __name__ == "__main__":

    config = {
        "no_plots": 3,
        "filenames": [
            "../results/test_results_MLP2_sup1_color_balanced_filtered_{30}_no_layernorm.pickle",
            "../results/test_results_MLP2_pos_shape_unbalanced_filtered_{30}_no_layernorm.pickle",
            "../results/test_results_MLP2_pos_binding_problem_unbalanced_filtered_{30}_no_layernorm.pickle",
        ],
        "labels": ["color (sup1)", "shape (pos)", "binding problem"],
        "info_fig": "(old)",
        "save": True,
        "only_transformer": False,
        "entire_y_axis": False,
    }

    results = plot_accuracy_probes(config)
    for i, result in enumerate(results):
        print(
            f"Final accuracy {i}: {np.mean(result[16]):.4f} ± {np.std(result[16]):.4f}"
        )
        max_i = np.argmax([np.mean(seq) for seq in result.values()])
        print(
            f"Highest accuracy {i}: {np.mean(result[max_i]):.4f} ± {np.std(result[max_i]):.4f}"
        )
