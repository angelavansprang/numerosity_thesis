import numpy as np
import matplotlib.pyplot as plt
import pickle
import os


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
    if filename == "eval_kernels":
        return get_scores_eval_kernels(
            kernel_type="poly", dataset="pos", amn_objective="color"
        )
    with open(filename, "rb") as handle:
        y = pickle.load(handle)
    return y


def get_scores_eval_kernels(kernel_type, dataset, amn_objective):
    if kernel_type == "poly":
        d = 1024
        gamma = 0.1
        alpha = 1
        degree = 2
    elif kernel_type == "sigmoid":
        d = 1024
        gamma = 0.005
        alpha = 0
        degree = None

    params_str = "kernel-type={}_d={}_gamma={}_degree={}_alpha={}".format(
        kernel_type, d, str(gamma), str(degree), str(alpha), kernel_type
    )

    results = {}
    for layer in range(17):
        file = f"../kernel_removal/{kernel_type}/{dataset}/{amn_objective}/layer{layer}/{params_str}/preimage/eval/scores_with_multiple2.pickle"
        if os.path.exists(file):
            with open(file, "rb") as f:
                score = pickle.load(f)
            results[layer] = [score[params_str + f"_adv-type={kernel_type}"]]
        # else:
        #     results[layer] = np.nan
    print(results)
    return results


def plot_accuracy_probes(config):
    """input:
    config (dict): contains the keys, "filenames", "labels", "fig_title", "save", "only_transformer"
    """
    results = []
    xs = []
    ys = []
    yerrors = []
    N_probes = len(config["filenames"])
    fig_title = config["fig_title"]

    for name in config["filenames"]:
        results.append(open_results(name))

    print(results)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # plt.rcParams.update(
    #     {
    #         "font.family": "serif",  # use serif/main font for text elements
    #         # "text.usetex": True,  # use inline math for ticks
    #         "pgf.rcfonts": False,  # don't setup fonts from rc parameters
    #     }
    # )

    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")

    for result in results:
        x = [int(key) for key in result.keys()]
        while len(x) < 17:
            x.append(np.nan)
        y = [
            np.mean(result[key]) if result[key] is not np.nan else np.nan
            for key in result.keys()
        ]
        while len(y) < 17:
            y.append(np.nan)
        yerr = [
            np.std(result[key]) if result[key] is not np.nan else np.nan
            for key in result.keys()
        ]
        while len(yerr) < 17:
            yerr.append(np.nan)
        xs.append(x)
        ys.append(y)
        yerrors.append(yerr)

    if not config["only_transformer"]:
        # color = ["blue", "orange", "green", "orange", "green"]
        # marker = ["o", "o", "o", "x", "x"]
        for i in range(N_probes):
            ax.errorbar(
                xs[i],
                ys[i],
                yerrors[i],
                linestyle="--",
                # marker=marker[i],
                marker="o",
                label=config["labels"][i],
                # color=color[i],
            )
        plt.xticks(range(17), labels=layer2name.values(), rotation=45)
        plt.suptitle(fig_title)
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
        plt.suptitle(f"{fig_title} - transformer only")

    # plt.axhline(y=0.25, color="orange", linestyle="--")
    # plt.axhline(y=0.2, color="blue", linestyle="--")
    ax.legend()

    if config["entire_y_axis"]:
        plt.ylim([0, 1.0])

    if config["save"]:
        if config["img_name"] is not None:
            imgname = "../plots/" + config["img_name"]
        else:
            if N_probes == 4:
                imgname = config["filenames"][1].replace("../results/test_results_", "")
                imgname = imgname.replace(
                    ".pickle",
                    f'{"_transformeronly" if config["only_transformer"] else ""}',
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
                    ".pickle",
                    f'{"_transformeronly" if config["only_transformer"] else ""}',
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
        plt.savefig(imgname, bbox_inches="tight")

    plt.show()

    return results


if __name__ == "__main__":
    config = {
        "no_plots": 1,
        "filenames": [
            # "../results/new_test_results_MLP2_pos_binding_problem_unbalanced_filtered_{30}_no_layernorm.pickle",
            # "../results/test_results_MLP2_sup1_binding_problem_filtered_{30}_no_layernorm_mode{args.mode}.pickle",
            # "../results/test_results_MLP2_posmo_binding_problem_filtered_{30}_no_layernorm_mode{args.mode}.pickle",
            # "../results/test_results_MLP2_sup1mo_binding_problem_filtered_{30}_no_layernorm_mode{args.mode}.pickle",
            # "../results/test_results_MLP2_pos_binding_problem_filtered_{30}_no_layernorm_mode{args.mode}.pickle",
            # "../results/test_results_MLP2_pos_binding_problem_filtered_{30}_no_layernorm_mode:normal_with_black?.pickle",
            # "../results/test_results_MLP2_trainedonpos_testedonsup1_binding_problem_filtered_{30}_no_layernorm_mode{args.mode}.pickle",
            # "../results/test_results_MLP2_trainedonpos_testedonposmo_binding_problem_filtered_{30}_no_layernorm_mode{args.mode}.pickle",
            # "../results/test_results_MLP2_trainedonpos_testedonsup1mo_binding_problem_filtered_{30}_no_layernorm_mode{args.mode}.pickle",
            "../results/test_results_linear_layer_pos_n_objects_balanced_unfiltered_no_layernorm.pickle",
            "../results/test_results_MLP2_pos_n_objects_balanced_unfiltered_no_layernorm.pickle",
        ],
        "labels": [
            # "no color",
            # "no shape",
            # "original",
            # "no color",
            # "no shape",
            # "no color (1 iter)",
            # "no shape (1 iter)",
            # "linear probe",
            # "MLP probe",
            # '"poly" kernel',
            # "diff color, diff shape",
            # "diff color, diff shape v2",
            # "diff color, same shape",
            # "same color, diff shape",
            # "same color, same shape",
            "linear",
            "non-linear",
        ],
        # "labels": ["MLP (sup1)"],
        "fig_title": "Probe accuracy on number of objects",
        "save": False,
        "img_name": "n_objects.png",  # else: None
        "only_transformer": False,
        "entire_y_axis": True,
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
