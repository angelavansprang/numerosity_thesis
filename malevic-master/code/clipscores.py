import sys
import torch
from collections import defaultdict
import argparse
from PIL import Image
import pickle
import visualize_result
import numpy as np
import random

import utils
import utils_amnesic_probing
import matplotlib.pyplot as plt

sys.path.append("../../")
import Transformer_MM_Explainability.CLIP.clip as clip


# def get_clipscore(model, preprocess, texts, img_repr, from_step):
#     """Get CLIP scores of texts and and img.
#     Input:
#     texts (list): containing strings
#     img (Pillow img): retrieve with Image.open(img_path)
#     from_step (int): next layer from the ViT for the image representation
#     """
#     text = clip.tokenize(texts).to(device)
#     text_features = model.encode_text(text)

#     # img = preprocess(img).unsqueeze(0).to(device)
#     img_features = encode_image(img_repr, device, model, preprocess, from_step)

#     cosi = torch.nn.CosineSimilarity()
#     output = cosi(img_features, text_features)

#     del image_features
#     del img
#     del text
#     del text_features
#     torch.cuda.empty_cache()
#     return output

#     # originally:
#     logits_image, logits_text = model(
#         img, text
#     )  # logits_image and logits_text are equal(?)

#     return logits_image


# def get_probs_from_logits(logits):
#     probs = logits.softmax(dim=-1)
#     return [prob.item() for prob in probs[0]]


# TODO: Make pipeline such that an image can enter anytime to be processed into final representation,
# based on the layer it came from


def encode_image(img, device, model, preprocess, from_step=1):
    """input:
    model (CLIP instance)
    """
    z = img.type(model.visual.conv1.weight.dtype)
    # if from_step == 0:
    # img = preprocess(img).unsqueeze(0).to(device)

    # # First step: make patches by a 2d convolution with a kernel_size and stride of 32 (the patch size)
    # # here, we get 768 patches of 7x7 pixels
    # z = model.visual.conv1(z)  # shape = [*, width, grid, grid]

    # # Second step: concatenate embeddings
    # z = z.reshape(z.shape[0], z.shape[1], -1)  # shape = [*, width, grid ** 2]
    # z = z.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
    # print("1) Shape z: ", z.shape)

    # z = torch.cat(
    #     [
    #         model.visual.class_embedding.to(z.dtype)
    #         + torch.zeros(
    #             z.shape[0], 1, z.shape[-1], dtype=z.dtype, device=z.device
    #         ),
    #         z,
    #     ],
    #     dim=1,
    # )  # shape = [*, grid ** 2 + 1, width]
    if from_step == 1:
        # Third step: add positional embeddings
        z = torch.unsqueeze(z, 0)
        z = z + model.visual.positional_embedding.to(z.dtype)
    if from_step == 1 or from_step == 2:
        if len(z.shape) == 2:
            z = torch.unsqueeze(z, 1)
        # Fourth step: layer normalization
        z = model.visual.ln_pre(z)

        # Fifth step: through the transformer; maybe exploit this further?
        # !! Info, there are 12 layers in here
        # print("2) Shape z: ", z.shape)
        z = z.permute(1, 0, 2)  # NLD -> LND

    # if from_step < 4:

    for i, block in enumerate(
        model.visual.transformer.resblocks
    ):  # deze loop vervangt:       z = model.visual.transformer(z)
        if not from_step < 4 + i:  # TODO: check if this works!
            continue
        if len(z.shape) == 2:
            z = torch.unsqueeze(z, 1)
        z = block(z)

    if from_step < 15:
        if len(z.shape) == 2:
            z = torch.unsqueeze(z, 1)
        z = z.permute(1, 0, 2)  # LND -> NLD

        # Sixth step: another layer normalization
        z = model.visual.ln_post(z[:, 0, :])

    if from_step < 16:
        if len(z.shape) == 2:
            z = torch.unsqueeze(z, 1)
        # Seventh step: project back
        if model.visual.proj is not None:
            z = z @ model.visual.proj

    z = torch.squeeze(z, 0)
    return z


def experiment_per_layer(
    dataset, split, amnesic_obj=None, to_save=False, first_projection_only=False
):
    max_layers = 15

    repr_path = f"../data/{dataset}/representations/{dataset}_{split}_visual.pickle"
    with open(repr_path, "rb") as f:
        repr = pickle.load(f)

    results = defaultdict(lambda: [])
    for layer in range(max_layers):
        if amnesic_obj is not None:
            if not first_projection_only:
                P = utils_amnesic_probing.open_intersection_nullspaces(
                    dataset, amnesic_obj, layer
                )
            else:
                P = utils_amnesic_probing.open_first_rowspace_projection(
                    dataset, amnesic_obj, layer
                )
            P = torch.from_numpy(P)
            P = P.to(torch.float32)
        for img_id, reprs_img in repr.items():
            if layer == 0 or layer == 1:
                reprs_img = [reprs_img[layer][0][i] for i in range(50)]
            else:
                reprs_img = [reprs_img[layer][i][0] for i in range(50)]
            if amnesic_obj is not None:
                amnesic_repr = []
                for z in reprs_img:
                    z = torch.from_numpy(z)
                    z = torch.unsqueeze(z, 0)
                    z = z.to(torch.float32)
                    z = z @ P
                    amnesic_z = z.flatten()
                    amnesic_repr.append(amnesic_z)
                reprs_img = amnesic_repr
            # print("Size one patch: ", reprs_img[0].shape)
            else:
                reprs_img = [torch.from_numpy(z) for z in reprs_img]
            reprs_img = torch.stack(
                reprs_img
            )  # TODO: MAKE LIST OF PATCHES BACK INTO ONE IMAGE
            # print("Size after stacking: ", reprs_img.shape)
            reprs_img = reprs_img.to(device)
            extrapolated_img = encode_image(
                reprs_img, device, model, preprocess, from_step=layer + 1
            )
            outcome = single_experiment(model, extrapolated_img)
            results[layer].append(outcome)

    if to_save:
        results = dict(results)
        # file_path = f'../results/clip_{dataset}_{split}{"_amnesic" + str(amnesic_obj) if amnesic_obj is not None else ""}.pickle'
        file_path = f'../results/clip_{dataset}_{split}{"_amnesic" + str(amnesic_obj) if amnesic_obj is not None else ""}{"_firstprojectiononly" if first_projection_only else ""}.pickle'
        with open(file_path, "wb") as f:
            pickle.dump(results, f)

    return results


def experiment_final_layer(
    dataset,
    split,
    objective,
    to_save=False,
):
    final_layer = 16

    repr_path = f"../data/{dataset}/representations/{dataset}_{split}_visual.pickle"
    with open(repr_path, "rb") as f:
        repr = pickle.load(f)

    results = defaultdict(lambda: [])
    labels = utils.make_labels_dict(dataset, split)

    idx = 0
    for img_id, reprs_img in repr.items():
        if idx % 100 == 0:
            print(f"{idx}/{len(repr)}")
        img = torch.from_numpy(reprs_img[final_layer])
        texts = []
        words = " " + objective.replace("n_", "")
        correct_n = int(labels[int(img_id)][objective])
        if objective == "n_objects":
            incorrect_n = random.choice([correct_n + i for i in [-3, -2, -1, 1, 2, 3]])
        elif objective == "n_colors":
            incorrect_n = random.choice([correct_n + i for i in [-2, -1, 1, 2]])
        texts.append(str(correct_n) + words)
        texts.append(str(incorrect_n) + words)
        # print(texts)
        # outcome = single_experiment(model, img, texts)
        outcome1, outcome2 = single_experiment_prob(model, img, device, texts)
        results[final_layer].append(np.absolute(outcome1 - outcome2))
        idx += 1

    if to_save:
        results = dict(results)
        # file_path = f'../results/clip_{dataset}_{split}{"_amnesic" + str(amnesic_obj) if amnesic_obj is not None else ""}.pickle'
        file_path = f"../results/clip_{dataset}_{split}_{objective}_differences.pickle"
        with open(file_path, "wb") as f:
            pickle.dump(results, f)

    return results


# TODO: Make function that returns indication whether representation img is still good for CLIP:
# argmax or so over different texts


def single_experiment(
    model,
    img_repr,
    texts=["geometric shapes", "mathematics", "city map", "piano", "revolution"],
):
    text = clip.tokenize(texts).to(device)
    text_features = model.encode_text(text)
    # print("shape text_features: ", text_features.shape)
    # print("shape img_features: ", img_repr.shape)

    cosi = torch.nn.CosineSimilarity()
    output = cosi(img_repr, text_features)

    probs = output.softmax(dim=-1)
    # print(probs)
    correct_outcome = torch.argmax(probs) == 0
    return 1 if correct_outcome else 0


def single_experiment_prob(
    model,
    img_repr,
    device,
    texts=["geometric shapes", "mathematics", "city map", "piano", "revolution"],
):
    text = clip.tokenize(texts).to(device)
    text_features = model.encode_text(text)
    # print("shape text_features: ", text_features.shape)
    # print("shape img_features: ", img_repr.shape)

    cosi = torch.nn.CosineSimilarity()
    output = cosi(img_repr, text_features)

    probs = output.softmax(dim=-1)
    return probs[0].item(), probs[1].item()


def open_results(dataset, split, amnesic_obj, first_projection_only):
    if amnesic_obj is None:
        file_path = f'../results/clip_{dataset}_{split}{"_amnesic" + str(amnesic_obj) if amnesic_obj is not None else ""}.pickle'
    else:
        file_path = f'../results/clip_{dataset}_{split}{"_amnesic" + str(amnesic_obj) if amnesic_obj is not None else ""}{"_firstprojectiononly" if first_projection_only else ""}.pickle'
    with open(file_path, "rb") as f:
        results = pickle.load(f)
    return results


def visualize_results(dataset, split, first_projection_only, to_save=False):
    results = {}
    results["orig"] = open_results(
        dataset, split, amnesic_obj=None, first_projection_only=first_projection_only
    )
    results["no color"] = open_results(
        dataset, split, amnesic_obj="color", first_projection_only=first_projection_only
    )
    results["no shape"] = open_results(
        dataset, split, amnesic_obj="shape", first_projection_only=first_projection_only
    )

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for name, result in results.items():
        x = [int(key) for key in result.keys()]
        while len(x) < 17:
            x.append(np.nan)
        y = [sum(values) / len(values) for values in result.values()]
        while len(y) < 17:
            y.append(np.nan)
        plt.plot(x, y, label=name)

    plt.xticks(range(17), labels=visualize_result.layer2name.values(), rotation=45)
    additional_name = "(first projection only)" if first_projection_only else ""
    plt.suptitle(f"Performance CLIP on MALeViC ({dataset}, {split}) " + additional_name)
    ax.legend()
    plt.xlabel("Layer")
    plt.ylabel("Accuracy")

    if to_save:
        imgname = f"../plots/clip_amnesic_{dataset}_{split}{additional_name}.png"
        plt.savefig(imgname, bbox_inches="tight")

    # plt.show()


def show_example(img_id, split, dataset, objective, model, device):
    repr_path = f"../data/{dataset}/representations/{dataset}_{split}_visual.pickle"
    with open(repr_path, "rb") as f:
        repr = pickle.load(f)

    labels = utils.make_labels_dict(dataset, split)
    reprs_img = repr[img_id]
    img = torch.from_numpy(reprs_img[16])
    texts = []
    words = " " + objective.replace("n_", "")
    correct_n = int(labels[int(img_id)][objective])
    incorrect_n = random.choice([correct_n + i for i in [-2, -1, 1, 2]])
    texts.append(str(correct_n) + words)
    texts.append(str(incorrect_n) + words)
    outcome = single_experiment_prob(model, img, device, texts)
    img_path = f"../data/{dataset}/images/{split}/{img_id}.png"
    img = np.array(Image.open(img_path))
    plot_probs(img, texts, outcome)


def plot_probs(img, texts, probs):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    # fig.suptitle("Zero-Shot Classification MALeViC with CLIP")

    ax1.imshow(img)
    ax1.axis("off")

    y = np.arange(2)
    ax2.grid()
    ax2.barh(y, probs, color="m", height=0.2),
    ax2.invert_yaxis()
    ax2.set_axisbelow(True)
    ax2.set_yticks(y)
    ax2.set_yticklabels(["true", "foil"])
    ax2.set_xlabel("Probability")
    ax2.grid()

    for i, bar in enumerate(ax2.barh(y, probs)):
        width = bar.get_width() / 2
        label_y = bar.get_y() + bar.get_height() / 2
        ax2.text(width, label_y, s=texts[i])

    plt.show()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = utils.get_model_preprocess(device, model_type="ViT-B/32")

    parser = argparse.ArgumentParser(
        description="Get CLIP scores of texts and (amnesic) images"
    )
    parser.add_argument("--dataset", choices=["sup1", "pos"], required=True)
    parser.add_argument(
        "--split",
        choices=["train", "test", "val"],
        required=True,
    )
    parser.add_argument(
        "--objective", choices=["n_objects", "n_colors"], default="n_objects"
    )
    parser.add_argument("--amnesic_obj", choices=["shape", "color"], default=None)
    parser.add_argument("--to_save", action="store_true")
    parser.add_argument("--first_projection_only", action="store_true")
    args = parser.parse_args()

    print("Started with experiment")
    results = experiment_final_layer(
        args.dataset,
        args.split,
        args.objective,
        args.to_save,
    )
    # results = experiment_per_layer(
    #     dataset=args.dataset,
    #     split=args.split,
    #     amnesic_obj=args.amnesic_obj,
    #     to_save=args.to_save,
    #     first_projection_only=args.first_projection_only,
    # )
    print(
        f"Results {args.dataset}, {args.split}, {args.amnesic_obj}, {args.first_projection_only}: "
    )
    for layer, outcomes in results.items():
        print(f"    Accuracy layer {layer}: {sum(outcomes)/len(outcomes)}")

    # visualize_results(
    #     dataset=args.dataset,
    #     split=args.split,
    #     first_projection_only=args.first_projection_only,
    #     to_save=True,
    # )
    # visualize_results(
    #     dataset="pos",
    #     split="train",
    #     first_projection_only=True,
    #     to_save=True,
    # )
    # visualize_results(
    #     dataset="pos",
    #     split="train",
    #     first_projection_only=False,
    #     to_save=True,
    # )


# def compute_cos_sim_imgandtext(img_path, text_prompt, to_print=False):
#     img = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
#     image_features = model.encode_image(img)

#     text = clip.tokenize(text_prompt).to(device)
#     text_features = model.encode_text(text)

#     cosi = torch.nn.CosineSimilarity()
#     output = cosi(image_features, text_features)

#     if to_print:
#         print("\n Computed Cosine Similarity: ", output)

#     del image_features
#     del img
#     del text
#     del text_features
#     torch.cuda.empty_cache()
#     return output


# def get_image_malevic(img_id, img_folder="/content/drive/MyDrive/AA MSc Thesis/Data/MALeViC/pos/images/test/"):
#   img_path = img_folder + str(img_id) + ".png"
#   img = Image.open(img_path)
#   return img
