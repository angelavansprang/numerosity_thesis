# see https://github.com/openai/CLIP/issues/83
import json
from PIL import Image
import sys

sys.path.append("../")
import Transformer_MM_Explainability.CLIP.clip as clip
from torch.utils.data import DataLoader
from num2words import num2words
import torch
import torch.nn as nn
import numpy as np
import argparse
import urllib.request
import random
import pickle
import os


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


def build_dataloaders(images, captions, foils, batch_size):
    # print("len(images) : ", len(images))
    # print("len(captions) :", len(captions))
    # print("len(foils): ", len(foils))
    dataset = list(zip(images, captions, foils))
    # print("len(dataset)", len(dataset))

    train_part = 0.8
    random.shuffle(dataset)
    idx = int(train_part * len(images))

    dataloader_train = DataLoader(dataset[:idx], batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset[idx:], batch_size=batch_size, shuffle=True)

    return dataloader_train, dataloader_val


def get_dataset_VALSE(device, preprocess):
    VALSE_data_location = "data/counting-hard.json"
    with open(VALSE_data_location) as json_file:
        foils_data = json.load(json_file)

    valid_data = {}
    for key, value in foils_data.items():
        if value["mturk"]["caption"] >= 2:
            valid_data[key] = value

    images = []
    captions = []
    foils = []
    global_img_path = "data/images_valse/"

    for key, value in valid_data.items():
        img_path = global_img_path + str(value["dataset_idx"]) + ".jpg"
        image = preprocess(Image.open(img_path)).to(device)

        caption = value["caption"]
        correct_num = value["classes"]
        correct_word = num2words(correct_num)
        caption = caption.replace(str(correct_num), correct_word)
        caption_tokenized = clip.tokenize(caption).to(device)

        foil = value["foil"]
        incorrect_num = value["classes_foil"]
        incorrect_word = num2words(incorrect_num)
        foil = foil.replace(str(incorrect_num), incorrect_word)
        foil_tokenized = clip.tokenize(foil).to(device)

        images.append(image)
        captions.append(caption_tokenized)
        foils.append(foil_tokenized)

    return images, captions, foils


def preprare_countbench():
    countbench_location = "data/CountBench.json"
    with open(countbench_location) as json_file:
        bench_data = json.load(json_file)

    overview = {}
    for i, dict in enumerate(bench_data):
        try:
            urllib.request.urlretrieve(
                dict["image_url"], f"data/images_countbench/{i}.png"
            )
        except:
            print("Error: ", dict["image_url"])
        else:
            overview[i] = dict["image_url"]
            try:
                img = Image.open(f"data/images_countbench/{i}.png")
            except:
                print("Error: image unusable")

    with open("data/countbench_ids.pickle", "wb") as handle:
        pickle.dump(overview, handle)


def get_dataset_countbench(device, preprocess):
    countbench_location = "data/CountBench.json"
    with open(countbench_location) as json_file:
        bench_data = json.load(json_file)

    images = []
    captions = []
    foils = []

    with open("data/countbench_ids.pickle", "rb") as handle:
        overview = pickle.load(handle)

    for img_name in os.listdir("data/images_countbench"):
        try:
            img = Image.open(f"data/images_countbench/{img_name}")
        except:
            pass
        else:
            image = preprocess(img).to(device)

            image_url = overview[int(img_name.replace(".png", ""))]
            dict, *rest = [
                dict for dict in bench_data if dict["image_url"] == image_url
            ]

            caption = dict["text"]
            try:
                caption_tokenized = clip.tokenize(caption).to(device)
            except:
                pass
            else:
                choices = list(range(2, 11))
                choices.remove(dict["number"])
                foil_int = random.choice(choices)
                foil = caption.replace(num2words(dict["number"]), num2words(foil_int))
                foil_tokenized = clip.tokenize(foil).to(device)

                images.append(image)
                captions.append(caption_tokenized)
                foils.append(foil_tokenized)

    return images, captions, foils


def get_dataset_malevic(device, preprocess, dataset="pos", objective="n_objects"):
    # data_location = f"../malevic-master/data/{dataset}/"
    data_location = f"../MALeViC/data/{dataset}/"

    images = []
    captions = []
    foils = []

    with open(data_location + "annotation/train_annotation.json", "r") as f:
        annotation = json.load(f)

    for img_name in os.listdir(data_location + "images/train/"):
        img_id = img_name.replace(".png", "")
        n_colors = int(annotation[img_id][0]["n_colors"])
        n_objects = int(annotation[img_id][0]["n_objects"])
        if objective == "n_colors":
            caption = f"{num2words(n_colors)} different colors"
            choices = list(range(2, 11))
            choices.remove(n_colors)
            foil_int = random.choice(choices)
            foil = caption.replace(num2words(n_colors), num2words(foil_int))
        if objective == "n_objects":
            caption = f"{num2words(n_objects)} geometric objects"
            choices = list(range(2, 11))
            choices.remove(n_objects)
            foil_int = random.choice(choices)
            foil = caption.replace(num2words(n_objects), num2words(foil_int))

        image = preprocess(Image.open(data_location + "images/train/" + img_name)).to(
            device
        )
        caption_tokenized = clip.tokenize(caption).to(device)
        foil_tokenized = clip.tokenize(foil).to(device)

        images.append(image)
        captions.append(caption_tokenized)
        foils.append(foil_tokenized)

    return images, captions, foils


def train(
    train_loader,
    val_loader,
    lamb,
    num_epochs,
    device,
    model,
    save_model,
    info_modelname,
):
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=5e-6) # original, but too small, introduces nan after 1st update
    optimizer = torch.optim.Adam(
        model.parameters(), lr=5e-5, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2
    )
    min_valid_loss = np.inf
    last_update_epoch = 0
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=)

    for epoch in range(num_epochs):
        train_loss = 0
        train_clip_loss = 0
        train_count_loss = 0
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()

            # images = torch.stack([items[0] for items in batch])
            # captions = torch.stack([items[1] for items in batch]).squeeze()
            # foils = torch.stack([items[2] for items in batch]).squeeze()
            images = batch[0]
            captions = batch[1].squeeze()
            foils = batch[2].squeeze()

            # print("train: images.shape: ", images.shape)
            # print("train: captions.shape: ", captions.shape)
            # print("train: foils.shape: ", foils.shape)
            # print(f"train {epoch}: images: ", images)
            # print(f"train {epoch}: captions: ", captions)
            # print("train: foils: ", foils)

            logits_per_image, logits_per_text = model(images, captions)

            # print(f"train {epoch}: logtis_per_image: ", logits_per_image)
            # print(f"train {epoch}: logtis_per_text: ", logits_per_text)

            logits_per_image = (
                logits_per_image.to(dtype=torch.float64) / model.logit_scale.exp()
            )
            logits_per_text = (
                logits_per_text.to(dtype=torch.float64) / model.logit_scale.exp()
            )

            ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
            # ground_truth = torch.ones(len(images), dtype=torch.long, device=device)

            clip_loss = (
                loss_img(logits_per_image, ground_truth)
                + loss_txt(logits_per_text, ground_truth)
            ) / 2

            # new loss component
            _, logits_per_foil = model(images, foils)
            logits_per_foil = (
                logits_per_foil.to(dtype=torch.float64) / model.logit_scale.exp()
            )

            # epsilon = 1e-7
            count_loss = -torch.mean(
                torch.log(
                    torch.exp(logits_per_text)
                    / (torch.exp(logits_per_text) + torch.exp(logits_per_foil))
                )
            )
            # count_loss = 0
            # for i in range(len(images)):
            #     eiet = logits_per_text[i]
            #     eietcf = logits_per_foil[i]
            #     count_loss -= torch.log(
            #         torch.exp(eiet) / (torch.exp(eiet) + torch.exp(eietcf))
            #     )
            # count_loss = count_loss / len(images)

            total_loss = clip_loss + lamb * count_loss
            # print("clip_loss: ", clip_loss)
            # print("count_loss: ", count_loss)
            # print("lamb: ", lamb)
            # print("total_loss: ", total_loss)
            # print("epoch: ", epoch, "; loss: ,", total_loss)
            train_loss += total_loss.item()
            train_count_loss += count_loss.item()
            train_clip_loss += clip_loss.item()
            # print("train_loss: ", train_loss)

            total_loss.mean().backward()
            optimizer.step()

            del images, captions, foils

            # train_loss += total_loss

        # before_lr = optimizer.param_groups[0]["lr"]
        # scheduler.step()
        # after_lr = optimizer.param_groups[0]["lr"]

        valid_loss = 0
        valid_clip_loss = 0
        valid_count_loss = 0
        model.eval()
        for batch in val_loader:
            # print("sizes batch: ", batch[0].shape, batch[1].shape, batch[2].shape)
            images = batch[0]
            captions = batch[1].squeeze()
            foils = batch[2].squeeze()

            # print("images.shape: ", images.shape)
            # print("captions.shape: ", captions.shape)
            # print("foils.shape: ", foils.shape)
            # print(f"images {epoch}: ", images)
            # print(f"captions {epoch}: ", captions)
            # print("foils: ", foils)

            logits_per_image, logits_per_text = model(images, captions)

            # print(f"eval {epoch}: logtis_per_image: ", logits_per_image)
            # print(f"eval {epoch}: logtis_per_text: ", logits_per_text)

            logits_per_image = (
                logits_per_image.to(dtype=torch.float64) / model.logit_scale.exp()
            )
            logits_per_text = (
                logits_per_text.to(dtype=torch.float64) / model.logit_scale.exp()
            )

            ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
            # ground_truth = torch.ones(len(images), dtype=torch.long, device=device)

            clip_loss = (
                loss_img(logits_per_image, ground_truth)
                + loss_txt(logits_per_text, ground_truth)
            ) / 2

            # new loss component
            _, logits_per_foil = model(images, foils)
            logits_per_foil = (
                logits_per_foil.to(dtype=torch.float64) / model.logit_scale.exp()
            )
            count_loss = -torch.mean(
                torch.log(
                    torch.exp(logits_per_text)
                    / (torch.exp(logits_per_text) + torch.exp(logits_per_foil))
                )
            )
            # print("logits_per_text: ", torch.exp(logits_per_text))
            # print("logits_per_foil: ", torch.exp(logits_per_foil))
            # print(
            #     "torch log: ",
            #     torch.log(
            #         torch.exp(logits_per_text)
            #         / (torch.exp(logits_per_text) + torch.exp(logits_per_foil))
            #     ),
            # )
            total_loss = clip_loss + lamb * count_loss

            valid_loss += total_loss.item()
            valid_clip_loss += clip_loss.item()
            valid_count_loss += count_loss.item()

            del images, captions, foils

            # print("training loss: ", train_loss)
            # print("len(dataloader): ", len(train_loader))
            # print(f"training loss: {train_loss.item()/len(train_loader)}")

        print(
            f"Epoch {epoch+1} \t\t Training Loss: {train_loss / len(train_loader)} \t\t Validation Loss: {valid_loss / len(val_loader)} \t\t Train Count Loss: {train_count_loss / len(train_loader)} \t\t Valid Count Loss: {valid_count_loss / len(val_loader)} \t\t Train Clip Loss: {train_clip_loss / len(train_loader)} \t\t Valid Clip Loss: {valid_clip_loss / len(val_loader)}"
        )

        # if device == "cpu":
        #     optimizer.step()
        # else:
        #     convert_models_to_fp32(model)
        #     optimizer.step()
        #     clip.model.convert_weights(model)

        if save_model:
            epsilon = 0.01
            if min_valid_loss > valid_loss + epsilon:
                print(
                    f"Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model"
                )
                min_valid_loss = valid_loss
                last_update_epoch = epoch

                # Saving State Dict
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": total_loss,
                    },
                    f"model_checkpoint/model_lambda{lamb}_{info_modelname}.pt",
                )  # just change to your preferred folder/filename

    if last_update_epoch - epoch > 50:
        return


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, preprocess = get_model_preprocess(device, model_type="ViT-B/32")

    parser = argparse.ArgumentParser(description="Train a probe on representations ViT")
    parser.add_argument("--lamb", default=1.0)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--valse", action="store_true")
    parser.add_argument("--countbench", action="store_true")
    parser.add_argument("--malevic", action="store_true")
    parser.add_argument("--mal_dataset", choices=["pos", "sup"], default="pos")
    parser.add_argument(
        "--mal_objective", choices=["n_colors", "n_objects"], default="n_objects"
    )
    args = parser.parse_args()

    images, captions, foils = [], [], []
    if args.valse:
        new_images, new_captions, new_foils = get_dataset_VALSE(device, preprocess)
        images.extend(new_images)
        captions.extend(new_captions)
        foils.extend(new_foils)
    if args.countbench:
        # preprare_countbench()
        new_images, new_captions, new_foils = get_dataset_countbench(device, preprocess)
        images.extend(new_images)
        captions.extend(new_captions)
        foils.extend(new_foils)
    if args.malevic:
        new_images, new_captions, new_foils = get_dataset_malevic(
            device, preprocess, dataset=args.mal_dataset, objective=args.mal_objective
        )
        images.extend(new_images)
        captions.extend(new_captions)
        foils.extend(new_foils)

    print(f"Number of images: {len(images)}")

    trainloader, valloader = build_dataloaders(images, captions, foils, args.batch_size)
    print("len trainloader: ", len(trainloader))
    print("len valloader: ", len(valloader))

    train(
        trainloader,
        valloader,
        args.lamb,
        args.num_epochs,
        device,
        model,
        args.save_model,
        info_modelname=f"{'valse_' if args.valse else ''}{'countbench_' if args.countbench else ''}{'malevic' if args.malevic else ''}",
    )
