# ┌─────────────────────────────┐
# │ attempt for amnesic probing │
# └─────────────────────────────┘

import utils
import models
import utils_amnesic_probing
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# 1. Open representations

dataset = "pos"
layer = "12"
objective = "color"
balanced = False
batch_size = 10

X_train, y_train, class2label = utils_amnesic_probing.build_dataloader_patchbased(
    dataset,
    layer,
    objective,
    split="train",
    balanced=balanced,
    batch_size=10,
    threshold=30,
)
X_val, y_val, class2label = utils_amnesic_probing.build_dataloader_patchbased(
    dataset,
    layer,
    objective,
    split="val",
    balanced=balanced,
    batch_size=10,
    threshold=30,
)
X_test, y_test, class2label = utils_amnesic_probing.build_dataloader_patchbased(
    dataset,
    layer,
    objective,
    split="test",
    balanced=balanced,
    batch_size=10,
    threshold=30,
)


loader_train = utils_amnesic_probing.build_dataloader(
    X_train, y_train, "train", batch_size
)
loader_val = utils_amnesic_probing.build_dataloader(X_val, y_val, "val", batch_size)
loader_test = utils_amnesic_probing.build_dataloader(X_test, y_test, "test", batch_size)


# 2. Make a linear classifier class

if objective == "color":
    D_out = 5
elif objective == "shape":
    D_out = 4

classifier = models.ProbingHead(D_in=768, D_out=D_out)  # this is a linear classifier


# 3. Train a linear classifier to predict Z (the property to remove)

trainer = pl.Trainer(
    accelerator="gpu",
    callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
    enable_progress_bar=False,
    log_every_n_steps=100,
)
train_info = trainer.fit(classifier, loader_train, loader_val)
performance = trainer.test(dataloaders=loader_test)


# 4. Project data onto its nullspace using the projection matrix of the trained classifier

W = classifier.coef_
P_rowspace_wi = utils.amnesic_probing.get_rowspace_projection(W)

X_train = X_train.dot(P_rowspace_wi)
X_val = X_val.dot(P_rowspace_wi)


# 5. Check whether it is now harder for a new classifier to predict color

loader_train = utils_amnesic_probing.build_dataloader(
    X_train, y_train, "train", batch_size
)
loader_val = utils_amnesic_probing.build_dataloader(X_val, y_val, "val", batch_size)

classifier = models.ProbingHead(D_in=768, D_out=D_out)  # this is a linear classifier
trainer = pl.Trainer(
    accelerator="gpu",
    callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
    enable_progress_bar=False,
    log_every_n_steps=100,
)
train_info = trainer.fit(classifier, loader_train, loader_val)
performance = trainer.test(dataloaders=loader_test)

# 5. Continue until last linear classifier achieves majority accuracy
