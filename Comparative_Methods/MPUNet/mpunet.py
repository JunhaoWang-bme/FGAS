import os
import numpy as np
import nibabel as nib
from mpunet.hyperparameters import YAMLHParams
from mpunet.preprocessing.data_preparation_funcs import prepare_for_multi_view_unet
from mpunet.train import Trainer
from mpunet.evaluate import evaluate_folder
from mpunet.utils.fusion import predict_volume
from mpunet.logging import ScreenLogger


class FibroidConfig:
    def __init__(self):
        self.base_dir = r"./uterine_fibroid"
        self.data_dir = r"./data"
        self.train_hparams_path = os.path.join(self.base_dir, "train_hparams.yaml")

        self.data_structure = {
            "train": {"images": os.path.join(self.data_dir, "train/images"),
                      "labels": os.path.join(self.data_dir, "train/labels")},
            "val": {"images": os.path.join(self.data_dir, "val/images"),
                    "labels": os.path.join(self.data_dir, "val/labels")},
            "test": {"images": os.path.join(self.data_dir, "test/images"),
                     "labels": os.path.join(self.data_dir, "test/labels")}
        }

        self.model_params = {
            "build": {
                "n_channels": 1,
                "n_classes": 2,
                "dim": 256
            },
            "fit": {
                "views": 3,
                "batch_size": 8,
                "epochs": 100,
                "learning_rate": 1e-4,
                "real_space_span": 120
            },
            "augmentation": {
                "rotation_range": 15,
                "zoom_range": 0.1,
                "horizontal_flip": True
            }
        }


def prepare_fibroid_data(config):
    if not os.path.exists(config.base_dir):
        os.makedirs(config.base_dir)

    hparams = YAMLHParams(config.train_hparams_path, create=True)

    for phase in ["train", "val", "test"]:
        hparams[phase + "_data"] = {
            "image_dir": config.data_structure[phase]["images"],
            "label_dir": config.data_structure[phase]["labels"],
            "pattern": "*.nii.gz"
        }

    for key, val in config.model_params.items():
        hparams[key] = val

    hparams.save()
    return hparams


def train_fibroid_model(hparams, config):
    logger = ScreenLogger()
    logger("Starting uterine fibroid segmentation model training...")

    train_seq, val_seq, hparams = prepare_for_multi_view_unet(
        hparams=hparams,
        base_path=config.base_dir,
        logger=logger
    )

    trainer = Trainer(
        model=None,
        hparams=hparams,
        train_sequence=train_seq,
        val_sequence=val_seq,
        logger=logger,
        loss="sparse_exponential_logarithmic_loss"
    )

    trainer.train()
    return trainer.model


def predict_and_evaluate(model, config):
    logger = ScreenLogger()
    hparams = YAMLHParams(config.train_hparams_path)

    test_images = [os.path.join(config.data_structure["test"]["images"], f)
                   for f in os.listdir(config.data_structure["test"]["images"])
                   if f.endswith(".nii.gz")]

    pred_dir = os.path.join(config.base_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    for img_path in test_images:
        logger(f"Predicting image: {img_path}")
        pred = predict_volume(
            model=model,
            image_path=img_path,
            hparams=hparams,
            out_dir=pred_dir
        )
        pred_nii = nib.Nifti1Image(pred, affine=nib.load(img_path).affine)
        pred_path = os.path.join(pred_dir, os.path.basename(img_path))
        nib.save(pred_nii, pred_path)

    logger("Starting evaluation...")
    evaluate_folder(
        pred_dir=pred_dir,
        lab_dir=config.data_structure["test"]["labels"],
        out_dir=os.path.join(config.base_dir, "evaluation"),
        logger=logger
    )


if __name__ == "__main__":
    config = FibroidConfig()
    hparams = prepare_fibroid_data(config)
    model = train_fibroid_model(hparams, config)
    predict_and_evaluate(model, config)