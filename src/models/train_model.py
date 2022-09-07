from datetime import datetime

import click
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything

from src.config import *
from src.data.dataloader import OHLDataModule
from src.models.classifier import OHLModel


@click.command()
def train():
    seed_everything(SEED, workers=True)

    data_module = OHLDataModule(
        data_path=PATH_DATASET,
        max_length=MAX_LENGTH,
        train_bs=TRAIN_BATCH_SIZE,
        valid_bs=VALID_BATCH_SIZE,
        test_size=TEST_SIZE,
        model_name=MODEL_NAME,
        aug=True,
    )
    data_module.setup()

    model = OHLModel(
        model_name=MODEL_NAME,
        lr_top=LR_TOP,
        lr_bottom=LR_BOTTOM,
        n_classes=data_module.num_classes,
        class_names=data_module.class_names,
    )

    current_datetime = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_last=True,
        dirpath="./models/chkpts/" + current_datetime,
        filename="{epoch}-{val_f1:.2f}-{val_accuracy:.2f}-{val_loss:.4f}",
    )

    # run_id = None
    wandb_logger = WandbLogger(
        project="OHL",
        log_model="False",
        # id=run_id,
        name="{}_{}".format(model.hparams.model_name, current_datetime),
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        precision=16,
        gpus=1,
        callbacks=[checkpoint_callback, lr_monitor],
        max_epochs=N_EPOCHS,
        log_every_n_steps=6,
        deterministic=True,
        logger=wandb_logger,
        # val_check_interval=1000,
    )

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    train()
