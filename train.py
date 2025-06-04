import os
from pprint import pprint
from configs.config import parser
from dataset.data_module import DataModule
from lightning_tools.callbacks import add_callbacks
from models.MedKit import MedKit
from models.MedKit_BASE import MedKit_BASE
from lightning.pytorch import seed_everything
import lightning.pytorch as pl

 
def train(args):
    dm = DataModule(args)
    callbacks = add_callbacks(args)

    trainer = pl.Trainer(
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=args.strategy,
        accelerator=args.accelerator,
        precision=args.precision,
        val_check_interval = args.val_check_interval,
        limit_val_batches = args.limit_val_batches,
        max_epochs = args.max_epochs,
        num_sanity_val_steps = args.num_sanity_val_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks["callbacks"], 
        logger=callbacks["loggers"]
    )



    if args.ckpt_file is not None:
        if args.model_type == 'base_cls':
            model = MedKit_BASE.load_from_checkpoint(args.ckpt_file, strict=False)
        else:
            model = MedKit.load_from_checkpoint(args.ckpt_file, strict=False)
    else:
        if args.model_type == 'base_cls':
            model = MedKit_BASE(args)
        else:
            model = MedKit(args)

    if args.test:
        model.eval()
        trainer.test(model, datamodule=dm)
    elif args.validate:
        trainer.validate(model, datamodule=dm)
    else:
        trainer.fit(model, datamodule=dm)

def main():
    args = parser.parse_args()
    os.makedirs(args.savedmodel_path, exist_ok=True)
    pprint(vars(args))
    seed_everything(44, workers=True)
    train(args)


if __name__ == '__main__':
    main()