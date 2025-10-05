import argparse

import torch, torch.nn as nn
from omegaconf import OmegaConf
from src.registry import build
from src.utils.seed import set_seed
from src.aug.augmentaion import build_transform

import src.datasets
import src.models
import src.optim.optimizer
import src.scheduler.scheduler
import src.trainer
import src.loss.loss
import src.logger
import src.aug.batch
from src.hook.checkpoint_saver import CheckpointSaver
from src.hook.early_stopping import EarlyStopping

# from src.engine.hooks import EarlyStopping

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ML Training')
    parser.add_argument('--cfg', default='configs/default.yaml', type=str,
                        help='cfg path')
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg)
    set_seed(cfg.seed)

    device = cfg.device if torch.cuda.is_available() else "cpu"
    
    # 1) 증강
    train_tf = build_transform(getattr(getattr(cfg, "augment", None), "train", []))
    eval_tf  = build_transform(getattr(getattr(cfg, "augment", None), "eval", [])) 

    batch_aug = None
    if hasattr(cfg, "batch_aug"):
        ba = cfg.batch_aug
        batch_aug = build(
            "batch_aug", ba.name, 
            **{k:v for k,v in ba.items() if k!="name"}
        )

    # 2) 데이터로더
    train_loader, val_loader = build(
        "dataset", cfg.dataset.name, 
        train_tf=train_tf,
        eval_tf=eval_tf,
        cfg = cfg.dataset
    )

    # 3) 모델/옵티마이저/스케줄러
    model = build(
        "model", cfg.model.name, 
        **{k:v for k,v in cfg.model.items() if k!="name"}
    )

    optimizer = build(
        "optimizer", cfg.optimizer.name, 
        params=model.parameters(), 
        **{k:v for k,v in cfg.optimizer.items() if k!="name"}
    )

    scheduler = build(
        "scheduler", cfg.scheduler.name, 
        optimizer=optimizer, 
        **{k:v for k,v in cfg.scheduler.items() if k!="name"}
    )


    if hasattr(cfg, "checkpoint_loader"):
        ckp_path = cfg.checkpoint_loader.path
        checkpoint = torch.load(ckp_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        if checkpoint["optimizer"] is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])

        if checkpoint["scheduler"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])


    # 4) 로거
    logger = build(
        "logger", cfg.logger.name, 
        **{k:v for k,v in cfg.logger.items() if k!="name"}
    )
    
    hooks = [CheckpointSaver(**cfg.checkpoint_saver)]
    if hasattr(cfg, "early_stop"):
        hooks.append(EarlyStopping(**cfg.early_stop))

    # 5) 트레이너
    trainer = build(
        "trainer", cfg.trainer.name, 
        model=model, optimizer=optimizer, scheduler=scheduler, logger=logger, 
        hooks=hooks,
        batch_aug=batch_aug,
        cfg=cfg.trainer
    )
        
    # 6) 손실함수
    criterion = build(
        "loss", cfg.loss.name, 
        **{k:v for k,v in cfg.loss.items() if k!="name"}
    )

    trainer.train(train_loader, val_loader, criterion, device)
    logger.finalize(status="success")
