import argparse

import torch, torch.nn as nn
from omegaconf import OmegaConf
from src.registry import build
from src.utils.seed import set_seed
from src.aug.augmentaion import build_transform

import src.datasets
# from src.engine.trainer import Trainer
# from src.engine.hooks import EarlyStopping

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ML Training')
    parser.add_argument('--cfg', default='configs/default.yaml', type=str,
                        help='cfg path')
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg)
    set_seed(cfg.seed)

    # 1) 증강
    train_tf = build_transform(cfg.augment.train)
    eval_tf  = build_transform(cfg.augment.eval)
    print("#### train_tf ####")
    print(train_tf)
    print("#### eval_tf ####")
    print(eval_tf)    

    # 2) 데이터로더
    print(cfg.dataset)
    train_loader, val_loader = build(
        "dataset", cfg.dataset.name, 
        train_tf=train_tf,
        eval_tf=eval_tf,
        cfg = cfg.dataset
    )
    print("#### train_loader ####")
    print(train_loader)
    print("#### val_loader ####")
    print(val_loader)

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

    # 4) 로거
    logger = build(
        "logger", cfg.logger.name, 
        **{k:v for k,v in cfg.logger.items() if k!="name"}
    )

    # 5) 트레이너
    trainer = build(
        "trainer", cfg.trainer.name, 
        model=model, optimizer=optimizer, scheduler=scheduler, logger=logger, hooks=[],
        cfg=cfg
    )
    # trainer = Trainer(model, optimizer, scheduler, logger, hooks=[EarlyStopping(**cfg.train.early_stop)], cfg=cfg)
    
    # 6) 손실함수
    criterion = build(
        "loss", cfg.loss.name, 
        **{k:v for k,v in cfg.loss.items() if k!="name"}
    )
    # criterion = nn.CrossEntropyLoss()

    device = cfg.device if torch.cuda.is_available() else "cpu"
    trainer.train(train_loader, val_loader, criterion, device)

    logger.finalize()
