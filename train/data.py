"""ImageNet augmentation and validation preprocessing using timm."""

from timm.data import FastCollateMixup, Mixup, create_dataset, create_loader, resolve_data_config


def classification_loaders(config, model, distributed):
    data = resolve_data_config({"input_size": (3, config.image_size, config.image_size),
                                "crop_pct": config.crop_pct, "interpolation": "bicubic"}, model=model)
    train_set = create_dataset("", root=config.data_root, split="train", is_training=True,
                               batch_size=config.batch_size)
    validation_set = create_dataset("", root=config.data_root, split="validation", is_training=False,
                                    batch_size=config.validation_batch_size)
    mixup = collate = None
    mixing = config.mixup > 0 or config.cutmix > 0
    if mixing:
        arguments = dict(mixup_alpha=config.mixup, cutmix_alpha=config.cutmix, prob=1.0,
                         switch_prob=0.5, mode="batch", label_smoothing=config.smoothing,
                         num_classes=config.num_classes)
        if config.prefetcher:
            collate = FastCollateMixup(**arguments)
        else:
            mixup = Mixup(**arguments)
    # Disable persistent workers when collate-time mixing has an end epoch.
    persistent = config.workers > 0 and not (mixing and config.prefetcher and config.mixup_off_epoch)
    common = dict(input_size=data["input_size"], use_prefetcher=config.prefetcher,
                  mean=data["mean"], std=data["std"], num_workers=config.workers,
                  distributed=distributed, pin_memory=True)
    train_loader = create_loader(
        train_set, batch_size=config.batch_size, is_training=True,
        re_prob=0.25, re_mode="pixel", re_count=1, re_split=False,
        scale=(0.08, 1.0), ratio=(0.75, 4.0 / 3.0), hflip=0.5, vflip=0.0,
        color_jitter=0.4, auto_augment="rand-m9-mstd0.5-inc1",
        num_aug_repeats=0, num_aug_splits=0, interpolation=config.train_interpolation,
        collate_fn=collate, persistent_workers=persistent, **common,
    )
    validation_loader = create_loader(
        validation_set, batch_size=config.validation_batch_size, is_training=False,
        interpolation=data["interpolation"], crop_pct=data["crop_pct"],
        persistent_workers=config.workers > 0, **common,
    )
    if config.workers > 0:
        for loader in (train_loader, validation_loader):
            data_loader = loader.loader if config.prefetcher else loader
            data_loader.multiprocessing_context = "spawn"
    return train_loader, validation_loader, mixup, data
