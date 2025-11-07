# train.py
import argparse
import os
import os.path as op
import time
from contextlib import nullcontext

import torch
from torch.cuda.amp import autocast, GradScaler

from datasets import build_dataloader
from model import build_model
from solver import build_optimizer, build_lr_scheduler
from utils.checkpoint import Checkpointer
from utils.comm import is_main_process, reduce_dict
from utils.iotools import load_train_configs, save_train_configs
from utils.logger import setup_logger
from utils.meter import AverageMeter
from utils.metrics import Evaluator


# ----------------------------
# CLI
# ----------------------------
def build_parser():
    parser = argparse.ArgumentParser(description="ReID5o Training Script (auto log dir under log/)")
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the checkpoint specified in the configuration.",
    )
    parser.add_argument(
        "--resume_ckpt_file",
        type=str,
        default=None,
        help="Path to a checkpoint to resume from. Overrides the config if provided.",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip training and run evaluation only.",
    )
    return parser


def parse_arguments():
    parser = build_parser()
    args = parser.parse_args()
    return args, parser


# ----------------------------
# Utils
# ----------------------------
def allocate_run_dir(base_dir: str = "log") -> str:
    """
    Create and return the first non-existing numbered directory under base_dir.
    i.e., log/0, log/1, ...  The first gap will be created and returned.
    """
    os.makedirs(base_dir, exist_ok=True)
    i = 0
    while True:
        cand = op.join(base_dir, str(i))
        if not op.exists(cand):
            os.makedirs(cand, exist_ok=False)
            return cand
        i += 1


def setup_environment(args):
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger("ORBench", save_dir=args.output_dir, if_train=args.training)
    logger.info("Using device: %s", "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(args)
    return logger


def move_batch_to_device(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def reduce_loss_dict(loss_dict):
    if not loss_dict:
        return loss_dict
    loss_dict_tensor = {k: v.detach() for k, v in loss_dict.items() if isinstance(v, torch.Tensor)}
    reduced = reduce_dict(loss_dict_tensor)
    return {k: v.item() for k, v in reduced.items()}


def build_query_loader_dict(loaders):
    (
        train_loader,
        test_gallery_loader,
        nir_query_loader,
        cp_query_loader,
        sk_query_loader,
        text_query_loader,
        nir_cp_query_loader,
        cp_nir_query_loader,
        nir_sk_query_loader,
        sk_nir_query_loader,
        nir_text_query_loader,
        text_nir_query_loader,
        cp_sk_query_loader,
        sk_cp_query_loader,
        cp_text_query_loader,
        text_cp_query_loader,
        sk_text_query_loader,
        text_sk_query_loader,
        nir_cp_sk_query_loader,
        cp_nir_sk_query_loader,
        sk_nir_cp_query_loader,
        nir_cp_text_query_loader,
        cp_nir_text_query_loader,
        text_nir_cp_text_query_loader,
        nir_sk_text_query_loader,
        sk_nir_text_query_loader,
        text_nir_sk_text_query_loader,
        cp_sk_text_query_loader,
        sk_cp_text_query_loader,
        text_cp_sk_text_query_loader,
        nir_cp_sk_text_query_loader,
        cp_nir_sk_text_query_loader,
        sk_nir_cp_text_query_loader,
        text_nir_cp_sk_text_query_loader,
        num_classes,
    ) = loaders

    query_loaders = {
        "nir_query_loader": nir_query_loader,
        "cp_query_loader": cp_query_loader,
        "sk_query_loader": sk_query_loader,
        "text_query_loader": text_query_loader,
        "nir_cp_query_loader": nir_cp_query_loader,
        "cp_nir_query_loader": cp_nir_query_loader,
        "nir_sk_query_loader": nir_sk_query_loader,
        "sk_nir_query_loader": sk_nir_query_loader,
        "nir_text_query_loader": nir_text_query_loader,
        "text_nir_query_loader": text_nir_query_loader,
        "cp_sk_query_loader": cp_sk_query_loader,
        "sk_cp_query_loader": sk_cp_query_loader,
        "cp_text_query_loader": cp_text_query_loader,
        "text_cp_query_loader": text_cp_query_loader,
        "sk_text_query_loader": sk_text_query_loader,
        "text_sk_query_loader": text_sk_query_loader,
        "nir_cp_sk_query_loader": nir_cp_sk_query_loader,
        "cp_nir_sk_query_loader": cp_nir_sk_query_loader,
        "sk_nir_cp_query_loader": sk_nir_cp_query_loader,
        "nir_cp_text_query_loader": nir_cp_text_query_loader,
        "cp_nir_text_query_loader": cp_nir_text_query_loader,
        "text_nir_cp_text_query_loader": text_nir_cp_text_query_loader,
        "nir_sk_text_query_loader": nir_sk_text_query_loader,
        "sk_nir_text_query_loader": sk_nir_text_query_loader,
        "text_nir_sk_text_query_loader": text_nir_sk_text_query_loader,
        "cp_sk_text_query_loader": cp_sk_text_query_loader,
        "sk_cp_text_query_loader": sk_cp_text_query_loader,
        "text_cp_sk_text_query_loader": text_cp_sk_text_query_loader,
        "nir_cp_sk_text_query_loader": nir_cp_sk_text_query_loader,
        "cp_nir_sk_text_query_loader": cp_nir_sk_text_query_loader,
        "sk_nir_cp_text_query_loader": sk_nir_cp_text_query_loader,
        "text_nir_cp_sk_text_query_loader": text_nir_cp_sk_text_query_loader,
    }

    return train_loader, test_gallery_loader, query_loaders, num_classes


def run_evaluation(model, evaluator, epoch, logger, device):
    logger.info("Start evaluating at epoch %d", epoch)
    model.eval()
    with torch.no_grad():
        top1 = evaluator.eval(model)
    logger.info("Evaluation finished: Top-1 Average = %.3f", top1)
    model.train()
    return top1


# ----------------------------
# Train / Eval
# ----------------------------
def do_train(args, logger):
    # dataloaders
    loaders = build_dataloader(args)
    train_loader, test_gallery_loader, query_loaders, num_classes = build_query_loader_dict(loaders)

    # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args, num_classes=num_classes).to(device)

    # optimizer / scheduler
    optimizer = build_optimizer(args, model)
    scheduler = build_lr_scheduler(args, optimizer)

    # ckpt manager
    checkpointer = Checkpointer(
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        save_dir=args.output_dir,
        save_to_disk=is_main_process(),
        logger=logger,
    )

    start_epoch = 0
    best_top1 = 0.0

    # resume logic
    resume_ckpt = None
    if args.resume_ckpt_file:
        resume_ckpt = args.resume_ckpt_file
    elif getattr(args, "resume", False):
        resume_ckpt = args.resume_ckpt_file or op.join(args.output_dir, "last.pth")

    if resume_ckpt and os.path.isfile(resume_ckpt):
        logger.info("Resuming training from %s", resume_ckpt)
        checkpoint_data = checkpointer.resume(resume_ckpt)
        start_epoch = checkpoint_data.get("epoch", 0)
        best_top1 = checkpoint_data.get("best_top1", 0.0)
    elif resume_ckpt:
        logger.warning("Requested checkpoint %s not found, starting from scratch.", resume_ckpt)

    evaluator = Evaluator(
        gallery_loader=test_gallery_loader,
        get_mAP=True,
        **query_loaders,
    )

    loss_meter = AverageMeter()
    model.train()

    # AMP (autocast only if CUDA available)
    use_amp = torch.cuda.is_available()
    autocast_ctx = autocast if use_amp else nullcontext
    scaler = GradScaler(enabled=use_amp)

    for epoch in range(start_epoch, args.num_epoch):
        epoch_start = time.time()
        loss_meter.reset()

        for iteration, batch in enumerate(train_loader, 1):
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            with autocast_ctx(dtype=torch.float16):
                loss_dict = model(batch)
                losses = [value for key, value in loss_dict.items() if "loss" in key]
                if not losses:
                    continue
                total_loss = torch.stack(losses).sum()

            if use_amp:
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()

            loss_meter.update(total_loss.item(), batch["pids"].size(0))

            if iteration % args.log_period == 0:
                reduced_losses = reduce_loss_dict(loss_dict)
                log_items = ", ".join(
                    f"{k}: {v:.4f}" for k, v in reduced_losses.items() if "loss" in k
                )
                current_lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    "Epoch[%d/%d] Iter[%d/%d] loss: %.4f (%s) lr: %.6f",
                    epoch + 1,
                    args.num_epoch,
                    iteration,
                    len(train_loader),
                    loss_meter.avg,
                    log_items,
                    current_lr,
                )

        # epoch end
        scheduler.step()
        epoch_time = time.time() - epoch_start
        logger.info("Epoch %d finished in %.2f seconds", epoch + 1, epoch_time)

        # evaluation schedule
        should_eval = args.eval_period == -1 or (
            args.eval_period > 0 and (epoch + 1) % args.eval_period == 0
        )
        if should_eval:
            top1 = run_evaluation(model, evaluator, epoch + 1, logger, device)
        else:
            top1 = -1

        # save last
        checkpoint_data = {"epoch": epoch + 1, "best_top1": best_top1}
        checkpointer.save("last", **checkpoint_data)

        # save best
        if top1 > best_top1:
            best_top1 = top1
            checkpoint_data["best_top1"] = best_top1
            logger.info("New best Top-1 Average: %.3f", best_top1)
            checkpointer.save("best", **checkpoint_data)

    logger.info("Training finished. Best Top-1 Average: %.3f", best_top1)


def main():
    cli_args, parser = parse_arguments()
    cfg = load_train_configs(cli_args.config_file)

    # Merge CLI overrides
    for key, value in vars(cli_args).items():
        if key == "config_file":
            continue
        default_value = parser.get_default(key)
        if value == default_value or value is None:
            continue
        if hasattr(cfg, key):
            setattr(cfg, key, value)
        else:
            cfg.__dict__[key] = value

    cfg.training = not getattr(cli_args, "eval_only", False)

    # === Auto allocate output dir under log/ on each training run ===
    if cfg.training:
        cfg.output_dir = allocate_run_dir(base_dir="logs")

    logger = setup_environment(cfg)
    save_train_configs(cfg.output_dir, argparse.Namespace(**dict(cfg)))

    if cli_args.eval_only:
        cfg.training = False
        loaders = build_dataloader(cfg)
        _, test_gallery_loader, query_loaders, num_classes = build_query_loader_dict(loaders)
        model = build_model(cfg, num_classes=num_classes)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        checkpointer = Checkpointer(
            model,
            save_dir=cfg.output_dir,
            save_to_disk=False,
            logger=logger,
        )
        eval_ckpt = cli_args.resume_ckpt_file or op.join(cfg.output_dir, "best.pth")
        if not eval_ckpt or not os.path.isfile(eval_ckpt):
            logger.error("Checkpoint %s not found for evaluation.", eval_ckpt)
            return
        checkpointer.load(eval_ckpt)

        evaluator = Evaluator(
            gallery_loader=test_gallery_loader,
            get_mAP=True,
            **query_loaders,
        )
        run_evaluation(model, evaluator, 0, logger, device)
        return

    do_train(cfg, logger)


if __name__ == "__main__":
    main()
