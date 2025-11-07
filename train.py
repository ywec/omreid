import argparse
import os
import os.path as op
import time
from contextlib import nullcontext

import torch

from datasets import build_dataloader
from model import build_model
from solver import build_optimizer, build_lr_scheduler
from utils.checkpoint import Checkpointer
from utils.comm import is_main_process, reduce_dict
from utils.iotools import load_train_configs, save_train_configs
from utils.logger import setup_logger
from utils.meter import AverageMeter
from utils.metrics import Evaluator


def build_parser():
    parser = argparse.ArgumentParser(description="ReID5o Training Script")
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


def setup_environment(args):
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger("ORBench", save_dir=args.output_dir, if_train=args.training)
    logger.info("Using device: %s", "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(args)
    return logger


def freeze_vision_backbone(model):
    modules_to_freeze = [
        "vision_encoder",
        "rgb_tokenizer",
        "nir_tokenizer",
        "cp_tokenizer",
        "sk_tokenizer",
    ]
    for module_name in modules_to_freeze:
        module = getattr(model, module_name, None)
        if module is None:
            continue
        for param in module.parameters():
            param.requires_grad = False


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
        text_nir_cp_query_loader,
        nir_sk_text_query_loader,
        sk_nir_text_query_loader,
        text_nir_sk_query_loader,
        cp_sk_text_query_loader,
        sk_cp_text_query_loader,
        text_cp_sk_query_loader,
        nir_cp_sk_text_query_loader,
        cp_nir_sk_text_query_loader,
        sk_nir_cp_text_query_loader,
        text_nir_cp_sk_query_loader,
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
        "text_nir_cp_query_loader": text_nir_cp_query_loader,
        "nir_sk_text_query_loader": nir_sk_text_query_loader,
        "sk_nir_text_query_loader": sk_nir_text_query_loader,
        "text_nir_sk_query_loader": text_nir_sk_query_loader,
        "cp_sk_text_query_loader": cp_sk_text_query_loader,
        "sk_cp_text_query_loader": sk_cp_text_query_loader,
        "text_cp_sk_query_loader": text_cp_sk_query_loader,
        "nir_cp_sk_text_query_loader": nir_cp_sk_text_query_loader,
        "cp_nir_sk_text_query_loader": cp_nir_sk_text_query_loader,
        "sk_nir_cp_text_query_loader": sk_nir_cp_text_query_loader,
        "text_nir_cp_sk_query_loader": text_nir_cp_sk_query_loader,
    }

    return train_loader, test_gallery_loader, query_loaders, num_classes


def run_evaluation(model, evaluator, epoch, logger):
    logger.info("Start evaluating at epoch %d", epoch)
    top1 = evaluator.eval(model.eval())
    logger.info("Evaluation finished: Top-1 Average = %.3f", top1)
    model.train()
    return top1


def resolve_amp_settings(args, logger):
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    default_use_amp = device_type == "cuda"
    use_amp = bool(getattr(args, "use_amp", default_use_amp)) and device_type == "cuda"
    if bool(getattr(args, "use_amp", None)) is False and default_use_amp and not use_amp:
        logger.info("AMP disabled by configuration.")
    if not torch.cuda.is_available() and getattr(args, "use_amp", False):
        logger.warning("AMP requested but CUDA is unavailable. Disabling AMP.")
    amp_dtype_str = getattr(args, "amp_dtype", "fp16").lower()
    amp_dtype = torch.bfloat16 if amp_dtype_str == "bf16" else torch.float16
    scaler = None
    if use_amp and amp_dtype == torch.float16:
        scaler = torch.amp.GradScaler("cuda")
    autocast_factory = (
        (lambda: torch.amp.autocast("cuda", dtype=amp_dtype))
        if use_amp
        else nullcontext
    )
    return use_amp, amp_dtype_str, scaler, autocast_factory


def ensure_fp32_trainable_params(model, logger):
    fp16_params = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and param.dtype == torch.float16
    ]
    if fp16_params:
        logger.info(
            "Casting %d trainable parameters from FP16 to FP32 for stable optimization.",
            len(fp16_params),
        )
        model.float()
    return model


def do_train(args, logger):
    loaders = build_dataloader(args)
    train_loader, test_gallery_loader, query_loaders, num_classes = build_query_loader_dict(loaders)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args, num_classes=num_classes)
    model = ensure_fp32_trainable_params(model, logger)
    model.to(device)

    if getattr(args, "freeze_vision_encoder", False):
        freeze_vision_backbone(model)
        logger.info("Frozen vision encoder parameters as requested in config.")

    optimizer = build_optimizer(args, model)
    scheduler = build_lr_scheduler(args, optimizer)

    use_amp, amp_dtype_str, scaler, autocast_factory = resolve_amp_settings(args, logger)
    args.use_amp = use_amp
    args.amp_dtype = amp_dtype_str
    if use_amp:
        logger.info("AMP enabled with %s precision.", amp_dtype_str.upper())

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

    for epoch in range(start_epoch, args.num_epoch):
        epoch_start = time.time()
        loss_meter.reset()

        for iteration, batch in enumerate(train_loader, 1):
            batch = move_batch_to_device(batch, device)

            optimizer.zero_grad(set_to_none=True)
            with autocast_factory():
                loss_dict = model(batch)

                losses = [value for key, value in loss_dict.items() if "loss" in key]
                if not losses:
                    continue
                total_loss = torch.stack(losses).sum()

            if scaler is not None:
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()

            loss_meter.update(total_loss.detach().item(), batch["pids"].size(0))

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

        scheduler.step()
        epoch_time = time.time() - epoch_start
        logger.info("Epoch %d finished in %.2f seconds", epoch + 1, epoch_time)

        should_eval = args.eval_period == -1 or (
            args.eval_period > 0 and (epoch + 1) % args.eval_period == 0
        )
        if should_eval:
            top1 = run_evaluation(model, evaluator, epoch + 1, logger)
        else:
            top1 = -1

        checkpoint_data = {"epoch": epoch + 1, "best_top1": best_top1}
        checkpointer.save("last", **checkpoint_data)

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
    if not hasattr(cfg, "use_amp"):
        cfg.use_amp = torch.cuda.is_available()
    if not hasattr(cfg, "amp_dtype"):
        cfg.amp_dtype = "fp16"

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
        run_evaluation(model, evaluator, 0, logger)
        return

    do_train(cfg, logger)


if __name__ == "__main__":
    main()
