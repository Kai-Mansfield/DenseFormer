# main.py

import os
import sys
import numpy as np
import torch
import inspect
import json
import copy
import argparse
import random
import wandb
import torch._dynamo
torch._dynamo.config.suppress_errors = True
import logging
logging.getLogger("torch").setLevel(logging.ERROR)
import math

import config
import models
from data.utils import get_dataset, prepare_dataset
from optim.base import train_base
import distributed

rank = int(os.environ.get('RANK', 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get('WORLD_SIZE', 1))

print("MASTER_ADDR:", os.environ.get("MASTER_ADDR"))
print("MASTER_PORT:", os.environ.get("MASTER_PORT"))
print("RANK:", rank)
print("LOCAL_RANK:", local_rank)
print("WORLD_SIZE:", world_size)

def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--config_format', default='base', choices=config.registered_formats())
    parser.add_argument('--prepare_dataset_only', action='store_true', help='Only run prepare_dataset() then exit')
    args, rem_args = parser.parse_known_args()
    return config.parse_args_with_format(format=args.config_format, base_parser=parser, args=rem_args, namespace=args)

# Adjust for missing/present "module." prefix
def adjust_state_dict(state_dict, model):
    model_keys = list(model.state_dict().keys())
    ckpt_keys = list(state_dict.keys())

    if all(k.startswith("module.") for k in model_keys) and not any(k.startswith("module.") for k in ckpt_keys):
        # Add "module." prefix
        state_dict = {"module." + k: v for k, v in state_dict.items()}
    elif not any(k.startswith("module.") for k in model_keys) and all(k.startswith("module.") for k in ckpt_keys):
        # Remove "module." prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    return state_dict

def main(args): 
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    distributed_backend = distributed.make_backend_from_args(args)
    print(f"Using backend: {type(distributed_backend)}")
    args = distributed_backend.get_adjusted_args_for_process(args)

    args.device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(args.device)
    device_type = 'cuda' if 'cuda' in str(args.device) else 'cpu'
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading dataset '{args.dataset}'")

    if distributed_backend.is_master_process():
        print("RANK:", rank, 'preparing dataset')
        prepare_dataset(args)
        print("RANK:", rank, 'done preparing dataset')
    else:
        print("RANK:", rank, 'skipping dataset preparation')
        
    try:
        print(f"RANK {rank} calling sync()")
        distributed_backend.sync()
        print(f"RANK {rank} done syncing")
    except Exception as e:
        print(f"RANK {rank} error during sync: {e}")
        raise

    data = get_dataset(args)
    if args.data_in_ram:
        data = {'train': np.array(data['train']), 'val': np.array(data['val'])}

    print(f"Num training tokens: {len(data['train'])}")
    print(f"Num validation tokens: {len(data['val'])}")

    model = models.make_model_from_args(args)
    model = distributed_backend.transform_model(model)

    group_specs = distributed_backend.get_raw_model(model).get_parameter_group_specs(dense_lr=.1 * args.lr, dense_weight_decay=args.weight_decay)
    param_name_mapping = {p_name: p for p_name, p in model.named_parameters()}
    optimized_params_cnt = 0
    param_dict = dict(model.named_parameters())

    for i, g in enumerate(group_specs):
        print(f"\n=== Param Group {i} ===")

        # Print all hyperparameters except params list
        for k, v in g.items():
            if k != "params":
                print(f"  {k}: {v}")

        print("  parameters:")

        for pname in g["params"]:
            p = param_dict[pname]        # <-- convert name -> tensor
            print(f"    - {pname} (shape={tuple(p.shape)})")
    for g in group_specs:
        params = []
        for p_name in g["params"]:
            translated_p_names = distributed_backend.translate_model_parameter_name_for_node(p_name)
            if p_name in param_name_mapping:
                params += [param_name_mapping[p_name] for p_name in translated_p_names]
            else:
                print(f"Skipping tied or missing param: {p_name}")
        g["params"] = params
        optimized_params_cnt += sum([p.numel() for p in g["params"]])
    print("number of optimized parameters: %.2fM" % (optimized_params_cnt / 1e6))

    # -----------------------
    #  OPTIMIZER CREATION
    # -----------------------
    use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
    print(f"Using fused AdamW: {use_fused}")

    if args.opt == 'adamw':
        extra_args = dict(fused=True) if use_fused else {}
        opt = torch.optim.AdamW(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            **extra_args
        )
    elif args.opt == 'sgd':
        opt = torch.optim.SGD(
            group_specs,
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.opt}")

    # Ensure schedulers have valid base LR reference
    for g in opt.param_groups:
        g.setdefault('initial_lr', args.lr)

    # -----------------------
    #  SCHEDULER CREATION
    # -----------------------
    def make_scheduler(opt):
        if args.scheduler == 'none':
            return None

        if args.scheduler == 'cos':
            def lr_lambda(current_step: int):
                warmup_steps = int(args.iterations * args.warmup_percent)
                max_lr = args.lr
                min_lr = max_lr * 0.1
                start_iter = getattr(args, 'start_iter', 0)

                if current_step < warmup_steps:
                    # Linear warmup
                    scale = float(current_step) / float(max(1, warmup_steps))
                    lr = min_lr + (max_lr - min_lr) * scale
                elif current_step < start_iter:
                    lr = max_lr
                else:
                    # Cosine decay
                    progress = float(current_step - start_iter) / float(max(1, args.iterations - start_iter))
                    progress = min(max(progress, 0.0), 1.0)
                    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                    lr = min_lr + (max_lr - min_lr) * cosine

                return lr / max_lr

            return torch.optim.lr_scheduler.LambdaLR(
                opt,
                lr_lambda=lr_lambda,
                last_epoch=getattr(args, 'start_iter', 0)
            )

        elif args.scheduler == 'constant':
            # lr_lambda should always return 1.0 → LR stays at args.lr
            def lr_lambda(current_step: int):
                return 1.0

            return torch.optim.lr_scheduler.LambdaLR(
                opt,
                lr_lambda=lr_lambda,
                last_epoch=getattr(args, 'start_iter', -1)
            )

        elif args.scheduler == 'linear':
            def lr_lambda(current_step: int):
                warmup_steps = int(args.iterations * args.warmup_percent)
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                return max(0.0, float(args.iterations - current_step) /
                            float(max(1, args.iterations - warmup_steps)))

            return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

        elif args.scheduler == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                mode='min',
                factor=0.9,
                patience=1,
                threshold=1e-4,
                min_lr=args.lr * 0.1,
                verbose=True
            )

        raise NotImplementedError(f"Unknown scheduler: {args.scheduler}")

    scheduler = make_scheduler(opt)

    # === Load checkpoint if specified ===
    resume_iter = 0
    if args.use_pretrained and args.use_pretrained != "none":
        print(f"Loading checkpoint from {args.use_pretrained}")
        checkpoint = torch.load(args.use_pretrained, map_location=args.device)

        # Load model weights
        state_dict = checkpoint.get('model', checkpoint)
        state_dict = adjust_state_dict(state_dict, model)
        model.load_state_dict(state_dict, strict=True)

        # Restore optimizer + scheduler
        # if 'optimizer' in checkpoint:
        #     print("Restoring optimizer state...")
        #     opt.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint and scheduler is not None:
            print("Restoring scheduler state...")
            scheduler.load_state_dict(checkpoint['scheduler'])

        # === OVERRIDE LR ===
        # if args.lr is not None:
        #     print(f"Overriding checkpoint LR with {args.lr:.2e}")
        #     for g in opt.param_groups:
        #         g['lr'] = args.lr
        #         g['initial_lr'] = args.lr

        #     # Force scheduler to recompute based on new LR
        #     if isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
        #         scheduler.base_lrs = [args.lr for _ in scheduler.base_lrs]
        #         scheduler.step(scheduler.last_epoch)  # resync internal state

        # Restore RNG states for deterministic continuation
        if 'rng_state' in checkpoint:
            state = checkpoint['rng_state']
            if state.is_cuda:
                state = state.cpu()
            torch.set_rng_state(state if isinstance(state, torch.ByteTensor) else torch.ByteTensor(state))

        if 'cuda_rng_state' in checkpoint:
            # ensure all are CPU ByteTensors
            cuda_state = []
            for s in checkpoint['cuda_rng_state']:
                s_cpu = s.cpu() if s.is_cuda else s
                if not isinstance(s_cpu, torch.ByteTensor):
                    s_cpu = torch.ByteTensor(s_cpu)
                cuda_state.append(s_cpu)
            torch.cuda.set_rng_state_all(cuda_state)

        if 'numpy_rng_state' in checkpoint:
            np.random.set_state(checkpoint['numpy_rng_state'])

        if 'python_rng_state' in checkpoint:
            random.setstate(checkpoint['python_rng_state'])

        resume_iter = checkpoint.get('itr', 0)
        print(f"Resuming training from iteration {resume_iter}")

    for i, group in enumerate(opt.param_groups):
        print(f"Param group {i}:")
        for k, v in group.items():
            if k != "params":
                print(f"  {k}: {v}")
        for j, p in enumerate(group['params']):
            if p is not None:
                # Print parameter name if available
                name = next((n for n, param in model.named_parameters() if param is p), None)
                print(f"    param {j}: {name}, shape: {p.shape}, requires_grad: {p.requires_grad}")

    args.world_size = distributed_backend.get_world_size()
    exp_name = args.exp_name
    if distributed_backend.is_master_process() and args.wandb:
        params_copy = copy.deepcopy(vars(args))
        del params_copy['device']
        wandb.init(project=args.wandb_project, name=exp_name, config=params_copy)

    ckpt_path = f"{args.results_base_folder}/{args.dataset}/{args.model}"
    if not os.path.exists(ckpt_path):
        if distributed_backend.is_master_process():
            os.makedirs(ckpt_path)

    if 'base' in args.model or 'mc' in args.model or True:
        train = train_base
    else:
        raise NotImplementedError(f"No training method implemented for model type '{args.model}'.")

    print(f"\nTraining model={args.model} \n{vars(args)}\n")

    stats = train(model, opt, data, scheduler, args.iterations, args.acc_steps, args.batch_size, args.sequence_length, 
                  eval_freq=args.eval_freq,
                  distributed_backend=distributed_backend,
                  ckpt_path=ckpt_path,
                  srt_iter=resume_iter,
                  extra_args=args)

    args.device = None
    args.dtype = None
    stats['args'] = vars(args)
    if distributed_backend.is_master_process():
        with open(f"{ckpt_path}/summary.json", "w") as fs:
            json.dump(stats, fs)
    distributed_backend.finalize()


if __name__ == "__main__":
    args = get_args()
    main(args)
