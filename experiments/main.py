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

    print('args.device', args.device)
    if args.device != 'cpu':
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

    model = models.make_model_from_args(args).to(args.device)
    model = distributed_backend.transform_model(model)

    group_specs = distributed_backend.get_raw_model(model).get_parameter_group_specs()
    param_name_mapping = {p_name: p for p_name, p in model.named_parameters()}
    optimized_params_cnt = 0
    for g in group_specs:
        params = []
        for p_name in g["params"]:
            translated_p_names = distributed_backend.translate_model_parameter_name_for_node(p_name)
            params += [param_name_mapping[p_name] for p_name in translated_p_names]
        g["params"] = params
        optimized_params_cnt += sum([p.numel() for p in g["params"]])
    print("number of optimized parameters: %.2fM" % (optimized_params_cnt / 1e6))

    if args.opt == 'adamw':
        use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        opt = torch.optim.AdamW(group_specs, lr=args.lr, betas=(args.beta1, args.beta2),
                                weight_decay=args.weight_decay, **extra_args)
    else:
        opt = torch.optim.SGD(group_specs, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    if args.scheduler != 'none':
        if args.scheduler in ['cos', 'linear']:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=opt, max_lr=args.lr, total_steps=args.iterations, 
                pct_start=args.warmup_percent, anneal_strategy=args.scheduler, 
                cycle_momentum=False, div_factor=1e2, final_div_factor=.1)
        else:
            raise NotImplementedError(f"Unknown scheduler type: {args.scheduler}.")
    else:
        scheduler = None

    # === Load checkpoint if specified ===
    resume_iter = 0
    if args.use_pretrained and args.use_pretrained != "none":
        print(f"Loading checkpoint from {args.use_pretrained}")
        checkpoint = torch.load(args.use_pretrained, map_location=args.device)
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        state_dict = adjust_state_dict(state_dict, model)
        model.load_state_dict(state_dict, strict=True)

        resume_iter = checkpoint.get('itr', 0)
        print(f"Resuming training from iteration {resume_iter}")

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
    distributed_backend.finalize()


if __name__ == "__main__":
    args = get_args()
    main(args)
