# Copyright 2023 Matteo Pagliardini, Amirkeivan Mohtashami, Francois Fleuret, Martin Jaggi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import nullcontext

import torch
import torch.nn.functional as F
import wandb
import time 
import copy
import traceback
import sys
import math

from .utils import eval, get_batch, save_checkpoint

import torch
import torch.nn as nn

def grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def train_base(model, opt, data, scheduler, iterations, acc_steps, batch_size, sequence_length, eval_freq, ckpt_path, distributed_backend, extra_args, srt_iter=0):
    device_type = 'cuda' if 'cuda' in str(extra_args.device) else 'cpu'
    type_ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(
        device_type=device_type, dtype=extra_args.dtype)  # extra_args.dtype)
    itr, substep, best_val_loss, text_table = srt_iter, 0, float('inf'), None # best_val_loss not used atm, early stopping not recommended but possible 

    stats = {'train_loss': [], 'val_loss': [], 'val_pp': [], 'val_acc': []}

    num_substeps_per_epoch = len(data['train']) // (batch_size * sequence_length)

    running_layer_grad_sum = {}
    running_layer_grad_count = {}
    
    if not extra_args.no_compile:
        print(f"Compiling model ...")
        import torch._dynamo as torchdynamo
        torchdynamo.config.guard_nn_modules = True
        model = torch.compile(model) # requires pytorch 2.0+

    model.train()

    t0 = time.time()

    while itr < iterations:
                
        for microstep_idx in range(acc_steps):  # gradient accumulation
            x, y = get_batch(data['train'], sequence_length, batch_size, device=extra_args.device)
            if torch.any(y.cpu() >= 50304) or torch.any(y.cpu() < -1):
                print("Found out-of-range targets!")
            if torch.any(y.cpu() >= model.config.vocab_size):
                print(f"Warning: targets contain indices >= vocab_size ({model.config.vocab_size})")
            with type_ctx:
                with distributed_backend.get_context_for_microstep_forward(model=model, microstep_idx=microstep_idx, gradient_accumulation_steps=acc_steps):
                    if getattr(distributed_backend.get_raw_model(model), "needs_iter", False):
                        outputs = model(x, targets=y, iter=itr)
                    else:
                        if torch.isnan(x).any():
                            print("Input x contains NaNs!")

                        if torch.isnan(y).any():
                            print("Target y contains NaNs!")
                        outputs = model(x, targets=y)

            loss = outputs['loss']
            loss.backward()
            substep += 1

        # # ---- TOP LAYERS ----
        layer_grads = []
        for name, p in model.named_parameters():
            if p.grad is not None:
                layer_grads.append((name, p.grad.data.norm().item()))
        layer_grads = sorted(layer_grads, key=lambda x: x[1], reverse=True)

        named_params = dict(model.named_parameters())
        elemwise_info = {}  # store per-layer (idx_string, max_val)

        for name, grad in layer_grads:
            p = named_params[name]

            if p.grad is None:
                elemwise_info[name] = ("NO_GRAD", 0.0)
                continue

            abs_grad = p.grad.detach().abs()
            max_val = abs_grad.max().item()

            # index of max
            argmax_flat = abs_grad.argmax()
            multi_idx = torch.unravel_index(argmax_flat, abs_grad.shape)

            idx_string = "".join(f"[{int(i)}]" for i in multi_idx)

            elemwise_info[name] = (idx_string, max_val)

        if extra_args.grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), extra_args.grad_clip)

        opt.step()

        if hasattr(scheduler, 'total_steps'):
            max_steps = scheduler.total_steps
        elif hasattr(scheduler, 'T_max'):
            max_steps = scheduler.T_max
        elif hasattr(scheduler, 'milestones'):
            max_steps = scheduler.milestones[-1] if len(scheduler.milestones) > 0 else 0
        else:
            max_steps = iterations  # fallback

        if itr < max_steps:
            scheduler.step()
        opt.zero_grad(set_to_none=True)
        itr += 1

        if itr % eval_freq == 0 or itr == iterations: # from here it's only evaluation code, all the training is above
            if distributed_backend.is_master_process():
                t1 = time.time()
                dt = t1 - t0
                epoch = substep//num_substeps_per_epoch

                model.eval()
                train_loss = loss.detach().cpu().item()
                group_lrs = [g["lr"] for g in opt.param_groups]
                val_acc, val_loss, val_perplexity = eval(model, data['val'], sequence_length, batch_size,
                                                         extra_args.device, max_num_batches=24, ctx=type_ctx)
                # test_acc, test_loss, test_perplexity = eval(model, data['test'], sequence_length, batch_size,
                #                                          extra_args.device, max_num_batches=24, ctx=type_ctx)

                print_string = (
                    f"{epoch}/{itr} [train] loss={train_loss:.3f}"
                    f" [val] loss={val_loss:.3f}"
                    # f" [test] loss={test_loss:.3f}"
                    f" [time per itr] {dt*1000/eval_freq:.2f}ms"
                    # f" [lr0] {group_lrs[0]:.7f} [lr1] {group_lrs[1]:.7f} [lr2] {group_lrs[2]:.7f}"
                    f" [lr0] {group_lrs[0]:.7f}"
                )

                print(print_string)

                # ---- Write all grads + stored max elementwise grads to file ----
                output_path = f"{ckpt_path}/{extra_args.ckpt_name}_layer_grads.txt"
                with open(output_path, "a") as f:
                    f.write(f"\n--- Iteration {itr} ---\n")

                    for name, grad in layer_grads:  # same order
                        idx_string, max_val = elemwise_info[name]

                        f.write(
                            f"{name}: layer_grad={grad:.6e}  max_elem={idx_string}  elem_grad={max_val:.6e}\n"
                        )

                if extra_args.wandb:
                    wandb.log({
                        "iter": itr,
                        "train/loss": train_loss,
                        "val/loss": val_loss,
                        "val/perplexity": val_perplexity,
                        "val/acc": val_acc,
                        "lr": current_lr,
                    })

                model.train()
                t0 = time.time()
        
        if distributed_backend.is_master_process():
            if extra_args.save_checkpoint_freq is not None and itr % extra_args.save_checkpoint_freq == 0:
                save_checkpoint(distributed_backend=distributed_backend,
                                model=model,
                                opt=opt,
                                scheduler=scheduler,
                                itr=itr,
                                ckpt_path=f"{ckpt_path}/{extra_args.ckpt_name}")
                print(f"saved checkpoint to {ckpt_path}/{extra_args.ckpt_name}")

    return stats
