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

from .utils import eval, get_batch, save_checkpoint

def safe_move(x, device):
    if isinstance(x, torch.Tensor):
        if x.requires_grad and x.device != device:
            return x.detach().cpu().to(device).requires_grad_(x.requires_grad)
        else:
            return x.to(device)
    # If x is a module (nn.Module), just call .to(device)
    elif isinstance(x, torch.nn.Module) or hasattr(x, 'to'):
        return x.to(device)
    else:
        # fallback, e.g. for other types, just return as is or raise error
        raise TypeError(f"safe_move expects nn.Module or Tensor, got {type(x)}")

def train_base(model, opt, data, scheduler, iterations, acc_steps, batch_size, sequence_length, eval_freq, ckpt_path, distributed_backend, extra_args, srt_iter=0):
    device_type = 'cuda' if 'cuda' in str(extra_args.device) else 'cpu'
    type_ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(
        device_type=device_type, dtype=extra_args.dtype)  # extra_args.dtype)
    itr, substep, best_val_loss, text_table = srt_iter, 0, float('inf'), None # best_val_loss not used atm, early stopping not recommended but possible 

    stats = {'train_loss': [], 'val_loss': [], 'val_pp': [], 'val_acc': []}

    num_substeps_per_epoch = len(data['train']) // (batch_size * sequence_length)
    
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

            # print("Target min/max:", y.cpu().min().item(), y.cpu().max().item())
            # print("Unique target values:", torch.unique(y.cpu()))
            # print('vocab size:', model.config.vocab_size)

            if torch.any(y.cpu() >= 50304) or torch.any(y.cpu() < -1):
                print("Found out-of-range targets!")
            if torch.any(y.cpu() >= model.config.vocab_size):
                print(f"Warning: targets contain indices >= vocab_size ({model.config.vocab_size})")
            with type_ctx:
                with distributed_backend.get_context_for_microstep_forward(model=model, microstep_idx=microstep_idx, gradient_accumulation_steps=acc_steps):
                    if getattr(model, "needs_iter", False):
                        outputs = model(x, targets=y, iter=itr)
                    else:
                        # x = x.to('cuda:1')
                        # y = y.to('cuda:1')
                        # model = model.to('cuda:1')
                        # print('x device:', x.device)
                        # print('y device:', y.device)
                        # print('model params device:', next(model.parameters()).device)
                        if torch.isnan(x).any():
                            print("Input x contains NaNs!")
                        # else:
                        #     print("Input x has no NaNs.")

                        if torch.isnan(y).any():
                            print("Target y contains NaNs!")
                        # else:
                        #     print("Target y has no NaNs.")
                        outputs = model(x, targets=y)

            loss = outputs['loss']
            for i, (name, p) in enumerate(model.named_parameters()):
                print(i, name, p.device, p.requires_grad)
                if torch.isnan(p).any():
                    print(f"NaN detected in parameter be4 loss: {name}")
                if torch.isinf(p).any():
                    print(f"Inf detected in parameter be4 loss: {name}")
            # print('finished cheking for inf and nans in parameters be4 loss')
            loss.backward()
            for i, (name, p) in enumerate(model.named_parameters()):
                print(i, name, p.device, p.requires_grad)
                if torch.isnan(p).any():
                    print(f"NaN detected in parameter after loss: {name}")
                if torch.isinf(p).any():
                    print(f"Inf detected in parameter after loss: {name}")
            #print('finished cheking for inf and nans in parameters after loss')
            substep += 1

        if extra_args.grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), extra_args.grad_clip)

        opt.step()
        scheduler.step()
        opt.zero_grad(set_to_none=True)
        itr += 1

        if itr % eval_freq == 0 or itr == iterations: # from here it's only evaluation code, all the training is above
            if True:
                t1 = time.time()
                dt = t1 - t0
                epoch = substep//num_substeps_per_epoch

                model.eval()
                train_loss = loss.detach().cpu().item()
                current_lr = scheduler.get_last_lr()[0] if scheduler is not None else extra_args.lr
                val_acc, val_loss, val_perplexity = eval(model, data['val'], sequence_length, batch_size,
                                                         extra_args.device, max_num_batches=24, ctx=type_ctx)

                print_string = f"{epoch}/{itr} [train] loss={train_loss:.3f} [val] loss={val_loss:.3f}, pp={val_perplexity:.2f}, acc={val_acc:3f}"
                print_string += f" [time per itr] {dt*1000/eval_freq:.2f}ms"
                if scheduler is not None:
                    print_string += f" [lr] {current_lr:.5f}"
                print(print_string)

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
        if True:
            if extra_args.save_checkpoint_freq is not None and itr % extra_args.save_checkpoint_freq == 0:
                print(f"saving checkpoint to {ckpt_path}/ckpt.pt")
                save_checkpoint(distributed_backend=distributed_backend,
                                model=model,
                                opt=opt,
                                scheduler=scheduler,
                                itr=itr,
                                ckpt_path=f"{ckpt_path}/ckpt.pt")

    if True:
        print(f"saving checkpoint to {ckpt_path}")
        torch.save({
            'model': model.state_dict(),
            'opt': opt.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None,
            'itr': itr
        }, f"{ckpt_path}/ckpt.pt")

    return stats
