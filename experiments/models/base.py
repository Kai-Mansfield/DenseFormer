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

"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect

import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F

from . import positional_encoders, caches

def safe_move(x, device):
    def move_tensor(t):
        return t.detach().to(device).requires_grad_(t.requires_grad)

    if isinstance(x, torch.Tensor):
        return move_tensor(x)

    elif isinstance(x, torch.nn.Module):
        # Move parameters directly in this module
        for name, param in list(x._parameters.items()):
            if param is not None:
                new_param = move_tensor(param)
                x._parameters[name] = nn.Parameter(new_param, requires_grad=param.requires_grad)

        # Move buffers
        for buffer_name, buffer in x._buffers.items():
            if buffer is not None:
                x._buffers[buffer_name] = buffer.detach().to(device)

        # Recurse into children
        for child_name, child in x._modules.items():
            if child is not None:
                x._modules[child_name] = safe_move(child, device)

        return x

    elif hasattr(x, "__dict__") and hasattr(x, "encoder"):
        x.encoder = safe_move(x.encoder, device)
        return x

    else:
        raise TypeError(f"safe_move expects nn.Module or Tensor, got {type(x)}")

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config, lm_cache):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.cache_storage = lm_cache.get_storage_for_layer(self)
        self.config = config
        self.allow_cache_during_training = getattr(config, "allow_cache_during_training", False)

        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if self.flash:
            assert config.attention_window_length is None
        else: 
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            bias = torch.tril(torch.ones(config.sequence_length, config.sequence_length))
            if config.attention_window_length is not None:
               bias = torch.triu(bias, diagonal=-config.attention_window_length)
            self.register_buffer("bias", bias.view(1, 1, config.sequence_length, config.sequence_length))
        

    def forward(self, x, pos_emb_closure, cache_context, start_index):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        q = pos_emb_closure.adapt_queries(q, start_index=start_index)
        if cache_context is not None and self.cache_storage is not None:
            att_prefix, cache_values_dict = \
                self.cache_storage.retrieve_for_query(q, cache_context, pos_emb_closure, start_index)
            if self.training and att_prefix is not None and not self.allow_cache_during_training:
                raise ValueError("Cache is not allowed during training")
        else:
            att_prefix = None
        k_before_pos = k
        k = pos_emb_closure.adapt_keys(k, start_index=start_index)
        
        if self.flash:
            if att_prefix is not None:
                raise NotImplementedError
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = pos_emb_closure.adapt_attention_before_softmax(att, start_query_index=start_index, start_key_index=start_index)
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            if att_prefix is not None:
                prefix_size = att_prefix.shape[-1]
                current_size = att.shape[-1]
                att = torch.cat((att_prefix, att), dim=-1)
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            if att_prefix is not None:
                att_prefix, att = torch.split(att, (prefix_size, current_size), dim=-1)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            if att_prefix is not None:
                cache_v = cache_values_dict['v']
                if cache_v.ndim == v.ndim:
                    y += att_prefix @ cache_v
                elif cache_v.ndim == v.ndim + 1:
                    y += (att_prefix.unsqueeze(3) @ cache_v).squeeze(3)
                else:
                    raise NotImplementedError
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        if cache_context is not None and self.cache_storage is not None:
            with torch.no_grad():
                self.cache_storage.store_in_cache(k_before_pos, {'v': v})
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config, lm_cache):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config, lm_cache)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, pos_emb_closure, cache_context, start_index):
        if torch.isnan(x).any():
            print("NaNs before Block")

        x_ln1 = self.ln_1(x)
        if torch.isnan(x_ln1).any():
            print("NaNs after ln_1")

        x_attn = self.attn(x_ln1, pos_emb_closure, cache_context, start_index)
        if torch.isnan(x_attn).any():
            print("NaNs after attn")

        x = x + x_attn
        if torch.isnan(x).any():
            print("NaNs after attn residual")

        x_ln2 = self.ln_2(x)
        if torch.isnan(x_ln2).any():
            print("NaNs after ln_2")

        x_mlp = self.mlp(x_ln2)
        if torch.isnan(x_mlp).any():
            print("NaNs after mlp")

        x = x + x_mlp
        if torch.isnan(x).any():
            print("NaNs after mlp residual")

        return x

class GPTBase(nn.Module):
    needs_iter = False
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.sequence_length is not None
        self.config = config
        self.tokenizer = tiktoken.get_encoding("gpt2")

        self.lm_cache = caches.get_cache(config.lm_cache)(config)

        num_layers = config.n_layer
        mid = num_layers // 2  # Split point for model parallelism

        # === Split model components across cuda:0 and cuda:1 ===
        self.transformer = nn.ModuleDict(dict(
            # Start everything on CPU
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = positional_encoders.get_encoder(config.positional_encoder)(config),
            drop = nn.Dropout(config.dropout),

            h = nn.ModuleList([
                Block(config, self.lm_cache) for _ in range(num_layers)
            ]),

            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # self.transformer.wte = self.transformer.wte.to("cuda:1")
        # self.lm_head = self.lm_head.to("cuda:1")
        self.transformer.wte.weight = self.lm_head.weight

        # Now move to GPUs selectively
        wte_before = self.transformer["wte"].weight.data.cpu().clone()
        print('\n wte_before.requires_grad', self.transformer["wte"].weight.requires_grad)
        print('wte_before.device', self.transformer["wte"].weight.device)
        self.transformer["wte"]  = safe_move(self.transformer["wte"], "cuda:0")
        #self.transformer.wte.weight = self.lm_head.weight
        wte_after = self.transformer["wte"].weight.data.cpu().clone()
        print('wte_after.requires_grad', self.transformer["wte"].weight.requires_grad)
        print('wte_after.device', self.transformer["wte"].weight.device)
        diff = torch.abs(wte_before - wte_after).max().item()
        print("Max difference between wte pre and post transfer weights:", diff)

        # wpe_before = []
        # for name, param in self.transformer["wpe"].named_parameters():
        #     wpe_before.append(param.data.cpu().clone())
        #     print(f'wpe_before_{name}.requires_grad', param.requires_grad)
        #     print(f'wpe_before_{name}.device', param.device)
        # self.transformer["wpe"]  = safe_move(self.transformer["wpe"], "cuda:0")
        # wpe_after = []
        # for name, param in self.transformer["wpe"].named_parameters():
        #     wpe_after.append(param.data.cpu().clone())
        #     print(f'wpe_after_{name}.requires_grad', param.requires_grad)
        #     print(f'wpe_after_{name}.device', param.device)
        # for name, (before, after) in zip(self.transformer["wpe"].named_parameters(), zip(wpe_before, wpe_after)):
        #     print(f"Checking {name} weights:")
        #     print("Before transfer min/max:", before.min().item(), before.max().item())
        #     print("After transfer min/max:", after.min().item(), after.max().item())
        #     if before.shape != after.shape:
        #         print(f"Shape mismatch for {name}: {before.shape} vs {after.shape}")
        #     else:
        #         print(f"Shape match for {name}: {before.shape}")
        #     diff = torch.abs(before - after).max().item()
        #     print("Max difference between wpe pre and post transfer weights:", diff)

        # Collect all buffers before transfer (on CPU)
        wpe_before = {}
        for name, buf in self.transformer["wpe"].named_buffers():
            wpe_before[name] = buf.cpu().clone()
            print(f'\n wpe_before_{name}.requires_grad', getattr(buf, "requires_grad", False))
            print(f'wpe_before_{name}.device', buf.device)

        # Move module to GPU
        self.transformer["wpe"] = safe_move(self.transformer["wpe"], "cuda:0")

        # Collect all buffers after transfer (on CPU for comparison)
        wpe_after = {}
        for name, buf in self.transformer["wpe"].named_buffers():
            wpe_after[name] = buf.cpu().clone()
            print(f'wpe_after_{name}.requires_grad', getattr(buf, "requires_grad", False))
            print(f'wpe_after_{name}.device', buf.device)

        # Compare before vs after
        for name in wpe_before.keys():
            before = wpe_before[name]
            after = wpe_after[name]

            if before.shape != after.shape:
                print(f"\n Shape mismatch: {before.shape} vs {after.shape}")
            else:
                print(f"\n Shape match: {before.shape}")
            diff = torch.abs(before - after).max().item()
            print("Max difference pre/post transfer:", diff)

        print("wpe type:", type(self.transformer["drop"]))
        print("wpe children:", list(self.transformer["drop"].children()))
        print("wpe named parameters:", list(self.transformer["drop"].named_parameters()))
        print("wpe named buffers:", list(self.transformer["drop"].named_buffers()), '\n')

        self.transformer["drop"] = safe_move(self.transformer["drop"], "cuda:0")

        import copy
        from torch.nn import Parameter

        def snapshot_block(block):
            """
            Take a CPU snapshot of a block's parameters and buffers
            without mutating the original block.
            Preserves requires_grad.
            """
            import copy
            block_copy = copy.deepcopy(block)
            for name, param in block_copy.named_parameters():
                block_copy._parameters[name] = Parameter(
                    param.detach().clone().cpu(), requires_grad=param.requires_grad
                )
            for name, buf in block_copy.named_buffers():
                if buf is not None:
                    block_copy._buffers[name] = buf.detach().clone().cpu()
            return block_copy

        def compare_blocks(blocks_before, blocks_after):
            """
            Compare two lists of blocks (before vs after transfer).
            Checks parameters, buffers, dtype, requires_grad, training mode, device, and values.
            """

            for i, (before_block, after_block) in enumerate(zip(blocks_before, blocks_after)):
                print(f"\n=== Checking Block {i} ===")

                # ---- Module-level state ----
                if before_block.training != after_block.training:
                    print(f"  Training mode mismatch: {before_block.training} vs {after_block.training}")
                else:
                    print(f"  Training mode: {before_block.training} (OK)")

                # ---- Parameters ----
                for (name_before, p_before), (name_after, p_after) in zip(
                    before_block.named_parameters(), after_block.named_parameters()
                ):
                    assert name_before == name_after, "Parameter name mismatch!"
                    print(f"  Param: {name_before}")

                    # Device check
                    print(f"    Device before: {p_before.device}, after: {p_after.device}")

                    # Shape check
                    if p_before.shape != p_after.shape:
                        print(f"    Shape mismatch: {p_before.shape} vs {p_after.shape}")
                        continue

                    # Dtype check
                    if p_before.dtype != p_after.dtype:
                        print(f"    Dtype mismatch: {p_before.dtype} vs {p_after.dtype}")

                    # requires_grad check
                    if p_before.requires_grad != p_after.requires_grad:
                        print(f"    requires_grad mismatch: {p_before.requires_grad} vs {p_after.requires_grad}")

                    # Value check
                    diff = torch.abs(p_before.detach().cpu() - p_after.detach().cpu()).max().item()
                    if diff == 0.0:
                        print(f"    No diff (OK)")
                    else:
                        print(f"    Max abs diff: {diff:.6e}")

                # ---- Buffers ----
                for (name_before, b_before), (name_after, b_after) in zip(
                    before_block.named_buffers(), after_block.named_buffers()
                ):
                    assert name_before == name_after, "Buffer name mismatch!"
                    print(f"  Buffer: {name_before}")

                    # Device check
                    print(f"    Device before: {b_before.device}, after: {b_after.device}")

                    # Shape check
                    if b_before.shape != b_after.shape:
                        print(f"    Shape mismatch: {b_before.shape} vs {b_after.shape}")
                        continue

                    # Dtype check
                    if b_before.dtype != b_after.dtype:
                        print(f"    Dtype mismatch: {b_before.dtype} vs {b_after.dtype}")

                    # Value check
                    diff = torch.abs(b_before.detach().cpu() - b_after.detach().cpu()).max().item()
                    if diff == 0.0:
                        print(f"    No diff (OK)")
                    else:
                        print(f"    Max abs diff: {diff:.6e}")


        # === Example usage inside your code ===
        blocks_before = []
        blocks_after = []

        for i, block in enumerate(self.transformer["h"]):
            # Snapshot block before transfer (CPU clone)
            blocks_before.append(snapshot_block(block))

            # Move the *original* block
            if i < mid:
                moved_block = safe_move(block, "cuda:0")
            else:
                moved_block = safe_move(block, "cuda:1")
            blocks_after.append(moved_block)

        # Run the comparison
        compare_blocks(blocks_before, blocks_after)

        ln_f_before = self.transformer["ln_f"].weight.data.cpu().clone()
        print('\n ln_f_before.requires_grad', self.transformer["ln_f"].weight.requires_grad)
        print('ln_f_before.device', self.transformer["ln_f"].weight.device)
        self.transformer["ln_f"] = safe_move(self.transformer["ln_f"], "cuda:1")
        ln_f_after = self.transformer["ln_f"].weight.data.cpu().clone()
        print('ln_f_after.requires_grad', self.transformer["ln_f"].weight.requires_grad)
        print('ln_f_after.device', self.transformer["ln_f"].weight.device)
        diff = torch.abs(ln_f_before - ln_f_after).max().item()
        print("Max difference between ln_f pre and post transfer weights:", diff)

        lm_head_before = self.lm_head.weight.data.cpu().clone()
        print('\n lm_head_before.requires_grad', self.lm_head.weight.requires_grad)
        print('lm_head_before.device', self.lm_head.weight.device)
        self.lm_head = safe_move(self.lm_head, "cuda:0")
        lm_head_after = self.lm_head.weight.data.cpu().clone()
        print('lm_head_after.requires_grad', self.lm_head.weight.requires_grad)
        print('lm_head_after.device', self.lm_head.weight.device)
        diff = torch.abs(lm_head_before - lm_head_after).max().item()
        print("Max difference between lm_head pre and post transfer weights:", diff, '\n')

        # Initialize all weights
        self.apply(self._init_weights)

        # Scaled init for residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= sum(p.numel() for p in self.transformer.wpe.parameters()) # TODO: Why do we need this?
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, get_logits=False, use_cache=False, iter=None):
        b, t = idx.size()
        assert t <= self.config.sequence_length, f"Cannot forward sequence of length {t}, block size is only {self.config.sequence_length}"

        # Step 1: cache lookup (always on cpu/cuda:0, same as idx)
        if use_cache:
            idx, index_shift, cache_context = self.lm_cache(idx)
        else:
            index_shift = 0
            cache_context = None
        
        if getattr(self.transformer.wpe, "needs_iter", False):
            idx, pos_emb_closure = self.transformer.wpe(idx, iter=iter)
        else:
            idx, pos_emb_closure = self.transformer.wpe(idx)

        tok_emb = self.transformer.wte(idx)
        x = pos_emb_closure.adapt_model_input(tok_emb, start_index=index_shift)
        if torch.isnan(x).any():
            print(f"NaNs found after pos_emb_closure.adapt_model_input(tok_emb, start_index=index_shift)")
        x = self.transformer.drop(x)
        if torch.isnan(x).any():
            print(f"NaNs found after self.transformer.drop(x)")

        mid = self.config.n_layer // 2
        for i in range(mid):
            if cache_context is not None and torch.isnan(cache_context).any():
                print(f"NaNs found in input cache_context before block {i}")

            x = self.transformer.h[i](x, pos_emb_closure, cache_context, start_index=index_shift)

            # Check output after block (optional)
            if torch.isnan(x).any():
                print(f"NaNs found in output of block {i}")

        # Assume self.transformer.wte is initially on cuda:0.
        # Get the embedding weights before transfer
        wte_before = x.data.cpu().clone()
        requires_grad_before = x.requires_grad
        device_before = x.device

        # # (Optional) If you have already computed some gradients or want to inspect them,
        # # you can check:
        # grad_before = x.grad  # might be None if no backward yet

        # # Now, move the module to cuda:1
        x = safe_move(x, "cuda:1")

        # Get the embedding weights after transfer
        wte_after = x.data.cpu().clone()
        requires_grad_after = x.requires_grad
        device_after = x.device

        # Compare the weights. We compare using .to() to ensure both tensors are on the same device.
        diff = torch.abs(wte_before - wte_after).max().item()
        print("Max difference between pre and post transfer weights:", diff)

        # Check devices
        print("Device before:", device_before)
        print("Device after:", device_after)

        # Check requires_grad attribute
        print("Requires grad before:", requires_grad_before)
        print("Requires grad after:", requires_grad_after)

        #print("x after move", x.min(), x.max(), x.dtype)
        if torch.isnan(x).any():
                print(f"NaNs found after x.to(cuda:1)")

        # --- Collect all tensors before move ---
        before_tensors = {}
        for name, param in pos_emb_closure.named_parameters():
            before_tensors[f"param:{name}"] = {
                "tensor": param.detach().cpu().clone(),
                "requires_grad": param.requires_grad,
                "device": param.device,
            }

        for name, buf in pos_emb_closure.named_buffers():
            before_tensors[f"buffer:{name}"] = {
                "tensor": buf.detach().cpu().clone(),
                "requires_grad": getattr(buf, "requires_grad", False),
                "device": buf.device,
            }

        # --- Move module ---
        pos_emb_closure = safe_move(pos_emb_closure, "cuda:1")

        # --- Collect all tensors after move ---
        after_tensors = {}
        for name, param in pos_emb_closure.named_parameters():
            after_tensors[f"param:{name}"] = {
                "tensor": param.detach().cpu().clone(),
                "requires_grad": param.requires_grad,
                "device": param.device,
            }

        for name, buf in pos_emb_closure.named_buffers():
            after_tensors[f"buffer:{name}"] = {
                "tensor": buf.detach().cpu().clone(),
                "requires_grad": getattr(buf, "requires_grad", False),
                "device": buf.device,
            }

        # --- Compare ---
        if not before_tensors:
            print("⚠️ No parameters or buffers found in this module.")
        else:
            for key in before_tensors:
                b, a = before_tensors[key], after_tensors[key]
                diff = torch.abs(b["tensor"] - a["tensor"]).max().item()
                print(f"{key}:")
                print(f"   max diff         = {diff}")
                print(f"   device before    = {b['device']}, after = {a['device']}")
                print(f"   requires_grad b4 = {b['requires_grad']}, after = {a['requires_grad']}")

        for i in range(mid, self.config.n_layer):
            x = self.transformer.h[i](x, pos_emb_closure, cache_context, start_index=index_shift)
        if torch.isnan(x).any():
            print('nans found after second half of transformer blocks')

        x = self.transformer.ln_f(x)
        if torch.isnan(x).any():
            print(f"NaNs found after self.transformer.ln_f(x)")

        if use_cache:
            x = self.lm_cache.get_final_logits(x)
        if torch.isnan(x).any():
            print(f"NaNs found after self.lm_cache.get_final_logits(x)")
        # print("x min:", x.min().item())
        # print("x max:", x.max().item())

        if targets is not None:
            
            wte_before = x.data.cpu().clone()
            requires_grad_before = x.requires_grad
            device_before = x.device

            # # (Optional) If you have already computed some gradients or want to inspect them,
            # # you can check:
            # grad_before = x.grad  # might be None if no backward yet

            # # Now, move the module to cuda:1
            x = safe_move(x, "cuda:0")

            # Get the embedding weights after transfer
            wte_after = x.data.cpu().clone()
            requires_grad_after = x.requires_grad
            device_after = x.device

            # Compare the weights. We compare using .to() to ensure both tensors are on the same device.
            diff = torch.abs(wte_before - wte_after).max().item()
            print("Max difference between pre and post transfer weights:", diff)

            # Check devices
            print("Device before:", device_before)
            print("Device after:", device_after)

            # Check requires_grad attribute
            print("Requires grad before:", requires_grad_before)
            print("Requires grad after:", requires_grad_after)

            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        logits = logits if get_logits else None
        return {'logits': logits, 'loss': loss}

    def clear_state(self):
        self.lm_cache.clear_state()

    def crop_sequence_length(self, sequence_length):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert sequence_length <= self.config.sequence_length
        self.config.sequence_length = sequence_length
        for block in self.transformer.h:
            block.attn.bias = block.attn.bias[:,:,:sequence_length,:sequence_length]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        # TODO
        pass

    def get_parameter_group_specs(self):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
        # will appear in the no_decay and decay sets respectively after the above.
        # In addition, because named_parameters() doesn't return duplicates, it
        # will only return the first occurence, key'd by 'transformer.wte.weight', below.
        # so let's manually remove 'lm_head.weight' from decay set. This will include
        # this tensor into optimization via transformer.wte.weight only, and not decayed.
        if 'lm_head.weight' in decay:
            decay.remove('lm_head.weight')
            no_decay.add('lm_head.weight')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        return [
            {"params": sorted(list(decay))},
            {"params": sorted(list(no_decay)), "weight_decay": 0.0},
        ]

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at sequence_length
            idx_cond = idx if idx.size(1) <= self.config.sequence_length else idx[:, -self.config.sequence_length:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond, get_logits=True)['logits']
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    
    @torch.no_grad()
    def generate_from_string(self, in_str, max_new_tokens, temperature=1.0, top_k=None):
        idx = safe_move(torch.tensor(self.tokenizer.encode(in_str, allowed_special={"<|endoftext|>"})).view(1,-1), self.lm_head.weight.device)
        out_idx = self.generate(idx, max_new_tokens, temperature, top_k).view(-1).to('cpu').numpy()
        return self.tokenizer.decode(out_idx)
