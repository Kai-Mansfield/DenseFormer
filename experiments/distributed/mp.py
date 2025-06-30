import os
import math
from contextlib import contextmanager

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed import init_process_group, destroy_process_group, get_world_size, barrier

from .backend import DistributedBackend


def transformer_auto_wrap_policy(module, recurse, nonwrapped_numel):
    from torch.nn import TransformerEncoderLayer, TransformerDecoderLayer
    return isinstance(module, (TransformerEncoderLayer, TransformerDecoderLayer))


class mp(DistributedBackend):
    def __init__(self, args):
        self.rank = int(os.environ.get('RANK', -1))
        assert self.rank != -1, "FSDP backend requires RANK"
        assert "cuda" in args.device, "FSDP backend requires CUDA"
        init_process_group(backend='nccl')
        self.local_rank = int(os.environ['LOCAL_RANK'])

    def get_adjusted_args_for_process(self, args):
        effective_batch_size = args.batch_size * args.acc_steps
        world_size = self.get_world_size()
        if effective_batch_size % world_size != 0:
            raise ValueError(f"Effective batch size "
                             f"{effective_batch_size} is not divisible "
                             f"by the world size {world_size}.")
        acc_steps_div = math.gcd(args.acc_steps, world_size)
        args.acc_steps = args.acc_steps // acc_steps_div
        args.batch_size = args.batch_size // (world_size // acc_steps_div)
        args.device = f'cuda:{self.local_rank}'
        args.seed = args.seed + self.local_rank
        return args

    def transform_model(self, model):
        mp_policy = MixedPrecision(param_dtype=torch.bfloat16)  # Remove if not supported on your GPUs

        return FSDP(
            model,
            auto_wrap_policy=transformer_auto_wrap_policy,
            mixed_precision=mp_policy,
            device_id=torch.device(f"cuda:{self.local_rank}")
        )

    @contextmanager
    def get_context_for_microstep_forward(self, model, microstep_idx, gradient_accumulation_steps):
        model.require_backward_grad_sync = (
            microstep_idx == gradient_accumulation_steps - 1)
        yield

    def is_master_process(self) -> bool:
        return self.rank == 0

    def get_raw_model(self, model):
        return model.module if hasattr(model, "module") else model

    def translate_model_parameter_name_for_node(self, parameter_name):
        return [f'module.{parameter_name}']

    def get_world_size(self):
        return get_world_size()
    
    def sync(self):
        import torch.distributed as dist
        print(f"Rank {dist.get_rank()} of {dist.get_world_size()} reached sync")
        barrier()
        print(f"[sync] RANK {dist.get_rank()} passed barrier")

    def finalize(self):
        destroy_process_group()
