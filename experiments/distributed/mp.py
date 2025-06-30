import os
import math
from contextlib import contextmanager

class mp:
    def __init__(self, args):
        self.rank = int(os.environ.get('RANK', -1))
        assert self.rank != -1, "MP backend requires RANK environment variable"
        assert "cuda" in args.device, "MP backend requires CUDA devices"
        self.local_rank = int(os.environ['LOCAL_RANK'])

    def get_adjusted_args_for_process(self, args):
        # Adjust batch size and acc_steps like DDP to split work evenly
        effective_batch_size = args.batch_size * args.acc_steps
        world_size = self.get_world_size()
        if effective_batch_size % world_size != 0:
            raise ValueError(f"Effective batch size {effective_batch_size} not divisible by world size {world_size}.")
        acc_steps_div = math.gcd(args.acc_steps, world_size)
        args.acc_steps = args.acc_steps // acc_steps_div
        args.batch_size = args.batch_size // (world_size // acc_steps_div)
        args.device = f'cuda:{self.local_rank}'
        args.seed = args.seed + self.local_rank
        return args

    def transform_model(self, model):
        # Do NOT wrap with DDP, just return model as-is
        return model

    @contextmanager
    def get_context_for_microstep_forward(self, model, microstep_idx, gradient_accumulation_steps):
        # No grad sync management needed
        yield

    def is_master_process(self) -> bool:
        return self.rank == 0

    def get_raw_model(self, model):
        # No DDP wrapper, so return model directly
        return model

    def translate_model_parameter_name_for_node(self, parameter_name):
        # No module prefix from DDP, return param as list
        return [parameter_name]

    def get_world_size(self):
        return int(os.environ.get('WORLD_SIZE', 1))

    def sync(self):
        # No distributed sync, so no-op or optionally print info
        pass

    def finalize(self):
        # No process group to destroy
        pass
