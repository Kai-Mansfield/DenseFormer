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

import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import Dataset
import glob
import json 

OWT2_DATA_PATH = '/mnt/lustre/users/inf/kajm20/df/openwebtext2'
tknzr = tiktoken.get_encoding("gpt2")


def prepare_openwebtext2_data(config):
    pass


def data_generator():
    for filename in sorted(glob.glob("/mnt/lustre/scratch/inf/kajm20/.cache/huggingface/datasets/downloads/extracted/d836b*/**/*.jsonl", recursive=True)):
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue  # skip malformed lines


def get_openwebtext2_data(config):
    num_proc = 40
    """ https://openwebtext2.readthedocs.io/en/latest/ 
    """
    if not os.path.exists(os.path.join(OWT2_DATA_PATH, 'train.bin')):
        os.makedirs(OWT2_DATA_PATH, exist_ok=True)
        dataset = Dataset.from_generator(data_generator)

        # === Create train / val / test splits ===
        # First split into (train+val) and test
        split_1 = dataset.train_test_split(test_size=0.0005, shuffle=True, seed=2357)
        train_dataset = split_1["train"]
        val_dataset = split_1["test"]

        # Now split (train+val)
        # split_2 = split_1["train"].train_test_split(test_size=0.0005, shuffle=True)
        # train_dataset = split_2["train"]
        # val_dataset = split_2["test"]

        # split_dataset = {"train": train_dataset, "val": val_dataset, "test": test_dataset}
        split_dataset = {"train": train_dataset, "val": val_dataset}

        def process(example):
            ids = tknzr.encode_ordinary(example['text'])
            ids.append(tknzr.eot_token)
            return {'ids': ids, 'len': len(ids)}

        # Tokenize all splits
        tokenized = {
            split: dset.map(
                process,
                remove_columns=['text'],
                desc=f"tokenizing {split} split",
                num_proc=num_proc,
            )
            for split, dset in split_dataset.items()
        }

        # Write each split to binary file
        for split, dset in tokenized.items():
            arr_len = np.sum(dset['len'])
            filename = os.path.join(OWT2_DATA_PATH, f'{split}.bin')
            dtype = np.uint16  # enc.max_token_value == 50256 < 2**16
            arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
            total_batches = 1024

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
                batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
                arr_batch = np.concatenate(batch['ids'])
                arr[idx: idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()

    train_data = np.memmap(os.path.join(OWT2_DATA_PATH, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(OWT2_DATA_PATH, 'val.bin'), dtype=np.uint16, mode='r')
    # test_data = np.memmap(os.path.join(OWT2_DATA_PATH, 'test.bin'), dtype=np.uint16, mode='r')

    # return {'train': train_data, 'val': val_data, 'test': test_data}
    return {'train': train_data, 'val': val_data}
