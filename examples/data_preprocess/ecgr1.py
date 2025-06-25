# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the ecg-r1 dataset to parquet format
"""

import argparse
import os
import io

import datasets
from PIL import Image

from verl.utils.hdfs_io import copy, makedirs

def resize_image(image, target_width=1000):
    """
    Resize image to target width while maintaining aspect ratio
    
    Args:
        image: PIL Image object
        target_width: Target width for resizing (default: 1000)
    
    Returns:
        Resized PIL Image object
    """
    target_height = int(target_width * image.size[1] / image.size[0])
    resized_img = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    return resized_img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/scratch/users/czakka/data/ecgr1")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--target_width", type=int, default=500, help="Target width for image resizing")
    parser.add_argument("--resize_images", action="store_true", help="Enable image resizing")

    args = parser.parse_args()

    data_source = "cyrilzakka/ecg_r1_data"

    dataset = datasets.load_dataset(data_source)

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    instruction_following = (
        r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
        r"The reasoning process MUST BE enclosed within <think> </think> tags. Output the final answer after `####`."
    )

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            problem = "<image>Given the following ECG, what is the R-R interval? "
            prompt = problem + " " + instruction_following
            answer = example.pop("json")["rr_interval"]
            images = example.pop("jpg")
            if args.resize_images:
                images = resize_image(images, args.target_width)

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "images": images,
                "ability": "medical",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer,
                    "question": problem,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=1)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True, num_proc=1)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train_500.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test_500.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)


