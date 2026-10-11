# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Any

from datasets import load_dataset

from nemo_rl.data.datasets.raw_dataset import RawDataset

class LeetCodeExecuteDataset(RawDataset):
    """Simple wrapper around the LeetCode Execute dataset.

    Args:
        output_key: Key for the output text, default is "expected_answer"
        split: Split name for the dataset, default is "train_1M"
        split_validation_size: Size of the validation data, default is 0.05
        seed: Seed for train/validation split when split_validation_size > 0, default is 42
    """

    def __init__(
        self,
        output_key: str = "outputs",
        split: str = "train",
        split_validation_size: float = 0.05,
        seed: int = 42,
        use_hint_in_prompt: bool = True,
        hint_key: str = "right_code",
    
        **kwargs,
    ):
        self.input_key = "question_content"
        self.output_key = output_key
        self.task_name = "math"
        
        
        self.use_hint_in_prompt = use_hint_in_prompt
        self.hint_key = hint_key


        # load from huggingface
        dataset = load_dataset("RewardGuided/liveruncodebench")
        
        # format the dataset
        dataset = dataset.map(self.format_data)

        self.val_dataset = dataset['test']
        self.dataset = dataset['train']
        
        #print(self.dataset[10])
        #raise

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        
        user_content = data[self.input_key]
        
        if self.use_hint_in_prompt:
             user_content =  f"Hint:\n\n{data[self.hint_key]}\n\nhint end.\n\n" + user_content
        
        return {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": data[self.output_key]},
            ],
            "task_name": self.task_name,
        }

