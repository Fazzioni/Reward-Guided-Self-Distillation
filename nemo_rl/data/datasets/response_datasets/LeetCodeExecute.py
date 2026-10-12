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
        split: str = "train",
        split_validation_size: float = 0.05,
        seed: int = 42,
        use_hint_in_prompt: bool = True,
        prefix_tokens_to_insert_hint: list[int] | None = None,
        
        **kwargs,
    ):
        self.input_key = "question_content"
        self.output_key = "outputs"
        self.input_input_key = 'inputs'
        self.hint_key = "right_code"
        
        self.task_name = "math"
        self.use_hint_in_prompt = use_hint_in_prompt        
        self.prefix_tokens_to_insert_hint = prefix_tokens_to_insert_hint

        # load from huggingface
        dataset = load_dataset("RewardGuided/liveruncodebench")
        
        
        dataset = dataset.filter(lambda x: len(x.get(self.hint_key,'')) < 5000)
        dataset = dataset.filter(lambda x: len(x.get(self.input_input_key,'')) < 5000)
        dataset = dataset.filter(lambda x: len(x.get(self.output_key,'')) < 5000)
        
        
        # format the dataset
        dataset = dataset.map(self.format_data_)
        

        self.val_dataset = dataset['test']
        self.dataset = dataset['train']
        
        

    def format_data_(self, data: dict[str, Any]) -> dict[str, Any]:
        
        user_content = f"{data[self.input_key]}\n\nNow, you need to provide the Output of these inputs:\n{data[self.input_input_key]}"
        
        result =  {
            "messages": [
                {"role": "user", "content": user_content},
                #{"role": "assistant", "content": data[self.output_key]},
            ],
            "task_name": self.task_name,
            'ground_truth': data[self.output_key]
        }
        
        if self.use_hint_in_prompt:
            #result['start_hint'] = self.hint_start_string
            #result['end_hint'] = self.hint_end_string
            result['prompt_hint'] =  '\n' + data[self.hint_key]
            result['hint_tokens'] = self.prefix_tokens_to_insert_hint

        return result

