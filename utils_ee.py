# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2020 Xiaozhi Wang
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


import csv
import glob
import json
import logging
import os
from typing import List

import tqdm

from transformers import PreTrainedTokenizer


logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for multiple choice"""

    def __init__(self, example_id, tokens, triggerL, triggerR, label=None):
        """Constructs a InputExample.

        Args:
            example_id: Unique id for the example.
            contexts: list of str. The untokenized text of the first sequence (context of corresponding question).
            question: string. The untokenized text of the second sequence (question).
            endings: list of str. multiple choice's options. Its length must be equal to contexts' length.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.example_id = example_id
        self.tokens = tokens
        self.triggerL = triggerL
        self.triggerR = triggerR
        self.label = label


class InputFeatures(object):
    def __init__(self, example_id, input_ids, input_mask, segment_ids, maskL, maskR, label):
        self.example_id = example_id
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.maskL = maskL
        self.maskR = maskR
        self.label = label


class DataProcessor(object):
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class ACEProcessor(DataProcessor):
    """Processor for the RACE data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(json.load(open(os.path.join(data_dir,'train.json'),"r")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(json.load(open(os.path.join(data_dir,'dev.json'),"r")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(json.load(open(os.path.join(data_dir,'test.json'),"r")), "test")

    def get_labels(self):
        """See base class."""
        return ['None', 'End-Position', 'Charge-Indict', 'Convict', 'Transfer-Ownership', 'Demonstrate', 'Transport', 'Sentence', 'Appeal', 'Start-Org', 'Start-Position', 'End-Org', 'Phone-Write', 'Nominate', 'Marry', 'Pardon', 'Release-Parole', 'Meet', 'Trial-Hearing', 'Extradite', 'Execute', 'Transfer-Money', 'Elect', 'Injure', 'Acquit', 'Divorce', 'Die', 'Arrest-Jail', 'Declare-Bankruptcy', 'Be-Born', 'Merge-Org', 'Fine', 'Sue', 'Attack']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (idx, data_raw) in enumerate(lines):
            e_id = "%s-%s" % (set_type, idx)
            examples.append(
                InputExample(
                    example_id=e_id,
                    tokens=data_raw['tokens'],
                    triggerL=data_raw['trigger_start'],
                    triggerR=data_raw['trigger_end']+1,
                    label=data_raw['event_type'],
                )
            )
        return examples

def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    max_length: int,
    tokenizer: PreTrainedTokenizer,
    pad_token_segment_id=0,
    pad_on_left=False,
    pad_token=0,
    mask_padding_with_zero=True,
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        textL = tokenizer.tokenize(" ".join(example.tokens[:example.triggerL]))
        textR = tokenizer.tokenize(" ".join(example.tokens[example.triggerL:]))
        maskL = [1.0 for i in range(0,len(textL)+1)] + [0.0 for i in range(0,len(textR)+2)]
        maskR = [0.0 for i in range(0,len(textL)+1)] + [1.0 for i in range(0,len(textR)+2)]
        if len(maskL)>max_length:
            maskL = maskL[:max_length]
        if len(maskR)>max_length:
            maskR = maskR[:max_length]
        inputs = tokenizer.encode_plus(
            textL + ['[unused0]'] + textR, add_special_tokens=True, max_length=max_length, return_token_type_ids=True
        )
        if "num_truncated_tokens" in inputs and inputs["num_truncated_tokens"] > 0:
            logger.info(
                "Attention! you are cropping tokens."
            )

        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        assert len(input_ids)==len(maskL)
        assert len(input_ids)==len(maskR)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            maskL = ([0.0] * padding_length) + maskL
            maskR = ([0.0] * padding_length) + maskR
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
            maskL = maskL + ([0.0] * padding_length)
            maskR = maskR + ([0.0] * padding_length)

        assert len(input_ids) == max_length
        assert len(attention_mask) == max_length
        assert len(token_type_ids) == max_length

        label = label_map[example.label]

        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("example_id: {}".format(example.example_id))
            logger.info("input_ids: {}".format(" ".join(map(str, input_ids))))
            logger.info("attention_mask: {}".format(" ".join(map(str, attention_mask))))
            logger.info("token_type_ids: {}".format(" ".join(map(str, token_type_ids))))
            logger.info("maskL: {}".format(" ".join(map(str, maskL))))
            logger.info("maskR: {}".format(" ".join(map(str, maskR))))
            logger.info("label: {}".format(label))

        features.append(InputFeatures(example_id=example.example_id, input_ids=input_ids, input_mask=attention_mask, segment_ids=token_type_ids, maskL=maskL, maskR=maskR, label=label))

    return features


processors = {"ace": ACEProcessor}


MULTIPLE_CHOICE_TASKS_NUM_LABELS = {"ace", 34}
