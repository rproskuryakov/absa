import xml.etree.ElementTree
from collections import defaultdict
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Union

import numpy as np
import spacy
import torch
import torch.nn
import torch.utils.data
from transformers import BertTokenizer

from src.util import pad_sequence
from src.util import tokenize_sentence_pair
from src.util import tokenize_and_preserve_labels
from src.util import get_auxiliary_sentence
from src.util import load_tabsa_dataset
from src.util import construct_label_to_id


class AspectExtractionDataset(torch.utils.data.Dataset):
    def __init__(
        self, path_to_file, tokenizer: BertTokenizer, max_len=75, label_to_id=None,
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        spacy_model = spacy.load("ru_core_news_sm")
        tree = xml.etree.ElementTree.parse(path_to_file)
        root = tree.getroot()

        self.sequences = []
        self.labels = []

        for review in root:
            text = review.find("text").text
            aspects = [aspect.attrib for aspect in review.find("aspects")]
            review_text = spacy_model(text)

            # for each review construct dictionary sentence - spans with labels
            sentence_to_spans = defaultdict(list)
            for aspect in filter(lambda x: x["type"] == "explicit", aspects):
                char_span = review_text.char_span(
                    int(aspect["from"]),
                    int(aspect["to"]),
                    label=aspect["category"],
                )
                if char_span is not None:
                    sentence_to_spans[char_span.sent].append(char_span)

            for sentence in review_text.sents:
                # construct labels sequence for each sentence
                sentence_spans = sentence_to_spans.get(sentence, [])
                sentence_tokens = [token.text for token in sentence]
                sentence_labels = ["O"] * len(sentence_tokens)
                for span in sentence_spans:
                    start = span.start - sentence.start
                    end = span.end - sentence.start
                    sentence_labels[start:end] = ["B"] + ["I"] * (end - start - 1)
                # skip sentence in case of error in sentence tokenizer
                if len(sentence_tokens) != len(sentence_labels):
                    continue
                self.sequences.append(sentence_tokens)
                self.labels.append(sentence_labels)

        # construct dictionary label - number
        self.label_to_id = label_to_id
        if label_to_id is None:
            unique_labels = set()
            for labels in self.labels:
                unique_labels = unique_labels | set(labels)
            self.label_to_id = {
                token: id_ for id_, token in enumerate(unique_labels, 1)
            }
            self.label_to_id["PAD"] = 0
        self.id_to_label = {id_: token for token, id_ in self.label_to_id.items()}

    def __getitem__(self, item):
        tokens, labels = self.sequences[item], self.labels[item]
        tokens, labels = tokenize_and_preserve_labels(tokens, labels, self.tokenizer)
        input_ids = pad_sequence(
            self.tokenizer.convert_tokens_to_ids(tokens),
            max_len=self.max_len,
            value=self.tokenizer.pad_token_id,
        )
        tags = pad_sequence(
            [self.label_to_id[label] for label in labels],
            max_len=self.max_len,
            value=self.label_to_id["PAD"],
        )
        attention_mask = [float(i != 0.0) for i in input_ids]
        return {
            "input_ids": torch.LongTensor(input_ids),
            "attention_masks": torch.FloatTensor(attention_mask),
            "labels": torch.LongTensor(tags),
        }

    def __len__(self):
        return len(self.sequences)


class TABSADataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path_to_file: Union[Path, str],
        tokenizer: BertTokenizer,
        max_len: int = 75,
        label_to_id: Optional[Dict[str, int]] = None,
        aspects: Optional[Set[str]] = None,
        mode: str = "QA-M",
        none_label: str = "none",
    ):
        if mode not in {"QA-M", "NLI-M"}:
            raise ValueError("`mode` argument should be either `QA-M` or `NLI-M`")
        self.mode = mode
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.aspect_set = aspects
        self.sequences: List[str] = []
        self.targets: List[str] = []
        self.aspects: List[str] = []
        self.sentiments: List[str] = []
        self.label_to_id: Dict[str, int] = label_to_id
        self.id_to_label = None
        self.spacy_model = spacy.load("ru_core_news_sm")
        load_tabsa_dataset(
            path_to_file,
            self.spacy_model,
            self.sequences,
            self.targets,
            self.sentiments,
            self.aspects,
            self.aspect_set,
            none_label
        )
        self.label_to_id = construct_label_to_id(self.label_to_id, self.sentiments)
        self.id_to_label = {id_: token for token, id_ in self.label_to_id.items()}

    def __getitem__(self, item):
        auxiliary_sentence = get_auxiliary_sentence(
            self.aspects[item], self.targets[item], self.mode
        )

        encoded_pair, token_type_ids = tokenize_sentence_pair(
            auxiliary_sentence, self.sequences[item], tokenizer=self.tokenizer
        )
        padded = pad_sequence(
            encoded_pair, max_len=self.max_len, value=self.tokenizer.pad_token_id
        )
        attention_mask = [float(i != self.tokenizer.pad_token_id) for i in padded]
        padded_token_type_ids = pad_sequence(
            token_type_ids,
            max_len=self.max_len,
            value=1,
        )
        return {
            "input_ids": torch.LongTensor(padded),
            "attention_masks": torch.FloatTensor(attention_mask),
            "token_type_ids": torch.LongTensor(padded_token_type_ids),
            "labels": self.label_to_id[self.sentiments[item]],
        }

    def __len__(self):
        return len(self.sequences)

    @property
    def class_weights(self):
        n_samples = len(self.sentiments)
        sample_label_ids = [self.label_to_id[label] for label in self.sentiments]
        return n_samples / (len(self.label_to_id) * np.bincount(sample_label_ids))


class AspectCategorizationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path_to_file,
        tokenizer: BertTokenizer,
        max_len=75,
        label_to_id=None,
        mode="QA-M",
    ):
        if mode not in {"QA-M", "NLI-M"}:
            raise ValueError("`mode` argument should be either `QA-M` or `NLI-M`")
        self.mode = mode
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.sequences: List[str] = []
        self.aspects: List[str] = []
        self.categories: List[str] = []
        self.label_to_id: Dict[str, int] = label_to_id
        self.id_to_label = None
        self.spacy_model = spacy.load("ru_core_news_sm")
        self.load_dataset(path_to_file)

    def __getitem__(self, item):
        encoded_seq = self.tokenizer.encode(["[CLS]", *self.tokenizer.tokenize(self.aspects[item]), "[SEP]"])
        padded = pad_sequence(
            encoded_seq, max_len=self.max_len, value=self.tokenizer.pad_token_id
        )
        attention_mask = [float(i != self.tokenizer.pad_token_id) for i in padded]
        return {
            "input_ids": torch.LongTensor(padded),
            "attention_masks": torch.FloatTensor(attention_mask),
            "labels": self.label_to_id[self.categories[item]],
        }

    def __len__(self):
        return len(self.sequences)

    def load_dataset(self, path_to_file):
        tree = xml.etree.ElementTree.parse(path_to_file)
        root = tree.getroot()
        for review in root:
            text = review.find("text").text
            aspects = [aspect.attrib for aspect in review.find("aspects")]
            review_text = self.spacy_model(text)

            # for each review construct dictionary sentence - spans with labels
            target_spans = []
            for aspect in filter(lambda x: x["type"] == "explicit", aspects):
                char_span = review_text.char_span(
                    int(aspect["from"]),
                    int(aspect["to"]),
                    label=(aspect["category"] + " " + aspect["sentiment"]),
                )
                if char_span is not None:
                    target_spans.append(char_span)

            for span in target_spans:
                # construct labels sequence for each sentence
                sentence = span.sent
                self.sequences.append(sentence.text)
                self.aspects.append(span.text)
                span_category, _ = span.label_.split()
                self.categories.append(span_category)
