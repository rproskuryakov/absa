import xml
import xml.etree.ElementTree
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Union

import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score
from transformers import BertTokenizer


def pad_sequence(sequence, max_len=75, value: Union[str, int] = "SEP"):
    if len(sequence) >= max_len:
        return sequence[:max_len]
    return sequence + (max_len - len(sequence)) * [value]


def tokenize_and_preserve_labels(sentence, text_labels, tokenizer):
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):

        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    tokenized_sentence = ["[CLS]"] + tokenized_sentence + ["[SEP]"]
    labels = ["O"] + labels + ["O"]
    return tokenized_sentence, labels


def tokenize_sentence_pair(
    first_sentence: str,
    second_sentence: str,
    tokenizer: BertTokenizer,
) -> (List[int], List[int]):
    first_sentence_ids = tokenizer.convert_tokens_to_ids(
        tokenizer.tokenize(first_sentence)
    )
    second_sentence_ids = tokenizer.convert_tokens_to_ids(
        tokenizer.tokenize(second_sentence)
    )
    encoded_pair = tokenizer.build_inputs_with_special_tokens(
        first_sentence_ids,
        second_sentence_ids,
    )
    token_type_ids = tokenizer.create_token_type_ids_from_sequences(
        first_sentence_ids,
        second_sentence_ids,
    )
    return encoded_pair, token_type_ids


def get_auxiliary_sentence(aspect: str, target: str, mode) -> str:
    auxiliary_sentence = ""
    if mode == "QA-M":
        auxiliary_sentence = "Что вы думаете о " + aspect + " " + target + "?"
    elif mode == "NLI-M":
        auxiliary_sentence = target + " - " + aspect
    return auxiliary_sentence


def load_tabsa_dataset(path_to_file: Path,
                       spacy_model,
                       sequences: List[str],
                       aspects: List[str],
                       sentiments: List[str],
                       categories: List[str],
                       aspect_set: Set[str],
                       none_label: Optional[str] = None):
    tree = xml.etree.ElementTree.parse(path_to_file)
    root = tree.getroot()
    for review in root:
        text = review.find("text").text
        review_aspects = [aspect.attrib for aspect in review.find("aspects")]
        review_text = spacy_model(text)

        # for each review construct dictionary sentence - spans with labels
        target_spans = []
        for aspect in filter(lambda x: x["type"] == "explicit", review_aspects):
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
            sequences.append(sentence.text)
            aspects.append(span.text)
            span_category, span_sentiment = span.label_.split()
            categories.append(span_category)
            sentiments.append(span_sentiment)

            # add none with other category
            if none_label:
                other_categories = aspect_set.difference(set(span_category))
                for category in other_categories:
                    sequences.append(sentence.text)
                    aspects.append(span.text)
                    categories.append(category)
                    sentiments.append(none_label)


def construct_label_to_id(label_to_id, sentiments):
    if label_to_id is None:
        unique_labels = set()
        for labels in sentiments:
            unique_labels = unique_labels | set(labels)
        label_to_id = {token: id_ for id_, token in enumerate(unique_labels)}
    return label_to_id
