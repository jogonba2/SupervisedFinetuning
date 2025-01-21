import random

import spacy
from transformers import PreTrainedTokenizerBase


def clean_text(text: str) -> str:
    """
    Cleans a text to ensure it does not contain tokens
    that are likely to be included as format tokens in
    the prompt's chat templates.

    Args:
        text (str): a text

    Returns:
        str: a cleaned text
    """
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = text.replace("\t", " ")
    text = text.replace(":", " ")
    text = " ".join(text.split())
    return text


def truncate(
    text: str, tokenizer: PreTrainedTokenizerBase, max_tokens: int
) -> str:
    """
    Truncates a text to have less than `max_tokens`
    tokens according to a given tokenizer. That is,
    if a text is tokenized with the same tokenizer
    after being passed to this function, it will
    have less than `max_tokens` tokens.

    Args:
        text (str): a text
        tokenizer (PreTrainedTokenizer): a tokenizer
        max_tokens (int): max number of tokens

    Returns:
        str: the text truncated to have less than `max_tokens` tokens
    """
    ids = tokenizer(text, truncation=True, max_length=max_tokens)["input_ids"]
    return tokenizer.decode(ids, skip_special_tokens=True)


def spacy_tokenize(
    texts: list[str],
    language: str,
    n_process: int = 4,
    batch_size: int = 2000,
) -> list[list[str]]:
    """
    Tokenizes a list of texts using spaCy, with optional multi-processing and token joining.

    Args:
        texts (list[str]): A list of text strings to be tokenized.
        language (str): The language code (e.g., "en" for English) used to initialize a blank spaCy model.
        n_process (int, optional): The number of processors to use for parallel tokenization. Defaults to 4.
        batch_size (int, optional): The number of texts to process in each batch. Defaults to 2000.
        join_tokens (bool, optional): If True, returns a flat list of tokens for all texts combined.
                                      If False, returns a list of token lists, one per text. Defaults to False.

    Returns:
        list[list[str]]: a list where each element is a list of token strings
                                     corresponding to the input texts.
    """
    nlp = spacy.blank(language)
    docs = nlp.pipe(texts, n_process=n_process, batch_size=batch_size)
    return [[token.text for token in doc] for doc in docs]


def get_vocab(
    texts: list[str],
    language: str,
    n_process: int = 4,
    batch_size: int = 2000,
) -> list[str]:
    """
    Generates a unique vocabulary list from a list of texts using spaCy tokenization.

    Parameters:
        texts (list[str]): A list of text strings to extract vocabulary from.
        language (str): The language code (e.g., "en" for English) to initialize a blank spaCy model.
        n_process (int, optional): The number of processors to use for parallel tokenization. Defaults to 4.
        batch_size (int, optional): The number of texts to process in each batch. Defaults to 2000.

    Returns:
        list[str]: unique token strings representing the vocabulary from the input texts.
    """
    tokens = spacy_tokenize(texts, language, n_process, batch_size)
    return list(set(sum(tokens, [])))


def get_random_string(
    vocab: list[str],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
):
    """
    Generates a random string from a shuffled vocabulary list,
    tokenized to a specified maximum length.

    Parameters:
        vocab (list[str]): A list of vocabulary tokens to be shuffled and concatenated into a string.
        tokenizer (PreTrainedTokenizerBase): A tokenizer (e.g., from Hugging Face Transformers) used to tokenize
                                             and decode the concatenated string.
        max_length (int): The maximum length of tokens for truncation when tokenizing the concatenated vocabulary.

    Returns:
        str: A decoded string representation of the tokenized vocabulary with special tokens removed,
             truncated to the specified max length.
    """
    shuffled_vocab = random.sample(vocab, len(vocab))
    concat_vocab = " ".join(shuffled_vocab)
    ids = tokenizer([concat_vocab], truncation=True, max_length=max_length)[
        "input_ids"
    ][0]
    return tokenizer.decode(ids, skip_special_tokens=True)
