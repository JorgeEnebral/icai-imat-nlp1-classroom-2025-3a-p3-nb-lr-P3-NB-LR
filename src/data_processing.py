from typing import List, Dict
import torch
import pandas as pd

try:
    from src.utils import SentimentExample, tokenize, remove_punctuations
except ImportError:
    from utils import SentimentExample, tokenize, remove_punctuations


def read_sentiment_examples(infile: str) -> List[SentimentExample]:
    """
    Reads sentiment examples from a file.

    Args:
        infile: Path to the file to read from.

    Returns:
        A list of SentimentExample objects parsed from the file.
    """
    # Open the file, go line by line, separate sentence and label, tokenize the sentence and create SentimentExample object
    examples: List[SentimentExample] = []
    with open(infile, "r", encoding="utf-8") as f:
        cont_ok = 0
        cont_no_ok = 0
        for line in f:
            # Separate sentence and label
            try:
                sentence, label = line.split("\t")
                # Tokenize the preprocessed sentence
                words = tokenize(remove_punctuations(sentence))
                # Create SentimentExample object
                example = SentimentExample(words, int(label))
                examples.append(example)
                cont_ok += 1
            except:
                cont_no_ok += 1
    return examples

def build_vocab(examples: List[SentimentExample]) -> Dict[str, int]:
    """
    Creates a vocabulary from a list of SentimentExample objects.

    The vocabulary is a dictionary where keys are unique words from the examples and values are their corresponding indices.

    Args:
        examples (List[SentimentExample]): A list of SentimentExample objects.

    Returns:
        Dict[str, int]: A dictionary representing the vocabulary, where each word is mapped to a unique index.
    """
    # Count unique words in all the examples from the training set
    vocab: Dict[str, int] = {}
    cont = 0
    for example in examples:
        for word in example.words:
            if word not in vocab:
                vocab[word] = cont
                cont += 1
    return vocab

def bag_of_words(
    text: List[str], vocab: Dict[str, int], binary: bool = False
) -> torch.Tensor:
    """
    Converts a list of words into a bag-of-words vector based on the provided vocabulary.
    Supports both binary and full (frequency-based) bag-of-words representations.

    Args:
        text (List[str]): A list of words to be vectorized.
        vocab (Dict[str, int]): A dictionary representing the vocabulary with words as keys and indices as values.
        binary (bool): If True, use binary BoW representation; otherwise, use full BoW representation.

    Returns:
        torch.Tensor: A tensor representing the bag-of-words vector.
    """
    # Converts list of words into BoW, take into account the binary vs full
    bow: torch.Tensor = torch.zeros(len(vocab))
    for word in text:
        if word in vocab:
            if binary:
                bow[vocab[word]] = 1
            else:
                bow[vocab[word]] += 1
    return bow

def create_datasets() -> None:
    df = pd.read_parquet('data/train-00000-of-00001.parquet')
    df.to_csv('data/train.txt', index=False, sep='\t', header=False)

    df = pd.read_parquet('data/test-00000-of-00001.parquet')
    df.to_csv('data/test.txt', index=False, sep='\t', header=False)