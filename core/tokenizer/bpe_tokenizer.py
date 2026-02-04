"""
Byte Pair Encoding (BPE) Tokenizer
---------------------------------
Implements a simple, deterministic BPE tokenizer from scratch.
No external tokenizer dependency is used.

Author: RANOELISON Dimbisoa Patrick
"""

from collections import Counter, defaultdict


class BPETokenizer:
    def __init__(self, vocab_size: int = 30000):
        """
        Initialize the tokenizer.

        :param vocab_size: Target vocabulary size
        """
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = []

    def train(self, texts):
        """
        Train BPE tokenizer on a list of texts.

        :param texts: List[str]
        """
        # Initialize vocabulary with characters
        words = Counter()
        for text in texts:
            for word in text.split():
                words[" ".join(list(word)) + " </w>"] += 1

        vocab = words

        # Iteratively merge most frequent pairs
        while len(self.vocab) < self.vocab_size:
            pairs = defaultdict(int)

            for word, freq in vocab.items():
                symbols = word.split()
                for i in range(len(symbols) - 1):
                    pairs[(symbols[i], symbols[i + 1])] += freq

            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            self.merges.append(best_pair)

            new_vocab = {}
            bigram = " ".join(best_pair)
            replacement = "".join(best_pair)

            for word in vocab:
                new_word = word.replace(bigram, replacement)
                new_vocab[new_word] = vocab[word]

            vocab = new_vocab

        self.vocab = {token: idx for idx, token in enumerate(vocab.keys())}

    def encode(self, text: str):
        """
        Encode text into token ids.

        :param text: input string
        :return: List[int]
        """
        tokens = []
        for word in text.split():
            word = list(word) + ["</w>"]
            tokens.extend(word)
        return [self.vocab.get(t, 0) for t in tokens]
