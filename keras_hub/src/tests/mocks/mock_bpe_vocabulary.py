"""A tiny BPE vocabulary and merge list shared by the model tests."""


def get_bpe_fixture():
    merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
    merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
    merges += ["Ġai r", "Ġa i", "pla ne"]
    vocab = []
    for merge in merges:
        a, b = merge.split(" ")
        vocab.extend([a, b, a + b])
    vocab += ["!", "<|endoftext|>"]
    vocab = sorted(set(vocab))
    vocab = dict([(token, i) for i, token in enumerate(vocab)])
    return vocab, merges
