import io
import pathlib

import sentencepiece


def train_sentencepiece(data, filename, *args, **kwargs):
    bytes_io = io.BytesIO()
    sentencepiece.SentencePieceTrainer.train(
        sentence_iterator=iter(data), model_writer=bytes_io, *args, **kwargs
    )
    with open(
        pathlib.Path(__file__).parent.parent.parent
        / "keras_hub"
        / "src"
        / "tests"
        / "test_data"
        / filename,
        mode="wb",
    ) as f:
        f.write(bytes_io.getbuffer())
