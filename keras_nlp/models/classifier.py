# Copyright 2023 The KerasNLP Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from keras_nlp.api_export import keras_nlp_export
from keras_nlp.models.task import Task


@keras_nlp_export("keras_nlp.models.Classifier")
class Classifier(Task):
    """Base class for all classification tasks.

    `Classifier` tasks wrap a `keras_nlp.models.Backbone` and
    a `keras_nlp.models.Preprocessor` to create a model that can be used for
    sequence classification. `Classifier` tasks take an additional
    `num_classes` argument, controlling the number of predicted output classes.

    To fine-tune with `fit()`, pass a dataset containing tuples of `(x, y)`
    labels where `x` is a string and `y` is a integer from `[0, num_classes)`.

    All `Classifier` tasks include a `from_preset()` constructor which can be
    used to load a pre-trained config and weights.

    Example:
    ```python
    # Load a BERT classifier with pre-trained weights.
    classifier = keras_nlp.models.Classifier.from_preset(
        "bert_base_en",
        num_classes=2,
    )
    # Fine-tune on IMDb movie reviews (or any dataset).
    imdb_train, imdb_test = tfds.load(
        "imdb_reviews",
        split=["train", "test"],
        as_supervised=True,
        batch_size=16,
    )
    classifier.fit(imdb_train, validation_data=imdb_test)
    # Predict two new examples.
    classifier.predict(["What an amazing movie!", "A total waste of my time."])
    ```
    """
