# GLUE Benchmark Score on KerasNLP Pretrained Models

We use `glue.py` to test out KerasNLP pretrained models, and report scores in
this doc. Our goal is to quickly verify our model's performance instead of 
searching for the best hyperparameters, so the reported score can be a little 
worse than reported by the original paper. 

Unless specifically noted, hyperparameter settings are the same across all GLUE 
tasks. 

## BERT

Test target is `keras_nlp.models.BertClassifier()`. WNLI is skipped because it 
was not evaluated at the original paper.

### Hyperparameter Settings

- Learning Rate: 
    We use a `PolynomialDecay` learning rate, with `initial_learning_rate=5e-5`.
    ```python
    lr = tf.keras.optimizers.schedules.PolynomialDecay(
        5e-5,
        decay_steps={total_training_steps},
        end_learning_rate=0.0,
    )
    ```
- Optimizer:
    We use `AdamW` optimizer, and exclude `bias` and variables in 
    `LayerNormalization` from weight decay.

    ```python
    optimizer = tf.keras.optimizers.experimental.AdamW(
        lr, weight_decay=0.01, global_clipnorm=1.0
    )
    optimizer.exclude_from_weight_decay(
        var_names=["LayerNorm", "layer_norm", "bias"]
    )
    ```
- Others:
    | Hyperparameter Name | Value |
    |---------------------|-------|
    | batch_size          | 32    |
    | epochs              | 3     |
    | dropout             | 0.1   |

### Benchmark Score

| Task Name | Metrics               | Score     |
|-----------|-----------------------|-----------|
| CoLA      | Matthew's Corr        | 52.2      |
| SST-2     | Accuracy              | 93.5      |
| MRPC      | F1 / Accuracy         | 88.2/83.9 |
| STSB      | Pearson-Spearman Corr | 84.5/83.1 |
| QQP       | F1 / Accuracy         | 71.3/89.3 |
| MNLI_M    | Accuracy              |      84.3 |
| MNLI_Mis  | Accuracy              |      83.3 |
| QNLI      | Accuracy              |      90.4 |
| RTE       | Accuracy              | 66.7      |
| AX        | Matthew's Corr        |      34.8 |

See the actual submission in this [link](https://gluebenchmark.com/submission/gnG9xUQGkjfVq6loRQYKTcM1YjG3/-NIe3Owl8pjHLXpistkI). 

## RoBERTa

Test target is `keras_nlp.models.RobertaClassifier()`.

### Hyperparameter Settings

#### WNLI

We choose a special setting for WNLI from other tasks.

- Learning Rate: 
    We use a `PolynomialDecay` learning rate, with `initial_learning_rate=2e-5`.
    ```python
    lr = tf.keras.optimizers.schedules.PolynomialDecay(
        2e-5,
        decay_steps={total_training_steps},
        end_learning_rate=0.0,
    )
    ```
- Optimizer:
    We use `Adam` optimizer.

    ```python
    optimizer = tf.keras.optimizers.Adam(lr)
    ```
- Others:
    | Hyperparameter Name | Value |
    |---------------------|-------|
    | batch_size          | 32    |
    | epochs              | 10    |
    | dropout             | 0.1   |

#### Other GLUE Tasks

- Learning Rate: 
    We use a `PolynomialDecay` learning rate, with `initial_learning_rate=2e-5`.
    ```python
    lr = tf.keras.optimizers.schedules.PolynomialDecay(
        2e-5,
        decay_steps={total_training_steps},
        end_learning_rate=0.0,
    )
    ```
- Optimizer:
    We use `AdamW` optimizer, and exclude `bias` and variables in 
    `LayerNormalization` from weight decay.

    ```python
    optimizer = tf.keras.optimizers.experimental.AdamW(
        lr, weight_decay=0.01, global_clipnorm=1.0
    )
    optimizer.exclude_from_weight_decay(
        var_names=["LayerNorm", "layer_norm", "bias"]
    )
    ```
- Others:
    | Hyperparameter Name | Value |
    |---------------------|-------|
    | batch_size          | 32    |
    | epochs              | 3     |
    | dropout             | 0.1   |

### Benchmark Score

| Task Name | Metrics               | Score     |
|-----------|-----------------------|-----------|
| CoLA      | Matthew's Corr        | 56.3      |
| SST-2     | Accuracy              | 96.1     |
| MRPC      | F1 / Accuracy         | 89.8/86.3 |
| STSB      | Pearson-Spearman Corr | 88.4/87.7 |
| QQP       | F1 / Accuracy         | 72.3/89.0 |
| MNLI_M    | Accuracy              |      87.7 |
| MNLI_Mis  | Accuracy              |      87.1 |
| QNLI      | Accuracy              |      92.8 |
| RTE       | Accuracy              | 69.2     |
| WNLI      | Accuracy              | 65.1    |
| AX        | Matthew's Corr        |      40.6 |

See the actual submission in this [link](https://gluebenchmark.com/submission/gnG9xUQGkjfVq6loRQYKTcM1YjG3/-NJS0XAX1o9p8DJst3wM). 