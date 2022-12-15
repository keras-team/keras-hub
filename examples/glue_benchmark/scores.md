# GLUE Benchmark Score on KerasNLP Pretrained Models

We use `glue.py` to test out KerasNLP pretrained models, and report scores in
this doc. Our goal is to quickly verify our model's performance instead of 
searching for the best hyperparameters, so the reported score can be a little 
worse than reported by the original paper. 

Unless specifically noted, hyperparameter settings are the same across all GLUE 
tasks. 

## BERT

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

