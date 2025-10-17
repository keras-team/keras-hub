import math
from functools import partial

import keras.src.saving
from jax.random import normal
from jax.random import truncated_normal
from jax.random import uniform
from keras.src.backend import backend
from keras.src.backend import distribution_lib
from keras.src.backend import random
from keras.src.backend.config import floatx
from keras.src.initializers.random_initializers import RandomNormal
from keras.src.initializers.random_initializers import RandomUniform
from keras.src.initializers.random_initializers import TruncatedNormal
from keras.src.initializers.random_initializers import VarianceScaling

# Constant for token embedding path in layout map for all text based models.
# This may need to be updated if models with different layout maps are used.
# The failsafe is to just use non-distributed initializers if no layout is found

TOKEN_EMBEDDING_PATH = "token_embedding/embeddings"


def _get_token_embedding_layout():
    """Get tensor layout for token embeddings if distribution is enabled."""
    from keras.src.distribution import distribution as get_distribution

    current_distribution = get_distribution()

    if current_distribution is None:
        return None
    if not hasattr(current_distribution, "_layout_map"):
        return None

    layout_map = current_distribution._layout_map
    tensor_layout = layout_map.get(TOKEN_EMBEDDING_PATH, None)

    if tensor_layout is None or tensor_layout.device_mesh is None:
        return None

    if backend() != "jax":
        return None

    return tensor_layout


def _get_number_of_shards_per_axis(tensor_layout):
    """Get number of shards on each axis from tensor layout.

    NOTE: This function extracts sharding info but does NOT affect scaling.

    Design principle: "Aggregation pattern dictates variance scaling"

    For standard model parallelism (vocab sharding):
    - Use GLOBAL fan_in/fan_out (don't divide by shards)
    - Because all vocab entries are logically accessible via communication
    - Example: Standard transformer with sharded embedding table

    For true vocabulary partitioning (rare):
    - Use LOCAL fan_in/fan_out (divide by shards)
    - Because each device operates on disjoint subsets with no aggregation
    - Example: Multi-lingual model where device 0 only handles English tokens,
      device 1 only handles French tokens, etc.

    Currently, we assume standard model parallelism (global fan_in/fan_out).
    """

    if tensor_layout is None or tensor_layout.device_mesh is None:
        return None

    mesh = tensor_layout.device_mesh

    # Currently only support 2D sharding for text models
    if len(mesh.shape) != 2 or len(tensor_layout.axes) != 2:
        return None

    mesh_dict = dict(zip(mesh.axis_names, mesh.shape))

    num_shards_per_axis = tuple(
        mesh_dict.get(axis, 1) for axis in tensor_layout.axes
    )

    return num_shards_per_axis


def jax_draw_seed(seed):
    import jax
    from keras.src.random.seed_generator import draw_seed

    if isinstance(seed, jax.Array):
        return seed
    else:
        return draw_seed(seed)


# Registered for deserialization, but not in public API
@keras.src.saving.register_keras_serializable(package="keras_hub")
class DistributedRandomNormal(RandomNormal):
    """Distributed Random normal initializer.
    Given we have the apt layout, this initializer will use the layout to
    shard the initialized weights. If no layout is found, it will fall back to
    standard RandomNormal initializer.
    Draws samples from a normal distribution for given parameters.

    Args:
        mean: A python scalar or a scalar keras tensor. Mean of the random
            values to generate.
        stddev: A python scalar or a scalar keras tensor. Standard deviation of
           the random values to generate.
        seed: A Python integer or instance of
            `keras.backend.SeedGenerator`.
            Used to make the behavior of the initializer
            deterministic. Note that an initializer seeded with an integer
            or `None` (unseeded) will produce the same random values
            across multiple calls. To get different random values
            across multiple calls, use as seed an instance
            of `keras.backend.SeedGenerator`.
    """

    def __call__(self, shape, dtype=None):
        tensor_layout = _get_token_embedding_layout()
        if tensor_layout is None:
            return random.normal(
                shape=shape,
                mean=self.mean,
                stddev=self.stddev,
                seed=self.seed,
                dtype=dtype,
            )
        else:
            dtype = dtype or floatx()
            seed = jax_draw_seed(self.seed)
            init_func = partial(
                normal,
                shape=shape,
                dtype=dtype,
            )
            return distribution_lib._distribute_initializer(
                init_func=init_func,
                mean=self.mean,
                stddev=self.stddev,
                seed=seed,
                layout=tensor_layout,
            )


# Registered for deserialization, but not in public API
@keras.src.saving.register_keras_serializable(package="keras_hub")
class DistributedTruncatedNormal(TruncatedNormal):
    """Distributed Initializer that generates a truncated normal distribution.
    Given we have the apt layout, this initializer will use the layout to
    shard the initialized weights. If no layout is found, it will fall back to
    standard TruncatedNormal initializer.
    The values generated are similar to values from a
    `RandomNormal` initializer, except that values more
    than two standard deviations from the mean are
    discarded and re-drawn.

    Args:
        mean: A python scalar or a scalar keras tensor. Mean of the random
            values to generate.
        stddev: A python scalar or a scalar keras tensor. Standard deviation of
           the random values to generate.
        seed: A Python integer or instance of
            `keras.backend.SeedGenerator`.
            Used to make the behavior of the initializer
            deterministic. Note that an initializer seeded with an integer
            or `None` (unseeded) will produce the same random values
            across multiple calls. To get different random values
            across multiple calls, use as seed an instance
            of `keras.backend.SeedGenerator`.
    """

    def __call__(self, shape, dtype=None):
        tensor_layout = _get_token_embedding_layout()
        if tensor_layout is None:
            return random.truncated_normal(
                shape=shape,
                mean=self.mean,
                stddev=self.stddev,
                seed=self.seed,
                dtype=dtype,
            )
        else:
            dtype = dtype or floatx()
            seed = jax_draw_seed(self.seed)
            init_func = partial(
                truncated_normal,
                shape=shape,
                dtype=dtype,
                lower=-2.0,
                upper=2.0,
            )
            return distribution_lib._distribute_initializer(
                init_func=init_func,
                mean=self.mean,
                stddev=self.stddev,
                seed=seed,
                layout=tensor_layout,
            )


# Registered for deserialization, but not in public API
@keras.src.saving.register_keras_serializable(package="keras_hub")
class DistributedRandomUniform(RandomUniform):
    """Distributed Random uniform initializer.
    Given we have the apt layout, this initializer will use the layout to
    shard the initialized weights. If no layout is found, it will fall back to
    standard RandomUniform initializer.
    Draws samples from a uniform distribution for given parameters.

    Args:
        minval: A python scalar or a scalar keras tensor. Lower bound of the
            range of random values to generate (inclusive).
        maxval: A python scalar or a scalar keras tensor. Upper bound of the
            range of random values to generate (exclusive).
        seed: A Python integer or instance of
            `keras.backend.SeedGenerator`.
            Used to make the behavior of the initializer
            deterministic. Note that an initializer seeded with an integer
            or `None` (unseeded) will produce the same random values
            across multiple calls. To get different random values
            across multiple calls, use as seed an instance
            of `keras.backend.SeedGenerator`.
    """

    def __call__(self, shape, dtype=None):
        tensor_layout = _get_token_embedding_layout()
        if tensor_layout is None:
            return random.uniform(
                shape=shape,
                minval=self.minval,
                maxval=self.maxval,
                seed=self.seed,
                dtype=dtype,
            )
        else:
            dtype = dtype or floatx()
            seed = jax_draw_seed(self.seed)
            init_func = partial(
                uniform,
                shape=shape,
                dtype=dtype,
                minval=self.minval,
                maxval=self.maxval,
            )

            return distribution_lib._distribute_initializer(
                init_func=init_func,
                mean=None,
                stddev=None,
                seed=seed,
                layout=tensor_layout,
            )


# Registered for deserialization, but not in public API
@keras.src.saving.register_keras_serializable(package="keras_hub")
class DistributedVarianceScaling(VarianceScaling):
    """Distributed Initializer that adapts its scaling.
    Given we have the apt layout, this initializer will use the layout to
    shard the initialized weights. If no layout is found, it will fall back to
    standard VarianceScaling initializer.
    With `distribution="truncated_normal" or "untruncated_normal"`, samples are
    drawn from a truncated/untruncated normal distribution with a mean of zero
    and a standard deviation (after truncation, if used) `stddev = sqrt(scale /
    n)`, where `n` is:

    - number of input units in the weight tensor, if `mode="fan_in"`
    - number of output units, if `mode="fan_out"`
    - average of the numbers of input and output units, if `mode="fan_avg"`

    With `distribution="uniform"`, samples are drawn from a uniform distribution
    within `[-limit, limit]`, where `limit = sqrt(3 * scale / n)`.


    Args:
        scale: Scaling factor (positive float).
        mode: One of `"fan_in"`, `"fan_out"`, `"fan_avg"`.
        distribution: Random distribution to use.
            One of `"truncated_normal"`, `"untruncated_normal"`, or `"uniform"`.
        seed: A Python integer or instance of
            `keras.backend.SeedGenerator`.
            Used to make the behavior of the initializer
            deterministic. Note that an initializer seeded with an integer
            or `None` (unseeded) will produce the same random values
            across multiple calls. To get different random values
            across multiple calls, use as seed an instance
            of `keras.backend.SeedGenerator`.
    """

    def __call__(self, shape, dtype=None):
        scale = self.scale
        fan_in, fan_out = compute_fans(shape)
        if self.mode == "fan_in":
            scale /= max(1.0, fan_in)
        elif self.mode == "fan_out":
            scale /= max(1.0, fan_out)
        else:
            scale /= max(1.0, (fan_in + fan_out) / 2.0)
        tensor_layout = _get_token_embedding_layout()
        if tensor_layout is None:
            if self.distribution == "truncated_normal":
                stddev = math.sqrt(scale) / 0.87962566103423978
                return random.truncated_normal(
                    shape, mean=0.0, stddev=stddev, dtype=dtype, seed=self.seed
                )
            elif self.distribution == "untruncated_normal":
                stddev = math.sqrt(scale)
                return random.normal(
                    shape, mean=0.0, stddev=stddev, dtype=dtype, seed=self.seed
                )
            else:
                limit = math.sqrt(3.0 * scale)
                return random.uniform(
                    shape,
                    minval=-limit,
                    maxval=limit,
                    dtype=dtype,
                    seed=self.seed,
                )
        else:
            dtype = dtype or floatx()
            seed = jax_draw_seed(self.seed)
            if self.distribution == "truncated_normal":
                stddev = math.sqrt(scale) / 0.87962566103423978
                init_func = partial(
                    truncated_normal,
                    shape=shape,
                    dtype=dtype,
                    lower=-2.0,
                    upper=2.0,
                )
                return distribution_lib._distribute_initializer(
                    init_func=init_func,
                    mean=0.0,
                    stddev=stddev,
                    seed=seed,
                    layout=tensor_layout,
                )
            elif self.distribution == "untruncated_normal":
                stddev = math.sqrt(scale)
                init_func = partial(
                    normal,
                    shape=shape,
                    dtype=dtype,
                )
                return distribution_lib._distribute_initializer(
                    init_func=init_func,
                    mean=0.0,
                    stddev=stddev,
                    seed=seed,
                    layout=tensor_layout,
                )
            else:
                limit = math.sqrt(3.0 * scale)
                init_func = partial(
                    uniform,
                    shape=shape,
                    dtype=dtype,
                    minval=-limit,
                    maxval=limit,
                )
                return distribution_lib._distribute_initializer(
                    init_func=init_func,
                    mean=None,
                    stddev=None,
                    seed=seed,
                    layout=tensor_layout,
                )


# Registered for deserialization, but not in public API
@keras.src.saving.register_keras_serializable(package="keras_hub")
class DistributedGlorotUniform(DistributedVarianceScaling):
    """Distributed The Glorot uniform initializer, also called Xavier uniform
    initializer. Given apt layout, this initializer will use the layout to
    shard the initialized weights. If no layout is found, it will fall back to
    standard GlorotUniform initializer.

    Draws samples from a uniform distribution within `[-limit, limit]`, where
    `limit = sqrt(6 / (fan_in + fan_out))` (`fan_in` is the number of input
    units in the weight tensor and `fan_out` is the number of output units).

    Args:
        seed: A Python integer or instance of
            `keras.backend.SeedGenerator`.
            Used to make the behavior of the initializer
            deterministic. Note that an initializer seeded with an integer
            or `None` (unseeded) will produce the same random values
            across multiple calls. To get different random values
            across multiple calls, use as seed an instance
            of `keras.backend.SeedGenerator`.

    Reference:

    - [Glorot et al., 2010](http://proceedings.mlr.press/v9/glorot10a.html)
    """

    def __init__(self, seed=None):
        super().__init__(
            scale=1.0, mode="fan_avg", distribution="uniform", seed=seed
        )

    def get_config(self):
        return {
            "seed": keras.src.saving.serialization_lib.serialize_keras_object(
                self._init_seed
            )
        }


# Registered for deserialization, but not in public API
@keras.src.saving.register_keras_serializable(package="keras_hub")
class DistributedGlorotNormal(DistributedVarianceScaling):
    """Distributed Glorot normal initializer, also called
    Xavier normal initializer. Given apt layout, this initializer
    will use the layout to shard the initialized weights.If no layout is found,
    it will fall back to standard GlorotNormal initializer.

    Draws samples from a truncated normal distribution centered on 0 with
    `stddev = sqrt(2 / (fan_in + fan_out))` where `fan_in` is the number of
    input units in the weight tensor and `fan_out` is the number of output units
    in the weight tensor.

    Args:
        seed: A Python integer or instance of
            `keras.backend.SeedGenerator`.
            Used to make the behavior of the initializer
            deterministic. Note that an initializer seeded with an integer
            or `None` (unseeded) will produce the same random values
            across multiple calls. To get different random values
            across multiple calls, use as seed an instance
            of `keras.backend.SeedGenerator`.

    Reference:

    - [Glorot et al., 2010](http://proceedings.mlr.press/v9/glorot10a.html)
    """

    def __init__(self, seed=None):
        super().__init__(
            scale=1.0,
            mode="fan_avg",
            distribution="truncated_normal",
            seed=seed,
        )

    def get_config(self):
        return {
            "seed": keras.src.saving.serialization_lib.serialize_keras_object(
                self._init_seed
            )
        }


# Registered for deserialization, but not in public API
@keras.src.saving.register_keras_serializable(package="keras_hub")
class DistributedLecunNormal(DistributedVarianceScaling):
    """Distributed Lecun normal initializer.
    Given apt layout, this initializer will use the layout to shard the
    initialized weights.If no layout is found, it will fall back to standard
    LecunNormal initializer.Initializers allow you to pre-specify an
    initialization strategy, encoded in the Initializer object, without
    knowing the shape and dtype of the variable being initialized.

    Draws samples from a truncated normal distribution centered on 0 with
    `stddev = sqrt(1 / fan_in)` where `fan_in` is the number of input units in
    the weight tensor.

    Args:
        seed: A Python integer or instance of
            `keras.backend.SeedGenerator`.
            Used to make the behavior of the initializer
            deterministic. Note that an initializer seeded with an integer
            or `None` (unseeded) will produce the same random values
            across multiple calls. To get different random values
            across multiple calls, use as seed an instance
            of `keras.backend.SeedGenerator`.

    Reference:

    - [Klambauer et al., 2017](https://arxiv.org/abs/1706.02515)
    """

    def __init__(self, seed=None):
        super().__init__(
            scale=1.0, mode="fan_in", distribution="truncated_normal", seed=seed
        )

    def get_config(self):
        return {
            "seed": keras.src.saving.serialization_lib.serialize_keras_object(
                self._init_seed
            )
        }


# Registered for deserialization, but not in public API
@keras.src.saving.register_keras_serializable(package="keras_hub")
class DistributedLecunUniform(DistributedVarianceScaling):
    """Distributed Lecun uniform initializer.
    Given apt layout, this initializer will use the layout to shard the
    initialized weights.If no layout is found, it will fall back to standard
    LecunUniform initializer. Initializers allow you to pre-specify an
    initialization strategy, encoded in the Initializer object, without knowing
    the shape and dtype of the variable being initialized.
    Draws samples from a uniform distribution within `[-limit, limit]`, where
    `limit = sqrt(3 / fan_in)` (`fan_in` is the number of input units in the
    weight tensor).

    Args:
        seed: A Python integer or instance of
            `keras.backend.SeedGenerator`.
            Used to make the behavior of the initializer
            deterministic. Note that an initializer seeded with an integer
            or `None` (unseeded) will produce the same random values
            across multiple calls. To get different random values
            across multiple calls, use as seed an instance
            of `keras.backend.SeedGenerator`.

    Reference:

    - [Klambauer et al., 2017](https://arxiv.org/abs/1706.02515)
    """

    def __init__(self, seed=None):
        super().__init__(
            scale=1.0, mode="fan_in", distribution="uniform", seed=seed
        )

    def get_config(self):
        return {
            "seed": keras.src.saving.serialization_lib.serialize_keras_object(
                self._init_seed
            )
        }


# Registered for deserialization, but not in public API
@keras.src.saving.register_keras_serializable(package="keras_hub")
class DistributedHeNormal(DistributedVarianceScaling):
    """Distributed He normal initializer.
    Given apt layout, this initializer will use the layout to shard the
    initialized weights. If no layout is found, it will fall back to standard
    HeNormal initializer. Initializers allow you to pre-specify an
    initialization strategy, encoded in the Initializer object, without knowing
    the shape and dtype of the variable being initialized.
    It draws samples from a truncated normal distribution centered on 0 with
    `stddev = sqrt(2 / fan_in)` where `fan_in` is the number of input units in
    the weight tensor.

    Args:
        seed: A Python integer or instance of
            `keras.backend.SeedGenerator`.
            Used to make the behavior of the initializer
            deterministic. Note that an initializer seeded with an integer
            or `None` (unseeded) will produce the same random values
            across multiple calls. To get different random values
            across multiple calls, use as seed an instance
            of `keras.backend.SeedGenerator`.

    Reference:

    - [He et al., 2015](https://arxiv.org/abs/1502.01852)
    """

    def __init__(self, seed=None):
        super().__init__(
            scale=2.0, mode="fan_in", distribution="truncated_normal", seed=seed
        )

    def get_config(self):
        return {
            "seed": keras.src.saving.serialization_lib.serialize_keras_object(
                self._init_seed
            )
        }


# Registered for deserialization, but not in public API
@keras.src.saving.register_keras_serializable(package="keras_hub")
class DistributedHeUniform(DistributedVarianceScaling):
    """Distributed He uniform variance scaling initializer.
    Given apt layout, this initializer will use the layout to shard the
    initialized weights. If no layout is found, it will fall back to standard
    HeUniform initializer. Initializers allow you to pre-specify an
    initialization strategy, encoded in the Initializer object, without
    knowing the shape and dtype of the variable being initialized.
    Draws samples from a uniform distribution within `[-limit, limit]`, where
    `limit = sqrt(6 / fan_in)` (`fan_in` is the number of input units in the
    weight tensor).

    Args:
        seed: A Python integer or instance of
            `keras.backend.SeedGenerator`.
            Used to make the behavior of the initializer
            deterministic. Note that an initializer seeded with an integer
            or `None` (unseeded) will produce the same random values
            across multiple calls. To get different random values
            across multiple calls, use as seed an instance
            of `keras.backend.SeedGenerator`.

    Reference:

    - [He et al., 2015](https://arxiv.org/abs/1502.01852)
    """

    def __init__(self, seed=None):
        super().__init__(
            scale=2.0, mode="fan_in", distribution="uniform", seed=seed
        )

    def get_config(self):
        return {
            "seed": keras.src.saving.serialization_lib.serialize_keras_object(
                self._init_seed
            )
        }


def compute_fans(shape):
    """Computes the number of input and output units for a weight shape.

    Args:
        shape: Integer shape tuple.

    Returns:
        A tuple of integer scalars: `(fan_in, fan_out)`.
    """
    shape = tuple(shape)
    if len(shape) < 1:  # Just to avoid errors for constants.
        fan_in = fan_out = 1
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    else:
        # Assuming convolution kernels (2D, 3D, or more).
        # kernel shape: (..., input_depth, depth)
        receptive_field_size = 1
        for dim in shape[:-2]:
            receptive_field_size *= dim
        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size
    return int(fan_in), int(fan_out)
