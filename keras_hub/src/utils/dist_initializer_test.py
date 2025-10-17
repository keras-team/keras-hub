import jax
import keras.saving
import pytest
from keras import backend
from keras import distribution

from keras_hub.src.utils import dist_initializer


@pytest.mark.skipif(backend.backend() != "jax", reason="jax only")
class TestDistributedInitializer:
    @pytest.fixture(autouse=True)
    def setUp(self):
        """Set up distribution context for all tests"""
        # Skip if not enough devices
        if len(jax.devices()) < 8:
            pytest.skip("requires 8+ devices")

        devices = jax.devices()
        self.device_mesh = distribution.DeviceMesh(
            (1, 8), ["batch", "model"], devices=devices
        )
        self.layout_map = distribution.LayoutMap(self.device_mesh)
        self.layout_map["token_embedding/embeddings"] = ("model", None)

        self.distribution = distribution.ModelParallel(
            device_mesh=self.device_mesh, layout_map=self.layout_map
        )
        distribution.set_distribution(self.distribution)

        yield

        distribution.set_distribution(None)

    def test_distributed_random_normal(self):
        """Test DistributedRandomNormal creates sharded arrays
        and config serialization/deserialization"""
        init = dist_initializer.DistributedRandomNormal()
        result = init(shape=(768, 256), dtype="float32")
        assert isinstance(result, jax.Array)
        assert result.shape == (768, 256)
        assert len(result.sharding.device_set) == 8

        # Get config
        config = init.get_config()

        # Serialize and deserialize
        serialized = keras.saving.serialize_keras_object(init)

        restored = keras.saving.deserialize_keras_object(serialized)

        # Check types match
        assert type(init) is type(restored), (
            f"Type mismatch: {type(init)} vs {type(restored)}"
        )

        # Check config matches
        restored_config = restored.get_config()
        assert config == restored_config, (
            f"Config mismatch: {config} vs {restored_config}"
        )

    def test_distributed_truncated_normal(self):
        """Test DistributedTruncatedNormal creates sharded arrays
        and config serialization/deserialization"""
        init = dist_initializer.DistributedTruncatedNormal()
        result = init(shape=(768, 256), dtype="float32")
        assert isinstance(result, jax.Array)
        assert result.shape == (768, 256)
        assert len(result.sharding.device_set) == 8

        # Get config
        config = init.get_config()

        # Serialize and deserialize
        serialized = keras.saving.serialize_keras_object(init)

        restored = keras.saving.deserialize_keras_object(serialized)

        # Check types match
        assert type(init) is type(restored), (
            f"Type mismatch: {type(init)} vs {type(restored)}"
        )

        # Check config matches
        restored_config = restored.get_config()
        assert config == restored_config, (
            f"Config mismatch: {config} vs {restored_config}"
        )

    def test_distributed_variance_scaling(self):
        """Test DistributedVarianceScaling creates sharded arrays
        and config serialization/deserialization"""
        init = dist_initializer.DistributedVarianceScaling(
            scale=2.0, mode="fan_in"
        )
        result = init(shape=(768, 256), dtype="float32")
        assert isinstance(result, jax.Array)
        assert result.shape == (768, 256)
        assert len(result.sharding.device_set) == 8

        # Get config
        config = init.get_config()

        # Serialize and deserialize
        serialized = keras.saving.serialize_keras_object(init)

        restored = keras.saving.deserialize_keras_object(serialized)

        # Check types match
        assert type(init) is type(restored), (
            f"Type mismatch: {type(init)} vs {type(restored)}"
        )

        # Check config matches
        restored_config = restored.get_config()
        assert config == restored_config, (
            f"Config mismatch: {config} vs {restored_config}"
        )

    def test_distributed_random_uniform(self):
        """Test DistributedRandomUniform creates sharded arrays
        and config serialization/deserialization"""
        init = dist_initializer.DistributedRandomUniform()
        result = init(shape=(768, 256), dtype="float32")
        assert isinstance(result, jax.Array)
        assert result.shape == (768, 256)
        assert len(result.sharding.device_set) == 8

        # Get config
        config = init.get_config()

        # Serialize and deserialize
        serialized = keras.saving.serialize_keras_object(init)

        restored = keras.saving.deserialize_keras_object(serialized)

        # Check types match
        assert type(init) is type(restored), (
            f"Type mismatch: {type(init)} vs {type(restored)}"
        )

        # Check config matches
        restored_config = restored.get_config()
        assert config == restored_config, (
            f"Config mismatch: {config} vs {restored_config}"
        )

    def test_distributed_glorot_uniform(self):
        """Test DistributedGlorotUniform creates sharded arrays
        and config serialization/deserialization"""
        init = dist_initializer.DistributedGlorotUniform()
        result = init(shape=(768, 256), dtype="float32")
        assert isinstance(result, jax.Array)
        assert result.shape == (768, 256)
        assert len(result.sharding.device_set) == 8

        # Get config
        config = init.get_config()

        # Serialize and deserialize
        serialized = keras.saving.serialize_keras_object(init)

        restored = keras.saving.deserialize_keras_object(serialized)

        # Check types match
        assert type(init) is type(restored), (
            f"Type mismatch: {type(init)} vs {type(restored)}"
        )

        # Check config matches
        restored_config = restored.get_config()
        assert config == restored_config, (
            f"Config mismatch: {config} vs {restored_config}"
        )

    def test_distributed_glorot_normal(self):
        """Test DistributedGlorotNormal creates sharded arrays
        and config serialization/deserialization"""
        init = dist_initializer.DistributedGlorotNormal()
        result = init(shape=(768, 256), dtype="float32")
        assert isinstance(result, jax.Array)
        assert result.shape == (768, 256)
        assert len(result.sharding.device_set) == 8

        # Get config
        config = init.get_config()

        # Serialize and deserialize
        serialized = keras.saving.serialize_keras_object(init)

        restored = keras.saving.deserialize_keras_object(serialized)

        # Check types match
        assert type(init) is type(restored), (
            f"Type mismatch: {type(init)} vs {type(restored)}"
        )

        # Check config matches
        restored_config = restored.get_config()
        assert config == restored_config, (
            f"Config mismatch: {config} vs {restored_config}"
        )

    def test_distributed_lecun_normal(self):
        """Test DistributedLecunNormal creates sharded arrays
        and config serialization/deserialization"""
        init = dist_initializer.DistributedLecunNormal()
        result = init(shape=(768, 256), dtype="float32")
        assert isinstance(result, jax.Array)
        assert result.shape == (768, 256)
        assert len(result.sharding.device_set) == 8

        # Get config
        config = init.get_config()

        # Serialize and deserialize
        serialized = keras.saving.serialize_keras_object(init)

        restored = keras.saving.deserialize_keras_object(serialized)

        # Check types match
        assert type(init) is type(restored), (
            f"Type mismatch: {type(init)} vs {type(restored)}"
        )

        # Check config matches
        restored_config = restored.get_config()
        assert config == restored_config, (
            f"Config mismatch: {config} vs {restored_config}"
        )

    def test_distributed_lecun_uniform(self):
        """Test DistributedLecunUniform creates sharded arrays
        and config serialization/deserialization"""
        init = dist_initializer.DistributedLecunUniform()
        result = init(shape=(768, 256), dtype="float32")
        assert isinstance(result, jax.Array)
        assert result.shape == (768, 256)
        assert len(result.sharding.device_set) == 8

        # Get config
        config = init.get_config()

        # Serialize and deserialize
        serialized = keras.saving.serialize_keras_object(init)

        restored = keras.saving.deserialize_keras_object(serialized)

        # Check types match
        assert type(init) is type(restored), (
            f"Type mismatch: {type(init)} vs {type(restored)}"
        )

        # Check config matches
        restored_config = restored.get_config()
        assert config == restored_config, (
            f"Config mismatch: {config} vs {restored_config}"
        )

    def test_distributed_he_normal(self):
        """Test DistributedHeNormal creates sharded arrays
        and config serialization/deserialization"""
        init = dist_initializer.DistributedHeNormal()
        result = init(shape=(768, 256), dtype="float32")
        assert isinstance(result, jax.Array)
        assert result.shape == (768, 256)
        assert len(result.sharding.device_set) == 8

        # Get config
        config = init.get_config()

        # Serialize and deserialize
        serialized = keras.saving.serialize_keras_object(init)

        restored = keras.saving.deserialize_keras_object(serialized)

        # Check types match
        assert type(init) is type(restored), (
            f"Type mismatch: {type(init)} vs {type(restored)}"
        )

        # Check config matches
        restored_config = restored.get_config()
        assert config == restored_config, (
            f"Config mismatch: {config} vs {restored_config}"
        )

    def test_distributed_he_uniform(self):
        """Test DistributedHeUniform creates sharded arrays
        and config serialization/deserialization"""
        init = dist_initializer.DistributedHeUniform()
        result = init(shape=(768, 256), dtype="float32")
        assert isinstance(result, jax.Array)
        assert result.shape == (768, 256)
        assert len(result.sharding.device_set) == 8

        # Get config
        config = init.get_config()

        # Serialize and deserialize
        serialized = keras.saving.serialize_keras_object(init)

        restored = keras.saving.deserialize_keras_object(serialized)

        # Check types match
        assert type(init) is type(restored), (
            f"Type mismatch: {type(init)} vs {type(restored)}"
        )

        # Check config matches
        restored_config = restored.get_config()
        assert config == restored_config, (
            f"Config mismatch: {config} vs {restored_config}"
        )
