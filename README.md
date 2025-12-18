---
library_name: keras-hub
---
### Model Overview
EfficientNets are a family of image classification models, which achieve state-of-the-art accuracy, yet being an order-of-magnitude smaller and faster than previous models.

We develop EfficientNets based on AutoML and Compound Scaling. In particular, we first use AutoML MNAS Mobile framework to develop a mobile-size baseline network, named as EfficientNet-B0; Then, we use the compound scaling method to scale up this baseline to obtain EfficientNet-B1 to EfficientNet-B7.

This class encapsulates the architectures for both EfficientNetV1 and EfficientNetV2. EfficientNetV2 uses Fused-MBConv Blocks and Neural Architecture Search (NAS) to make models sizes much smaller while still improving overall model quality.

This model is supported in both KerasCV and KerasHub. KerasCV will no longer be actively developed, so please try to use KerasHub.

## Links

* [EfficientNet Quickstart Notebook](https://www.kaggle.com/code/prasadsachin/efficientnet-quickstart-kerashub)
* [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)(ICML 2019)
* [Based on the original keras.applications EfficientNet](https://github.com/keras-team/keras/blob/master/keras/applications/efficientnet.py)
* [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298) (ICML 2021)
* [EfficientNet API Documentation](https://keras.io/keras_hub/api/models/efficientnet/)
* [KerasHub Beginner Guide](https://keras.io/guides/keras_hub/getting_started/)
* [KerasHub Model Publishing Guide](https://keras.io/guides/keras_hub/upload/)


## Installation

Keras and KerasHub can be installed with:

```
pip install -U -q keras-hub
pip install -U -q keras
```

Jax, TensorFlow, and Torch come preinstalled in Kaggle Notebooks. For instructions on installing them in another environment see the [Keras Getting Started](https://keras.io/getting_started/) page.

## Presets

The following model checkpoints are provided by the Keras team. Full code examples for each are available below.

| Preset name                        | Parameters | Description                                                                                                                                                                                                                                                                                               |
|------------------------------------|------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| efficientnet_b0_ra_imagenet | 5.3M      | EfficientNet B0 model pre-trained on the ImageNet 1k dataset with RandAugment recipe. |
| efficientnet_b0_ra4_e3600_r224_imagenet | 5.3M      | EfficientNet B0 model pre-trained on the ImageNet 1k dataset by Ross Wightman. Trained with timm scripts using hyper-parameters inspired by the MobileNet-V4 small, mixed with go-to hparams from timm and 'ResNet Strikes Back'. |
| efficientnet_b1_ft_imagenet | 7.8M      | EfficientNet B1 model fine-tuned on the ImageNet 1k dataset. |
| efficientnet_b1_ra4_e3600_r240_imagenet | 7.8M      | EfficientNet B1 model pre-trained on the ImageNet 1k dataset by Ross Wightman. Trained with timm scripts using hyper-parameters inspired by the MobileNet-V4 small, mixed with go-to hparams from timm and 'ResNet Strikes Back'. |
| efficientnet_b2_ra_imagenet | 9.1M      | EfficientNet B2 model pre-trained on the ImageNet 1k dataset with RandAugment recipe. |
| efficientnet_b3_ra2_imagenet | 12.2M      | EfficientNet B3 model pre-trained on the ImageNet 1k dataset with RandAugment2 recipe. |
| efficientnet_b4_ra2_imagenet | 19.3M      | EfficientNet B4 model pre-trained on the ImageNet 1k dataset with RandAugment2 recipe. |
| efficientnet_b5_sw_imagenet | 30.4M      | EfficientNet B5 model pre-trained on the ImageNet 12k dataset by Ross Wightman. Based on Swin Transformer train / pretrain recipe with modifications (related to both DeiT and ConvNeXt recipes). |
| efficientnet_b5_sw_ft_imagenet | 30.4M      | EfficientNet B5 model pre-trained on the ImageNet 12k dataset and fine-tuned on ImageNet-1k by Ross Wightman. Based on Swin Transformer train / pretrain recipe with modifications (related to both DeiT and ConvNeXt recipes). |
| efficientnet_el_ra_imagenet | 10.6M      | EfficientNet-EdgeTPU Large model trained on the ImageNet 1k dataset with RandAugment recipe. |
| efficientnet_em_ra2_imagenet | 6.9M      | EfficientNet-EdgeTPU Medium model trained on the ImageNet 1k dataset with RandAugment2 recipe. |
| efficientnet_es_ra_imagenet | 5.4M      | EfficientNet-EdgeTPU Small model trained on the ImageNet 1k dataset with RandAugment recipe. |
| efficientnet2_rw_m_agc_imagenet | 53.2M      | EfficientNet-v2 Medium model trained on the ImageNet 1k dataset with adaptive gradient clipping. |
| efficientnet2_rw_s_ra2_imagenet | 23.9M      | EfficientNet-v2 Small model trained on the ImageNet 1k dataset with RandAugment2 recipe. |
| efficientnet2_rw_t_ra2_imagenet | 13.6M      | EfficientNet-v2 Tiny model trained on the ImageNet 1k dataset with RandAugment2 recipe. |
| efficientnet_lite0_ra_imagenet | 4.7M      | EfficientNet-Lite model fine-trained on the ImageNet 1k dataset with RandAugment recipe. |

## Model card
https://arxiv.org/abs/1905.11946

## Example Usage
Load
```python
classifier = keras_hub.models.EfficientNetImageClassifier.from_preset(
    "efficientnet_b0_ra_imagenet",
)
```
Predict
```python
batch_size = 1
images = keras.random.normal(shape=(batch_size, 96, 96, 3))
classifier.predict(images)
```
Train, specify `num_classes` to load randomly initialized classifier head.
```python
num_classes = 2
labels = keras.random.randint(shape=(batch_size,), minval=0, maxval=num_classes)
classifier = keras_hub.models.EfficientNetImageClassifier.from_preset(
    "efficientnet_b0_ra_imagenet",
    num_classes=num_classes,
)
classifier.preprocessor.image_size = (96, 96)
classifier.fit(images, labels, epochs=3)
```

## Example Usage with Hugging Face URI

Load
```python
classifier = keras_hub.models.EfficientNetImageClassifier.from_preset(
    "efficientnet_b0_ra_imagenet",
)
```
Predict
```python
batch_size = 1
images = keras.random.normal(shape=(batch_size, 96, 96, 3))
classifier.predict(images)
```
Train, specify `num_classes` to load randomly initialized classifier head.
```python
num_classes = 2
labels = keras.random.randint(shape=(batch_size,), minval=0, maxval=num_classes)
classifier = keras_hub.models.EfficientNetImageClassifier.from_preset(
    "efficientnet_b0_ra_imagenet",
    num_classes=num_classes,
)
classifier.preprocessor.image_size = (96, 96)
classifier.fit(images, labels, epochs=3)
```
