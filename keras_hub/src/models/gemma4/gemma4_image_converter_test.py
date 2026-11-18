import numpy as np
import tensorflow as tf
from keras import ops

from keras_hub.src.models.gemma4.gemma4_image_converter import (
    Gemma4AspectRatioResizing,
)
from keras_hub.src.models.gemma4.gemma4_image_converter import (
    Gemma4ImageConverter,
)


class Gemma4ImageConverterTest(tf.test.TestCase):
    def test_image_converter_range(self):
        converter = Gemma4ImageConverter(
            patch_size=4,
            max_soft_tokens=1,
            scale=[1 / 255.0, 1 / 255.0, 1 / 255.0],
            offset=[0.0, 0.0, 0.0],
        )

        # Minimum valid size: patch_size * pooling_kernel_size = 4 * 3 = 12
        img = np.random.randint(0, 256, size=(1, 12, 12, 3)).astype("float32")

        # Process image
        outputs = converter(img)

        pixel_values = ops.convert_to_numpy(outputs["pixel_values"])

        # Verify range is in [0, 1] (or close due to float precision)
        # Note: scaled by 1/255
        self.assertAllInRange(pixel_values, 0.0, 1.0)

    def test_aspect_ratio_resizing(self):
        converter = Gemma4ImageConverter(patch_size=4, max_soft_tokens=1)

        # Non-square input: 2:1 aspect ratio, dimensions are multiples of
        # patch_size (4) so the aspect-ratio resizer has room to adjust.
        img = np.random.randint(0, 256, size=(1, 24, 12, 3)).astype("float32")

        outputs = converter(img)

        # Resizing should preserve aspect ratio or adjust logic
        # Here we just verify it runs and outputs expected keys
        self.assertIn("pixel_values", outputs)
        self.assertIn("pixel_position_ids", outputs)

    def test_two_pass_uint8_resize(self):
        """Two-pass uint8 resize against a PIL reference.

        `expected` is PIL `Image.fromarray(img).resize((12, 12),
        Image.BICUBIC)` on the 8x8 uint8 pattern below, identical under PIL
        12.3.0 and 13.0.0 (torchvision's uint8 path gives the same table),
        so the bounds anchor the layer to the HF processor rather than to
        itself. Measured gaps to that table: CPU (jax/tf/torch) max 1, 91.4%
        exact; A100 torch-CUDA max 2, 90.0% exact. Removing
        `a = ops.round(a)` gives max 5 and 61.1% (CPU) / 60.4% (CUDA) exact.
        """
        # Target pixels = 9 * (4**2) * (1**2) = 144
        # Input is 8x8, total 64 px. factor = sqrt(144/64) = 1.5.
        # Output height/width = 8 * 1.5 = 12
        resizer = Gemma4AspectRatioResizing(
            patch_size=4,
            max_soft_tokens=9,
            pooling_kernel_size=1,
            two_pass_uint8_resize=True,
        )
        expected = np.array(
            [
                [
                    [0, 21, 58],
                    [37, 99, 136],
                    [126, 144, 181],
                    [226, 9, 45],
                    [151, 33, 97],
                    [77, 129, 147],
                    [189, 229, 12],
                    [117, 154, 36],
                    [43, 80, 132],
                    [155, 192, 232],
                    [85, 122, 159],
                    [0, 27, 64],
                ],
                [
                    [45, 103, 140],
                    [110, 91, 127],
                    [180, 71, 112],
                    [156, 48, 106],
                    [147, 116, 94],
                    [138, 183, 74],
                    [120, 159, 51],
                    [113, 150, 119],
                    [105, 141, 186],
                    [87, 123, 162],
                    [69, 113, 150],
                    [57, 109, 146],
                ],
                [
                    [138, 155, 192],
                    [187, 78, 118],
                    [209, 25, 63],
                    [105, 125, 144],
                    [141, 201, 83],
                    [176, 212, 28],
                    [71, 108, 128],
                    [107, 144, 204],
                    [141, 179, 215],
                    [35, 74, 111],
                    [85, 104, 141],
                    [160, 161, 198],
                ],
                [
                    [244, 27, 63],
                    [168, 61, 120],
                    [111, 130, 150],
                    [192, 230, 46],
                    [135, 176, 58],
                    [77, 114, 133],
                    [158, 195, 233],
                    [101, 138, 179],
                    [41, 80, 117],
                    [111, 161, 198],
                    [195, 106, 143],
                    [250, 32, 69],
                ],
                [
                    [174, 73, 120],
                    [165, 143, 111],
                    [153, 212, 97],
                    [141, 181, 79],
                    [130, 166, 149],
                    [119, 156, 215],
                    [107, 144, 184],
                    [96, 133, 169],
                    [83, 122, 159],
                    [66, 110, 147],
                    [128, 97, 134],
                    [187, 89, 126],
                ],
                [
                    [106, 123, 176],
                    [163, 190, 102],
                    [194, 232, 44],
                    [89, 126, 115],
                    [125, 162, 203],
                    [160, 197, 235],
                    [55, 92, 129],
                    [91, 128, 165],
                    [125, 163, 200],
                    [17, 58, 95],
                    [60, 88, 125],
                    [128, 145, 182],
                ],
                [
                    [225, 11, 47],
                    [151, 63, 104],
                    [95, 145, 134],
                    [176, 216, 28],
                    [119, 156, 61],
                    [61, 98, 148],
                    [142, 179, 219],
                    [86, 122, 159],
                    [23, 64, 101],
                    [81, 145, 182],
                    [170, 90, 127],
                    [233, 16, 53],
                ],
                [
                    [158, 42, 104],
                    [149, 110, 96],
                    [137, 181, 79],
                    [125, 163, 49],
                    [114, 151, 115],
                    [103, 140, 184],
                    [91, 128, 166],
                    [62, 117, 154],
                    [71, 106, 143],
                    [182, 94, 131],
                    [189, 73, 118],
                    [162, 56, 110],
                ],
                [
                    [90, 143, 160],
                    [147, 192, 84],
                    [178, 214, 32],
                    [73, 110, 130],
                    [109, 146, 206],
                    [143, 181, 217],
                    [37, 76, 113],
                    [79, 112, 149],
                    [172, 146, 184],
                    [230, 40, 79],
                    [169, 90, 109],
                    [96, 164, 166],
                ],
                [
                    [209, 249, 31],
                    [135, 173, 74],
                    [79, 116, 149],
                    [160, 197, 237],
                    [103, 140, 180],
                    [43, 82, 119],
                    [113, 163, 200],
                    [203, 106, 143],
                    [236, 46, 85],
                    [128, 116, 166],
                    [158, 200, 111],
                    [215, 250, 37],
                ],
                [
                    [145, 182, 83],
                    [132, 169, 71],
                    [117, 154, 85],
                    [111, 148, 171],
                    [97, 134, 179],
                    [81, 120, 157],
                    [70, 114, 151],
                    [132, 100, 137],
                    [190, 84, 123],
                    [166, 73, 117],
                    [155, 136, 102],
                    [151, 193, 89],
                ],
                [
                    [55, 92, 165],
                    [133, 170, 71],
                    [178, 215, 0],
                    [42, 79, 81],
                    [93, 130, 185],
                    [144, 181, 218],
                    [9, 45, 82],
                    [33, 96, 133],
                    [129, 147, 184],
                    [229, 12, 48],
                    [156, 47, 93],
                    [61, 116, 171],
                ],
            ],
            dtype=np.float32,
        )

        # F1 input pattern
        img_np = ((np.arange(8 * 8 * 3).reshape(8, 8, 3) * 37) % 256).astype(
            "uint8"
        )
        img_np = img_np.astype(np.float32)[np.newaxis, ...]
        out = resizer(img_np)

        out_np = ops.convert_to_numpy(out[0])
        # Both sides are rounded to the uint8 grid, so differences are
        # integers. Measured against the PIL table: CPU max 1 on all three
        # backends, A100 torch-CUDA max 2 (GPU interpolation rounding); +1
        # guard band, one integer step, for other CUDA/cuDNN builds. The
        # single-pass mutant reaches 5.
        self.assertLessEqual(np.max(np.abs(out_np - expected)), 3.0)
        # Measured exact fraction: 91.4% CPU, 90.0% A100 torch-CUDA, i.e.
        # 10.0% inexact. The floor allows 30% inexact (3x headroom) and sits
        # above the single-pass mutant's 61.1% (CPU) / 60.4% (CUDA) exact,
        # so both this line and the max bound above (mutant max 5) keep the
        # mutation red.
        self.assertGreaterEqual(np.mean(out_np == expected), 0.70)

        # No-resize case (input already at the target size) returns the
        # input unchanged. A uint8-grid input, as HF would send, keeps the
        # identity true for the two-pass arithmetic as well as the early
        # return.
        img_12x12 = np.random.randint(0, 256, size=(1, 12, 12, 3)).astype(
            np.float32
        )
        out_no_resize = resizer(img_12x12)
        self.assertAllClose(ops.convert_to_numpy(out_no_resize), img_12x12)


if __name__ == "__main__":
    tf.test.main()
