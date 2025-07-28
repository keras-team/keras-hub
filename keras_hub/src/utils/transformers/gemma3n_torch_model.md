Gemma3nForConditionalGeneration(
  (model): Gemma3nModel(
    (vision_tower): TimmWrapperModel(
      (timm_model): MobileNetV5Encoder(
        (conv_stem): ConvNormAct(
          (conv): Conv2dSame(3, 64, kernel_size=(3, 3), stride=(2, 2))
          (bn): RmsNormAct2d(
            (drop): Identity()
            (act): GELU(approximate='tanh')
          )
        )
        (blocks): Sequential(
          (0): Sequential(
            (0): EdgeResidual(
              (conv_exp): Conv2dSame(64, 256, kernel_size=(3, 3), stride=(2, 2), bias=False)
              (bn1): RmsNormAct2d(
                (drop): Identity()
                (act): GELU(approximate='tanh')
              )
              (aa): Identity()
              (se): Identity()
              (conv_pwl): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn2): RmsNormAct2d(
                (drop): Identity()
                (act): Identity()
              )
              (drop_path): Identity()
            )
            (1): EdgeResidual(
              (conv_exp): Conv2d(128, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn1): RmsNormAct2d(
                (drop): Identity()
                (act): GELU(approximate='tanh')
              )
              (aa): Identity()
              (se): Identity()
              (conv_pwl): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn2): RmsNormAct2d(
                (drop): Identity()
                (act): Identity()
              )
              (drop_path): Identity()
            )
            (2): EdgeResidual(
              (conv_exp): Conv2d(128, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn1): RmsNormAct2d(
                (drop): Identity()
                (act): GELU(approximate='tanh')
              )
              (aa): Identity()
              (se): Identity()
              (conv_pwl): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn2): RmsNormAct2d(
                (drop): Identity()
                (act): Identity()
              )
              (drop_path): Identity()
            )
          )
          (1): Sequential(
            (0): UniversalInvertedResidual(
              (dw_start): ConvNormAct(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (pw_exp): ConvNormAct(
                (conv): Conv2d(128, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): GELU(approximate='tanh')
                )
              )
              (dw_mid): ConvNormAct(
                (conv): Conv2dSame(768, 768, kernel_size=(5, 5), stride=(2, 2), groups=768, bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): GELU(approximate='tanh')
                )
              )
              (se): Identity()
              (pw_proj): ConvNormAct(
                (conv): Conv2d(768, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (dw_end): Identity()
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (1): UniversalInvertedResidual(
              (dw_start): ConvNormAct(
                (conv): Conv2d(256, 256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=256, bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (pw_exp): ConvNormAct(
                (conv): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): GELU(approximate='tanh')
                )
              )
              (dw_mid): Identity()
              (se): Identity()
              (pw_proj): ConvNormAct(
                (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (dw_end): Identity()
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (2): UniversalInvertedResidual(
              (dw_start): ConvNormAct(
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (pw_exp): ConvNormAct(
                (conv): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): GELU(approximate='tanh')
                )
              )
              (dw_mid): Identity()
              (se): Identity()
              (pw_proj): ConvNormAct(
                (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (dw_end): Identity()
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (3): UniversalInvertedResidual(
              (dw_start): ConvNormAct(
                (conv): Conv2d(256, 256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=256, bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (pw_exp): ConvNormAct(
                (conv): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): GELU(approximate='tanh')
                )
              )
              (dw_mid): Identity()
              (se): Identity()
              (pw_proj): ConvNormAct(
                (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (dw_end): Identity()
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (4): UniversalInvertedResidual(
              (dw_start): ConvNormAct(
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (pw_exp): ConvNormAct(
                (conv): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): GELU(approximate='tanh')
                )
              )
              (dw_mid): Identity()
              (se): Identity()
              (pw_proj): ConvNormAct(
                (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (dw_end): Identity()
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
          )
          (2): Sequential(
            (0): UniversalInvertedResidual(
              (dw_start): ConvNormAct(
                (conv): Conv2d(256, 256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=256, bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (pw_exp): ConvNormAct(
                (conv): Conv2d(256, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): GELU(approximate='tanh')
                )
              )
              (dw_mid): ConvNormAct(
                (conv): Conv2dSame(1536, 1536, kernel_size=(5, 5), stride=(2, 2), groups=1536, bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): GELU(approximate='tanh')
                )
              )
              (se): Identity()
              (pw_proj): ConvNormAct(
                (conv): Conv2d(1536, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (dw_end): Identity()
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (1): UniversalInvertedResidual(
              (dw_start): ConvNormAct(
                (conv): Conv2d(640, 640, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=640, bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (pw_exp): ConvNormAct(
                (conv): Conv2d(640, 2560, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): GELU(approximate='tanh')
                )
              )
              (dw_mid): Identity()
              (se): Identity()
              (pw_proj): ConvNormAct(
                (conv): Conv2d(2560, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (dw_end): Identity()
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (2): UniversalInvertedResidual(
              (dw_start): ConvNormAct(
                (conv): Conv2d(640, 640, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=640, bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (pw_exp): ConvNormAct(
                (conv): Conv2d(640, 2560, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): GELU(approximate='tanh')
                )
              )
              (dw_mid): Identity()
              (se): Identity()
              (pw_proj): ConvNormAct(
                (conv): Conv2d(2560, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (dw_end): Identity()
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (3): UniversalInvertedResidual(
              (dw_start): ConvNormAct(
                (conv): Conv2d(640, 640, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=640, bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (pw_exp): ConvNormAct(
                (conv): Conv2d(640, 2560, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): GELU(approximate='tanh')
                )
              )
              (dw_mid): Identity()
              (se): Identity()
              (pw_proj): ConvNormAct(
                (conv): Conv2d(2560, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (dw_end): Identity()
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (4): UniversalInvertedResidual(
              (dw_start): ConvNormAct(
                (conv): Conv2d(640, 640, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=640, bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (pw_exp): ConvNormAct(
                (conv): Conv2d(640, 2560, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): GELU(approximate='tanh')
                )
              )
              (dw_mid): Identity()
              (se): Identity()
              (pw_proj): ConvNormAct(
                (conv): Conv2d(2560, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (dw_end): Identity()
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (5): UniversalInvertedResidual(
              (dw_start): ConvNormAct(
                (conv): Conv2d(640, 640, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=640, bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (pw_exp): ConvNormAct(
                (conv): Conv2d(640, 2560, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): GELU(approximate='tanh')
                )
              )
              (dw_mid): Identity()
              (se): Identity()
              (pw_proj): ConvNormAct(
                (conv): Conv2d(2560, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (dw_end): Identity()
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (6): UniversalInvertedResidual(
              (dw_start): ConvNormAct(
                (conv): Conv2d(640, 640, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=640, bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (pw_exp): ConvNormAct(
                (conv): Conv2d(640, 2560, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): GELU(approximate='tanh')
                )
              )
              (dw_mid): Identity()
              (se): Identity()
              (pw_proj): ConvNormAct(
                (conv): Conv2d(2560, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (dw_end): Identity()
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (7): UniversalInvertedResidual(
              (dw_start): ConvNormAct(
                (conv): Conv2d(640, 640, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=640, bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (pw_exp): ConvNormAct(
                (conv): Conv2d(640, 2560, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): GELU(approximate='tanh')
                )
              )
              (dw_mid): Identity()
              (se): Identity()
              (pw_proj): ConvNormAct(
                (conv): Conv2d(2560, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (dw_end): Identity()
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (8): UniversalInvertedResidual(
              (dw_start): Identity()
              (pw_exp): ConvNormAct(
                (conv): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): GELU(approximate='tanh')
                )
              )
              (dw_mid): Identity()
              (se): Identity()
              (pw_proj): ConvNormAct(
                (conv): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (dw_end): Identity()
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (9): MobileAttention(
              (norm): RmsNormAct2d(
                (drop): Identity()
                (act): Identity()
              )
              (attn): MultiQueryAttention2d(
                (query): Sequential(
                  (proj): Conv2d(640, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (key): Sequential(
                  (down_conv): Conv2dSame(640, 640, kernel_size=(3, 3), stride=(2, 2), groups=640, bias=False)
                  (norm): RmsNorm2d()
                  (proj): Conv2d(640, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (value): Sequential(
                  (down_conv): Conv2dSame(640, 640, kernel_size=(3, 3), stride=(2, 2), groups=640, bias=False)
                  (norm): RmsNorm2d()
                  (proj): Conv2d(640, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (attn_drop): Dropout(p=0.0, inplace=False)
                (output): Sequential(
                  (proj): Conv2d(768, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (10): UniversalInvertedResidual(
              (dw_start): Identity()
              (pw_exp): ConvNormAct(
                (conv): Conv2d(640, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): GELU(approximate='tanh')
                )
              )
              (dw_mid): Identity()
              (se): Identity()
              (pw_proj): ConvNormAct(
                (conv): Conv2d(1280, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (dw_end): Identity()
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (11): MobileAttention(
              (norm): RmsNormAct2d(
                (drop): Identity()
                (act): Identity()
              )
              (attn): MultiQueryAttention2d(
                (query): Sequential(
                  (proj): Conv2d(640, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (key): Sequential(
                  (down_conv): Conv2dSame(640, 640, kernel_size=(3, 3), stride=(2, 2), groups=640, bias=False)
                  (norm): RmsNorm2d()
                  (proj): Conv2d(640, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (value): Sequential(
                  (down_conv): Conv2dSame(640, 640, kernel_size=(3, 3), stride=(2, 2), groups=640, bias=False)
                  (norm): RmsNorm2d()
                  (proj): Conv2d(640, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (attn_drop): Dropout(p=0.0, inplace=False)
                (output): Sequential(
                  (proj): Conv2d(768, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (12): UniversalInvertedResidual(
              (dw_start): Identity()
              (pw_exp): ConvNormAct(
                (conv): Conv2d(640, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): GELU(approximate='tanh')
                )
              )
              (dw_mid): Identity()
              (se): Identity()
              (pw_proj): ConvNormAct(
                (conv): Conv2d(1280, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (dw_end): Identity()
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (13): MobileAttention(
              (norm): RmsNormAct2d(
                (drop): Identity()
                (act): Identity()
              )
              (attn): MultiQueryAttention2d(
                (query): Sequential(
                  (proj): Conv2d(640, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (key): Sequential(
                  (down_conv): Conv2dSame(640, 640, kernel_size=(3, 3), stride=(2, 2), groups=640, bias=False)
                  (norm): RmsNorm2d()
                  (proj): Conv2d(640, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (value): Sequential(
                  (down_conv): Conv2dSame(640, 640, kernel_size=(3, 3), stride=(2, 2), groups=640, bias=False)
                  (norm): RmsNorm2d()
                  (proj): Conv2d(640, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (attn_drop): Dropout(p=0.0, inplace=False)
                (output): Sequential(
                  (proj): Conv2d(768, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (14): UniversalInvertedResidual(
              (dw_start): Identity()
              (pw_exp): ConvNormAct(
                (conv): Conv2d(640, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): GELU(approximate='tanh')
                )
              )
              (dw_mid): Identity()
              (se): Identity()
              (pw_proj): ConvNormAct(
                (conv): Conv2d(1280, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (dw_end): Identity()
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (15): MobileAttention(
              (norm): RmsNormAct2d(
                (drop): Identity()
                (act): Identity()
              )
              (attn): MultiQueryAttention2d(
                (query): Sequential(
                  (proj): Conv2d(640, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (key): Sequential(
                  (down_conv): Conv2dSame(640, 640, kernel_size=(3, 3), stride=(2, 2), groups=640, bias=False)
                  (norm): RmsNorm2d()
                  (proj): Conv2d(640, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (value): Sequential(
                  (down_conv): Conv2dSame(640, 640, kernel_size=(3, 3), stride=(2, 2), groups=640, bias=False)
                  (norm): RmsNorm2d()
                  (proj): Conv2d(640, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (attn_drop): Dropout(p=0.0, inplace=False)
                (output): Sequential(
                  (proj): Conv2d(768, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (16): UniversalInvertedResidual(
              (dw_start): Identity()
              (pw_exp): ConvNormAct(
                (conv): Conv2d(640, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): GELU(approximate='tanh')
                )
              )
              (dw_mid): Identity()
              (se): Identity()
              (pw_proj): ConvNormAct(
                (conv): Conv2d(1280, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (dw_end): Identity()
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (17): MobileAttention(
              (norm): RmsNormAct2d(
                (drop): Identity()
                (act): Identity()
              )
              (attn): MultiQueryAttention2d(
                (query): Sequential(
                  (proj): Conv2d(640, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (key): Sequential(
                  (down_conv): Conv2dSame(640, 640, kernel_size=(3, 3), stride=(2, 2), groups=640, bias=False)
                  (norm): RmsNorm2d()
                  (proj): Conv2d(640, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (value): Sequential(
                  (down_conv): Conv2dSame(640, 640, kernel_size=(3, 3), stride=(2, 2), groups=640, bias=False)
                  (norm): RmsNorm2d()
                  (proj): Conv2d(640, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (attn_drop): Dropout(p=0.0, inplace=False)
                (output): Sequential(
                  (proj): Conv2d(768, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (18): UniversalInvertedResidual(
              (dw_start): Identity()
              (pw_exp): ConvNormAct(
                (conv): Conv2d(640, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): GELU(approximate='tanh')
                )
              )
              (dw_mid): Identity()
              (se): Identity()
              (pw_proj): ConvNormAct(
                (conv): Conv2d(1280, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (dw_end): Identity()
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (19): MobileAttention(
              (norm): RmsNormAct2d(
                (drop): Identity()
                (act): Identity()
              )
              (attn): MultiQueryAttention2d(
                (query): Sequential(
                  (proj): Conv2d(640, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (key): Sequential(
                  (down_conv): Conv2dSame(640, 640, kernel_size=(3, 3), stride=(2, 2), groups=640, bias=False)
                  (norm): RmsNorm2d()
                  (proj): Conv2d(640, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (value): Sequential(
                  (down_conv): Conv2dSame(640, 640, kernel_size=(3, 3), stride=(2, 2), groups=640, bias=False)
                  (norm): RmsNorm2d()
                  (proj): Conv2d(640, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (attn_drop): Dropout(p=0.0, inplace=False)
                (output): Sequential(
                  (proj): Conv2d(768, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (20): UniversalInvertedResidual(
              (dw_start): Identity()
              (pw_exp): ConvNormAct(
                (conv): Conv2d(640, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): GELU(approximate='tanh')
                )
              )
              (dw_mid): Identity()
              (se): Identity()
              (pw_proj): ConvNormAct(
                (conv): Conv2d(1280, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (dw_end): Identity()
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (21): MobileAttention(
              (norm): RmsNormAct2d(
                (drop): Identity()
                (act): Identity()
              )
              (attn): MultiQueryAttention2d(
                (query): Sequential(
                  (proj): Conv2d(640, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (key): Sequential(
                  (down_conv): Conv2dSame(640, 640, kernel_size=(3, 3), stride=(2, 2), groups=640, bias=False)
                  (norm): RmsNorm2d()
                  (proj): Conv2d(640, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (value): Sequential(
                  (down_conv): Conv2dSame(640, 640, kernel_size=(3, 3), stride=(2, 2), groups=640, bias=False)
                  (norm): RmsNorm2d()
                  (proj): Conv2d(640, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (attn_drop): Dropout(p=0.0, inplace=False)
                (output): Sequential(
                  (proj): Conv2d(768, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (22): UniversalInvertedResidual(
              (dw_start): Identity()
              (pw_exp): ConvNormAct(
                (conv): Conv2d(640, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): GELU(approximate='tanh')
                )
              )
              (dw_mid): Identity()
              (se): Identity()
              (pw_proj): ConvNormAct(
                (conv): Conv2d(1280, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (dw_end): Identity()
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (23): MobileAttention(
              (norm): RmsNormAct2d(
                (drop): Identity()
                (act): Identity()
              )
              (attn): MultiQueryAttention2d(
                (query): Sequential(
                  (proj): Conv2d(640, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (key): Sequential(
                  (down_conv): Conv2dSame(640, 640, kernel_size=(3, 3), stride=(2, 2), groups=640, bias=False)
                  (norm): RmsNorm2d()
                  (proj): Conv2d(640, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (value): Sequential(
                  (down_conv): Conv2dSame(640, 640, kernel_size=(3, 3), stride=(2, 2), groups=640, bias=False)
                  (norm): RmsNorm2d()
                  (proj): Conv2d(640, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (attn_drop): Dropout(p=0.0, inplace=False)
                (output): Sequential(
                  (proj): Conv2d(768, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (24): UniversalInvertedResidual(
              (dw_start): Identity()
              (pw_exp): ConvNormAct(
                (conv): Conv2d(640, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): GELU(approximate='tanh')
                )
              )
              (dw_mid): Identity()
              (se): Identity()
              (pw_proj): ConvNormAct(
                (conv): Conv2d(1280, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (dw_end): Identity()
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (25): MobileAttention(
              (norm): RmsNormAct2d(
                (drop): Identity()
                (act): Identity()
              )
              (attn): MultiQueryAttention2d(
                (query): Sequential(
                  (proj): Conv2d(640, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (key): Sequential(
                  (down_conv): Conv2dSame(640, 640, kernel_size=(3, 3), stride=(2, 2), groups=640, bias=False)
                  (norm): RmsNorm2d()
                  (proj): Conv2d(640, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (value): Sequential(
                  (down_conv): Conv2dSame(640, 640, kernel_size=(3, 3), stride=(2, 2), groups=640, bias=False)
                  (norm): RmsNorm2d()
                  (proj): Conv2d(640, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (attn_drop): Dropout(p=0.0, inplace=False)
                (output): Sequential(
                  (proj): Conv2d(768, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (26): UniversalInvertedResidual(
              (dw_start): Identity()
              (pw_exp): ConvNormAct(
                (conv): Conv2d(640, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): GELU(approximate='tanh')
                )
              )
              (dw_mid): Identity()
              (se): Identity()
              (pw_proj): ConvNormAct(
                (conv): Conv2d(1280, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (dw_end): Identity()
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (27): MobileAttention(
              (norm): RmsNormAct2d(
                (drop): Identity()
                (act): Identity()
              )
              (attn): MultiQueryAttention2d(
                (query): Sequential(
                  (proj): Conv2d(640, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (key): Sequential(
                  (down_conv): Conv2dSame(640, 640, kernel_size=(3, 3), stride=(2, 2), groups=640, bias=False)
                  (norm): RmsNorm2d()
                  (proj): Conv2d(640, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (value): Sequential(
                  (down_conv): Conv2dSame(640, 640, kernel_size=(3, 3), stride=(2, 2), groups=640, bias=False)
                  (norm): RmsNorm2d()
                  (proj): Conv2d(640, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (attn_drop): Dropout(p=0.0, inplace=False)
                (output): Sequential(
                  (proj): Conv2d(768, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (28): UniversalInvertedResidual(
              (dw_start): Identity()
              (pw_exp): ConvNormAct(
                (conv): Conv2d(640, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): GELU(approximate='tanh')
                )
              )
              (dw_mid): Identity()
              (se): Identity()
              (pw_proj): ConvNormAct(
                (conv): Conv2d(1280, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (dw_end): Identity()
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (29): MobileAttention(
              (norm): RmsNormAct2d(
                (drop): Identity()
                (act): Identity()
              )
              (attn): MultiQueryAttention2d(
                (query): Sequential(
                  (proj): Conv2d(640, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (key): Sequential(
                  (down_conv): Conv2dSame(640, 640, kernel_size=(3, 3), stride=(2, 2), groups=640, bias=False)
                  (norm): RmsNorm2d()
                  (proj): Conv2d(640, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (value): Sequential(
                  (down_conv): Conv2dSame(640, 640, kernel_size=(3, 3), stride=(2, 2), groups=640, bias=False)
                  (norm): RmsNorm2d()
                  (proj): Conv2d(640, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (attn_drop): Dropout(p=0.0, inplace=False)
                (output): Sequential(
                  (proj): Conv2d(768, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (30): UniversalInvertedResidual(
              (dw_start): Identity()
              (pw_exp): ConvNormAct(
                (conv): Conv2d(640, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): GELU(approximate='tanh')
                )
              )
              (dw_mid): Identity()
              (se): Identity()
              (pw_proj): ConvNormAct(
                (conv): Conv2d(1280, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (dw_end): Identity()
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (31): MobileAttention(
              (norm): RmsNormAct2d(
                (drop): Identity()
                (act): Identity()
              )
              (attn): MultiQueryAttention2d(
                (query): Sequential(
                  (proj): Conv2d(640, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (key): Sequential(
                  (down_conv): Conv2dSame(640, 640, kernel_size=(3, 3), stride=(2, 2), groups=640, bias=False)
                  (norm): RmsNorm2d()
                  (proj): Conv2d(640, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (value): Sequential(
                  (down_conv): Conv2dSame(640, 640, kernel_size=(3, 3), stride=(2, 2), groups=640, bias=False)
                  (norm): RmsNorm2d()
                  (proj): Conv2d(640, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (attn_drop): Dropout(p=0.0, inplace=False)
                (output): Sequential(
                  (proj): Conv2d(768, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (32): UniversalInvertedResidual(
              (dw_start): Identity()
              (pw_exp): ConvNormAct(
                (conv): Conv2d(640, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): GELU(approximate='tanh')
                )
              )
              (dw_mid): Identity()
              (se): Identity()
              (pw_proj): ConvNormAct(
                (conv): Conv2d(1280, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (dw_end): Identity()
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (33): MobileAttention(
              (norm): RmsNormAct2d(
                (drop): Identity()
                (act): Identity()
              )
              (attn): MultiQueryAttention2d(
                (query): Sequential(
                  (proj): Conv2d(640, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (key): Sequential(
                  (down_conv): Conv2dSame(640, 640, kernel_size=(3, 3), stride=(2, 2), groups=640, bias=False)
                  (norm): RmsNorm2d()
                  (proj): Conv2d(640, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (value): Sequential(
                  (down_conv): Conv2dSame(640, 640, kernel_size=(3, 3), stride=(2, 2), groups=640, bias=False)
                  (norm): RmsNorm2d()
                  (proj): Conv2d(640, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (attn_drop): Dropout(p=0.0, inplace=False)
                (output): Sequential(
                  (proj): Conv2d(768, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (34): UniversalInvertedResidual(
              (dw_start): Identity()
              (pw_exp): ConvNormAct(
                (conv): Conv2d(640, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): GELU(approximate='tanh')
                )
              )
              (dw_mid): Identity()
              (se): Identity()
              (pw_proj): ConvNormAct(
                (conv): Conv2d(1280, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (dw_end): Identity()
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (35): MobileAttention(
              (norm): RmsNormAct2d(
                (drop): Identity()
                (act): Identity()
              )
              (attn): MultiQueryAttention2d(
                (query): Sequential(
                  (proj): Conv2d(640, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (key): Sequential(
                  (down_conv): Conv2dSame(640, 640, kernel_size=(3, 3), stride=(2, 2), groups=640, bias=False)
                  (norm): RmsNorm2d()
                  (proj): Conv2d(640, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (value): Sequential(
                  (down_conv): Conv2dSame(640, 640, kernel_size=(3, 3), stride=(2, 2), groups=640, bias=False)
                  (norm): RmsNorm2d()
                  (proj): Conv2d(640, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (attn_drop): Dropout(p=0.0, inplace=False)
                (output): Sequential(
                  (proj): Conv2d(768, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (36): UniversalInvertedResidual(
              (dw_start): Identity()
              (pw_exp): ConvNormAct(
                (conv): Conv2d(640, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): GELU(approximate='tanh')
                )
              )
              (dw_mid): Identity()
              (se): Identity()
              (pw_proj): ConvNormAct(
                (conv): Conv2d(1280, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (dw_end): Identity()
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
          )
          (3): Sequential(
            (0): UniversalInvertedResidual(
              (dw_start): ConvNormAct(
                (conv): Conv2d(640, 640, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=640, bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (pw_exp): ConvNormAct(
                (conv): Conv2d(640, 3840, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): GELU(approximate='tanh')
                )
              )
              (dw_mid): ConvNormAct(
                (conv): Conv2dSame(3840, 3840, kernel_size=(5, 5), stride=(2, 2), groups=3840, bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): GELU(approximate='tanh')
                )
              )
              (se): Identity()
              (pw_proj): ConvNormAct(
                (conv): Conv2d(3840, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (dw_end): Identity()
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (1): MobileAttention(
              (norm): RmsNormAct2d(
                (drop): Identity()
                (act): Identity()
              )
              (attn): MultiQueryAttention2d(
                (query): Sequential(
                  (proj): Conv2d(1280, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (key): Sequential(
                  (proj): Conv2d(1280, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (value): Sequential(
                  (proj): Conv2d(1280, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (attn_drop): Dropout(p=0.0, inplace=False)
                (output): Sequential(
                  (proj): Conv2d(1536, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (2): UniversalInvertedResidual(
              (dw_start): Identity()
              (pw_exp): ConvNormAct(
                (conv): Conv2d(1280, 2560, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): GELU(approximate='tanh')
                )
              )
              (dw_mid): Identity()
              (se): Identity()
              (pw_proj): ConvNormAct(
                (conv): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (dw_end): Identity()
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (3): MobileAttention(
              (norm): RmsNormAct2d(
                (drop): Identity()
                (act): Identity()
              )
              (attn): MultiQueryAttention2d(
                (query): Sequential(
                  (proj): Conv2d(1280, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (key): Sequential(
                  (proj): Conv2d(1280, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (value): Sequential(
                  (proj): Conv2d(1280, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (attn_drop): Dropout(p=0.0, inplace=False)
                (output): Sequential(
                  (proj): Conv2d(1536, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (4): UniversalInvertedResidual(
              (dw_start): Identity()
              (pw_exp): ConvNormAct(
                (conv): Conv2d(1280, 2560, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): GELU(approximate='tanh')
                )
              )
              (dw_mid): Identity()
              (se): Identity()
              (pw_proj): ConvNormAct(
                (conv): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (dw_end): Identity()
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (5): MobileAttention(
              (norm): RmsNormAct2d(
                (drop): Identity()
                (act): Identity()
              )
              (attn): MultiQueryAttention2d(
                (query): Sequential(
                  (proj): Conv2d(1280, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (key): Sequential(
                  (proj): Conv2d(1280, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (value): Sequential(
                  (proj): Conv2d(1280, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (attn_drop): Dropout(p=0.0, inplace=False)
                (output): Sequential(
                  (proj): Conv2d(1536, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (6): UniversalInvertedResidual(
              (dw_start): Identity()
              (pw_exp): ConvNormAct(
                (conv): Conv2d(1280, 2560, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): GELU(approximate='tanh')
                )
              )
              (dw_mid): Identity()
              (se): Identity()
              (pw_proj): ConvNormAct(
                (conv): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (dw_end): Identity()
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (7): MobileAttention(
              (norm): RmsNormAct2d(
                (drop): Identity()
                (act): Identity()
              )
              (attn): MultiQueryAttention2d(
                (query): Sequential(
                  (proj): Conv2d(1280, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (key): Sequential(
                  (proj): Conv2d(1280, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (value): Sequential(
                  (proj): Conv2d(1280, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (attn_drop): Dropout(p=0.0, inplace=False)
                (output): Sequential(
                  (proj): Conv2d(1536, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (8): UniversalInvertedResidual(
              (dw_start): Identity()
              (pw_exp): ConvNormAct(
                (conv): Conv2d(1280, 2560, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): GELU(approximate='tanh')
                )
              )
              (dw_mid): Identity()
              (se): Identity()
              (pw_proj): ConvNormAct(
                (conv): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (dw_end): Identity()
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (9): MobileAttention(
              (norm): RmsNormAct2d(
                (drop): Identity()
                (act): Identity()
              )
              (attn): MultiQueryAttention2d(
                (query): Sequential(
                  (proj): Conv2d(1280, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (key): Sequential(
                  (proj): Conv2d(1280, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (value): Sequential(
                  (proj): Conv2d(1280, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (attn_drop): Dropout(p=0.0, inplace=False)
                (output): Sequential(
                  (proj): Conv2d(1536, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (10): UniversalInvertedResidual(
              (dw_start): Identity()
              (pw_exp): ConvNormAct(
                (conv): Conv2d(1280, 2560, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): GELU(approximate='tanh')
                )
              )
              (dw_mid): Identity()
              (se): Identity()
              (pw_proj): ConvNormAct(
                (conv): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (dw_end): Identity()
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (11): MobileAttention(
              (norm): RmsNormAct2d(
                (drop): Identity()
                (act): Identity()
              )
              (attn): MultiQueryAttention2d(
                (query): Sequential(
                  (proj): Conv2d(1280, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (key): Sequential(
                  (proj): Conv2d(1280, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (value): Sequential(
                  (proj): Conv2d(1280, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (attn_drop): Dropout(p=0.0, inplace=False)
                (output): Sequential(
                  (proj): Conv2d(1536, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (12): UniversalInvertedResidual(
              (dw_start): Identity()
              (pw_exp): ConvNormAct(
                (conv): Conv2d(1280, 2560, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): GELU(approximate='tanh')
                )
              )
              (dw_mid): Identity()
              (se): Identity()
              (pw_proj): ConvNormAct(
                (conv): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (dw_end): Identity()
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (13): MobileAttention(
              (norm): RmsNormAct2d(
                (drop): Identity()
                (act): Identity()
              )
              (attn): MultiQueryAttention2d(
                (query): Sequential(
                  (proj): Conv2d(1280, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (key): Sequential(
                  (proj): Conv2d(1280, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (value): Sequential(
                  (proj): Conv2d(1280, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (attn_drop): Dropout(p=0.0, inplace=False)
                (output): Sequential(
                  (proj): Conv2d(1536, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (14): UniversalInvertedResidual(
              (dw_start): Identity()
              (pw_exp): ConvNormAct(
                (conv): Conv2d(1280, 2560, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): GELU(approximate='tanh')
                )
              )
              (dw_mid): Identity()
              (se): Identity()
              (pw_proj): ConvNormAct(
                (conv): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (dw_end): Identity()
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (15): MobileAttention(
              (norm): RmsNormAct2d(
                (drop): Identity()
                (act): Identity()
              )
              (attn): MultiQueryAttention2d(
                (query): Sequential(
                  (proj): Conv2d(1280, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (key): Sequential(
                  (proj): Conv2d(1280, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (value): Sequential(
                  (proj): Conv2d(1280, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (attn_drop): Dropout(p=0.0, inplace=False)
                (output): Sequential(
                  (proj): Conv2d(1536, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (16): UniversalInvertedResidual(
              (dw_start): Identity()
              (pw_exp): ConvNormAct(
                (conv): Conv2d(1280, 2560, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): GELU(approximate='tanh')
                )
              )
              (dw_mid): Identity()
              (se): Identity()
              (pw_proj): ConvNormAct(
                (conv): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (dw_end): Identity()
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (17): MobileAttention(
              (norm): RmsNormAct2d(
                (drop): Identity()
                (act): Identity()
              )
              (attn): MultiQueryAttention2d(
                (query): Sequential(
                  (proj): Conv2d(1280, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (key): Sequential(
                  (proj): Conv2d(1280, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (value): Sequential(
                  (proj): Conv2d(1280, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (attn_drop): Dropout(p=0.0, inplace=False)
                (output): Sequential(
                  (proj): Conv2d(1536, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (18): UniversalInvertedResidual(
              (dw_start): Identity()
              (pw_exp): ConvNormAct(
                (conv): Conv2d(1280, 2560, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): GELU(approximate='tanh')
                )
              )
              (dw_mid): Identity()
              (se): Identity()
              (pw_proj): ConvNormAct(
                (conv): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (dw_end): Identity()
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (19): MobileAttention(
              (norm): RmsNormAct2d(
                (drop): Identity()
                (act): Identity()
              )
              (attn): MultiQueryAttention2d(
                (query): Sequential(
                  (proj): Conv2d(1280, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (key): Sequential(
                  (proj): Conv2d(1280, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (value): Sequential(
                  (proj): Conv2d(1280, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (attn_drop): Dropout(p=0.0, inplace=False)
                (output): Sequential(
                  (proj): Conv2d(1536, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (20): UniversalInvertedResidual(
              (dw_start): Identity()
              (pw_exp): ConvNormAct(
                (conv): Conv2d(1280, 2560, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): GELU(approximate='tanh')
                )
              )
              (dw_mid): Identity()
              (se): Identity()
              (pw_proj): ConvNormAct(
                (conv): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (dw_end): Identity()
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (21): MobileAttention(
              (norm): RmsNormAct2d(
                (drop): Identity()
                (act): Identity()
              )
              (attn): MultiQueryAttention2d(
                (query): Sequential(
                  (proj): Conv2d(1280, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (key): Sequential(
                  (proj): Conv2d(1280, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (value): Sequential(
                  (proj): Conv2d(1280, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (attn_drop): Dropout(p=0.0, inplace=False)
                (output): Sequential(
                  (proj): Conv2d(1536, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (22): UniversalInvertedResidual(
              (dw_start): Identity()
              (pw_exp): ConvNormAct(
                (conv): Conv2d(1280, 2560, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): GELU(approximate='tanh')
                )
              )
              (dw_mid): Identity()
              (se): Identity()
              (pw_proj): ConvNormAct(
                (conv): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (dw_end): Identity()
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (23): MobileAttention(
              (norm): RmsNormAct2d(
                (drop): Identity()
                (act): Identity()
              )
              (attn): MultiQueryAttention2d(
                (query): Sequential(
                  (proj): Conv2d(1280, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (key): Sequential(
                  (proj): Conv2d(1280, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (value): Sequential(
                  (proj): Conv2d(1280, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (attn_drop): Dropout(p=0.0, inplace=False)
                (output): Sequential(
                  (proj): Conv2d(1536, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (24): UniversalInvertedResidual(
              (dw_start): Identity()
              (pw_exp): ConvNormAct(
                (conv): Conv2d(1280, 2560, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): GELU(approximate='tanh')
                )
              )
              (dw_mid): Identity()
              (se): Identity()
              (pw_proj): ConvNormAct(
                (conv): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (dw_end): Identity()
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (25): MobileAttention(
              (norm): RmsNormAct2d(
                (drop): Identity()
                (act): Identity()
              )
              (attn): MultiQueryAttention2d(
                (query): Sequential(
                  (proj): Conv2d(1280, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (key): Sequential(
                  (proj): Conv2d(1280, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (value): Sequential(
                  (proj): Conv2d(1280, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (attn_drop): Dropout(p=0.0, inplace=False)
                (output): Sequential(
                  (proj): Conv2d(1536, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (26): UniversalInvertedResidual(
              (dw_start): Identity()
              (pw_exp): ConvNormAct(
                (conv): Conv2d(1280, 2560, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): GELU(approximate='tanh')
                )
              )
              (dw_mid): Identity()
              (se): Identity()
              (pw_proj): ConvNormAct(
                (conv): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (dw_end): Identity()
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (27): MobileAttention(
              (norm): RmsNormAct2d(
                (drop): Identity()
                (act): Identity()
              )
              (attn): MultiQueryAttention2d(
                (query): Sequential(
                  (proj): Conv2d(1280, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (key): Sequential(
                  (proj): Conv2d(1280, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (value): Sequential(
                  (proj): Conv2d(1280, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (attn_drop): Dropout(p=0.0, inplace=False)
                (output): Sequential(
                  (proj): Conv2d(1536, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (28): UniversalInvertedResidual(
              (dw_start): Identity()
              (pw_exp): ConvNormAct(
                (conv): Conv2d(1280, 2560, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): GELU(approximate='tanh')
                )
              )
              (dw_mid): Identity()
              (se): Identity()
              (pw_proj): ConvNormAct(
                (conv): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (dw_end): Identity()
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (29): MobileAttention(
              (norm): RmsNormAct2d(
                (drop): Identity()
                (act): Identity()
              )
              (attn): MultiQueryAttention2d(
                (query): Sequential(
                  (proj): Conv2d(1280, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (key): Sequential(
                  (proj): Conv2d(1280, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (value): Sequential(
                  (proj): Conv2d(1280, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (attn_drop): Dropout(p=0.0, inplace=False)
                (output): Sequential(
                  (proj): Conv2d(1536, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (30): UniversalInvertedResidual(
              (dw_start): Identity()
              (pw_exp): ConvNormAct(
                (conv): Conv2d(1280, 2560, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): GELU(approximate='tanh')
                )
              )
              (dw_mid): Identity()
              (se): Identity()
              (pw_proj): ConvNormAct(
                (conv): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (dw_end): Identity()
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (31): MobileAttention(
              (norm): RmsNormAct2d(
                (drop): Identity()
                (act): Identity()
              )
              (attn): MultiQueryAttention2d(
                (query): Sequential(
                  (proj): Conv2d(1280, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (key): Sequential(
                  (proj): Conv2d(1280, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (value): Sequential(
                  (proj): Conv2d(1280, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (attn_drop): Dropout(p=0.0, inplace=False)
                (output): Sequential(
                  (proj): Conv2d(1536, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (32): UniversalInvertedResidual(
              (dw_start): Identity()
              (pw_exp): ConvNormAct(
                (conv): Conv2d(1280, 2560, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): GELU(approximate='tanh')
                )
              )
              (dw_mid): Identity()
              (se): Identity()
              (pw_proj): ConvNormAct(
                (conv): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (dw_end): Identity()
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (33): MobileAttention(
              (norm): RmsNormAct2d(
                (drop): Identity()
                (act): Identity()
              )
              (attn): MultiQueryAttention2d(
                (query): Sequential(
                  (proj): Conv2d(1280, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (key): Sequential(
                  (proj): Conv2d(1280, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (value): Sequential(
                  (proj): Conv2d(1280, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (attn_drop): Dropout(p=0.0, inplace=False)
                (output): Sequential(
                  (proj): Conv2d(1536, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (34): UniversalInvertedResidual(
              (dw_start): Identity()
              (pw_exp): ConvNormAct(
                (conv): Conv2d(1280, 2560, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): GELU(approximate='tanh')
                )
              )
              (dw_mid): Identity()
              (se): Identity()
              (pw_proj): ConvNormAct(
                (conv): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (dw_end): Identity()
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (35): MobileAttention(
              (norm): RmsNormAct2d(
                (drop): Identity()
                (act): Identity()
              )
              (attn): MultiQueryAttention2d(
                (query): Sequential(
                  (proj): Conv2d(1280, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (key): Sequential(
                  (proj): Conv2d(1280, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (value): Sequential(
                  (proj): Conv2d(1280, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (attn_drop): Dropout(p=0.0, inplace=False)
                (output): Sequential(
                  (proj): Conv2d(1536, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (36): UniversalInvertedResidual(
              (dw_start): Identity()
              (pw_exp): ConvNormAct(
                (conv): Conv2d(1280, 2560, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): GELU(approximate='tanh')
                )
              )
              (dw_mid): Identity()
              (se): Identity()
              (pw_proj): ConvNormAct(
                (conv): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (dw_end): Identity()
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (37): MobileAttention(
              (norm): RmsNormAct2d(
                (drop): Identity()
                (act): Identity()
              )
              (attn): MultiQueryAttention2d(
                (query): Sequential(
                  (proj): Conv2d(1280, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (key): Sequential(
                  (proj): Conv2d(1280, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (value): Sequential(
                  (proj): Conv2d(1280, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
                )
                (attn_drop): Dropout(p=0.0, inplace=False)
                (output): Sequential(
                  (proj): Conv2d(1536, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (drop): Dropout(p=0.0, inplace=False)
                )
              )
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
            (38): UniversalInvertedResidual(
              (dw_start): Identity()
              (pw_exp): ConvNormAct(
                (conv): Conv2d(1280, 2560, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): GELU(approximate='tanh')
                )
              )
              (dw_mid): Identity()
              (se): Identity()
              (pw_proj): ConvNormAct(
                (conv): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): RmsNormAct2d(
                  (drop): Identity()
                  (act): Identity()
                )
              )
              (dw_end): Identity()
              (layer_scale): LayerScale2d()
              (drop_path): Identity()
            )
          )
        )
        (msfa): MobileNetV5MultiScaleFusionAdapter(
          (ffn): UniversalInvertedResidual(
            (dw_start): Identity()
            (pw_exp): ConvNormAct(
              (conv): Conv2d(1920, 3840, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): RmsNormAct2d(
                (drop): Identity()
                (act): GELU(approximate='tanh')
              )
            )
            (dw_mid): Identity()
            (se): Identity()
            (pw_proj): ConvNormAct(
              (conv): Conv2d(3840, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): RmsNormAct2d(
                (drop): Identity()
                (act): Identity()
              )
            )
            (dw_end): Identity()
            (layer_scale): Identity()
            (drop_path): Identity()
          )
          (norm): RmsNorm2d()
        )
      )
    )
    (language_model): Gemma3nTextModel(
      (embed_tokens): Gemma3nTextScaledWordEmbedding(262400, 2048, padding_idx=0)
      (layers): ModuleList(
        (0-29): 30 x Gemma3nTextDecoderLayer(
          (self_attn): Gemma3nTextAttention(
            (q_proj): Linear(in_features=2048, out_features=2048, bias=False)
            (k_proj): Linear(in_features=2048, out_features=512, bias=False)
            (v_proj): Linear(in_features=2048, out_features=512, bias=False)
            (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
            (q_norm): Gemma3nRMSNorm((256,), eps=1e-06)
            (k_norm): Gemma3nRMSNorm((256,), eps=1e-06)
            (v_norm): Gemma3nRMSNorm((), eps=1e-06)
          )
          (mlp): Gemma3nTextMLP(
            (gate_proj): Linear(in_features=2048, out_features=8192, bias=False)
            (up_proj): Linear(in_features=2048, out_features=8192, bias=False)
            (down_proj): Linear(in_features=8192, out_features=2048, bias=False)
            (act_fn): PytorchGELUTanh()
          )
          (input_layernorm): Gemma3nRMSNorm((2048,), eps=1e-06)
          (post_attention_layernorm): Gemma3nRMSNorm((2048,), eps=1e-06)
          (pre_feedforward_layernorm): Gemma3nRMSNorm((2048,), eps=1e-06)
          (post_feedforward_layernorm): Gemma3nRMSNorm((2048,), eps=1e-06)
          (act_fn): PytorchGELUTanh()
          (altup): Gemma3nTextAltUp(
            (correction_coefs): Linear(in_features=4, out_features=4, bias=False)
            (prediction_coefs): Linear(in_features=4, out_features=16, bias=False)
            (modality_router): Linear(in_features=2048, out_features=4, bias=False)
            (router_norm): Gemma3nRMSNorm((2048,), eps=1e-06)
          )
          (laurel): Gemma3nTextLaurelBlock(
            (linear_left): Linear(in_features=2048, out_features=64, bias=False)
            (linear_right): Linear(in_features=64, out_features=2048, bias=False)
            (post_laurel_norm): Gemma3nRMSNorm((2048,), eps=1e-06)
          )
          (per_layer_input_gate): Linear(in_features=2048, out_features=256, bias=False)
          (per_layer_projection): Linear(in_features=256, out_features=2048, bias=False)
          (post_per_layer_input_norm): Gemma3nRMSNorm((2048,), eps=1e-06)
        )
      )
      (norm): Gemma3nRMSNorm((2048,), eps=1e-06)
      (rotary_emb): Gemma3nTextRotaryEmbedding()
      (rotary_emb_local): Gemma3nTextRotaryEmbedding()
      (embed_tokens_per_layer): Gemma3nTextScaledWordEmbedding(262144, 7680, padding_idx=0)
      (per_layer_model_projection): Linear(in_features=2048, out_features=7680, bias=False)
      (per_layer_projection_norm): Gemma3nRMSNorm((256,), eps=1e-06)
      (altup_projections): ModuleList(
        (0-2): 3 x Linear(in_features=2048, out_features=2048, bias=False)
      )
      (altup_unembed_projections): ModuleList(
        (0-2): 3 x Linear(in_features=2048, out_features=2048, bias=False)
      )
    )
    (audio_tower): Gemma3nAudioEncoder(
      (subsample_conv_projection): Gemma3nAudioSubSampleConvProjection(
        (conv_0): Gemma3nAudioSSCPConvBlock(
          (conv): Conv2d(1, 128, kernel_size=(3, 3), stride=(2, 2), bias=False)
          (norm): Gemma3nAudioCumulativeGroupNorm()
          (activation): ReLU()
        )
        (conv_1): Gemma3nAudioSSCPConvBlock(
          (conv): Conv2d(128, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
          (norm): Gemma3nAudioCumulativeGroupNorm()
          (activation): ReLU()
        )
        (input_proj_linear): Linear(in_features=1024, out_features=1536, bias=False)
      )
      (conformer): ModuleList(
        (0-11): 12 x Gemma3nAudioConformerBlock(
          (ffw_layer_start): Gemma3nAudioConformerFeedForward(
            (pre_layer_norm): Gemma3nRMSNorm((1536,), eps=1e-06)
            (ffw_layer_1): Linear(in_features=1536, out_features=6144, bias=False)
            (ffw_layer_2): Linear(in_features=6144, out_features=1536, bias=False)
            (post_layer_norm): Gemma3nRMSNorm((1536,), eps=1e-06)
          )
          (attention): Gemma3nAudioConformerAttention(
            (pre_attn_norm): Gemma3nRMSNorm((1536,), eps=1e-06)
            (attn): Gemma3nAudioAttention(
              (relative_position_embedding): Gemma3nAudioRelativePositionEmbedding(
                (pos_proj): Linear(in_features=1536, out_features=1536, bias=False)
              )
              (q_proj): Linear(in_features=1536, out_features=1536, bias=False)
              (k_proj): Linear(in_features=1536, out_features=1536, bias=False)
              (v_proj): Linear(in_features=1536, out_features=1536, bias=False)
            )
            (post): Linear(in_features=1536, out_features=1536, bias=False)
            (post_norm): Gemma3nRMSNorm((1536,), eps=1e-06)
          )
          (lconv1d): Gemma3nAudioConformerLightConv1d(
            (pre_layer_norm): Gemma3nRMSNorm((1536,), eps=1e-06)
            (linear_start): Linear(in_features=1536, out_features=3072, bias=False)
            (depthwise_conv1d): Conv1d(1536, 1536, kernel_size=(5,), stride=(1,), groups=1536, bias=False)
            (conv_norm): Gemma3nRMSNorm((1536,), eps=1e-06)
            (linear_end): Linear(in_features=1536, out_features=1536, bias=False)
          )
          (ffw_layer_end): Gemma3nAudioConformerFeedForward(
            (pre_layer_norm): Gemma3nRMSNorm((1536,), eps=1e-06)
            (ffw_layer_1): Linear(in_features=1536, out_features=6144, bias=False)
            (ffw_layer_2): Linear(in_features=6144, out_features=1536, bias=False)
            (post_layer_norm): Gemma3nRMSNorm((1536,), eps=1e-06)
          )
          (norm): Gemma3nRMSNorm((1536,), eps=1e-06)
        )
      )
    )
    (embed_vision): Gemma3nMultimodalEmbedder(
      (embedding): Embedding(128, 2048)
      (hard_embedding_norm): Gemma3nRMSNorm((2048,), eps=1e-06)
      (soft_embedding_norm): Gemma3nRMSNorm((2048,), eps=1e-06)
      (embedding_projection): Linear(in_features=2048, out_features=2048, bias=False)
      (embedding_post_projection_norm): Gemma3nRMSNorm((), eps=1e-06)
    )
    (embed_audio): Gemma3nMultimodalEmbedder(
      (embedding): Embedding(128, 1536)
      (hard_embedding_norm): Gemma3nRMSNorm((1536,), eps=1e-06)
      (soft_embedding_norm): Gemma3nRMSNorm((1536,), eps=1e-06)
      (embedding_projection): Linear(in_features=1536, out_features=2048, bias=False)
      (embedding_post_projection_norm): Gemma3nRMSNorm((), eps=1e-06)
    )
  )
  (lm_head): Linear(in_features=2048, out_features=262400, bias=False)
)