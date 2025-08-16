"""Video Swin model preset configurations."""

backbone_presets = {
    "videoswin_tiny": {
        "metadata": {
            "description": (
                "Tiny Video Swin Transformer backbone. "
                "Untrained version without pretrained weights."
            ),
            "params": 27_850_470,
            "official_name": "VideoSwinT",
            "path": "video_swin",
        },
        "kaggle_handle": "kaggle://keras/video_swin/keras/videoswin_tiny/1",
    },
    "videoswin_small": {
        "metadata": {
            "description": (
                "Small Video Swin Transformer backbone. "
                "Untrained version without pretrained weights."
            ),
            "params": 49_509_078,
            "official_name": "VideoSwinS",
            "path": "video_swin",
        },
        "kaggle_handle": "kaggle://keras/video_swin/keras/videoswin_small/1",
    },
    "videoswin_base": {
        "metadata": {
            "description": (
                "Base Video Swin Transformer backbone. "
                "Untrained version without pretrained weights."
            ),
            "params": 87_638_984,
            "official_name": "VideoSwinB",
            "path": "video_swin",
        },
        "kaggle_handle": "kaggle://keras/video_swin/keras/videoswin_base/1",
    },
    "videoswin_tiny_kinetics400": {
        "metadata": {
            "description": (
                "Tiny Video Swin Transformer backbone. "
                "Pretrained on ImageNet-1K and trained on Kinetics-400."
            ),
            "params": 27_850_470,
            "official_name": "VideoSwinT",
            "path": "video_swin",
        },
        "kaggle_handle": "kaggle://keras/video_swin/keras/videoswin_tiny_kinetics400/1",
    },
    "videoswin_small_kinetics400": {
        "metadata": {
            "description": (
                "Small Video Swin Transformer backbone. "
                "Pretrained on ImageNet-1K and trained on Kinetics-400. "
                "Achieves 80.6% top-1 and 94.5% top-5 accuracy on Kinetics-400."
            ),
            "params": 49_509_078,
            "official_name": "VideoSwinS",
            "path": "video_swin",
        },
        "kaggle_handle": "kaggle://keras/video_swin/keras/videoswin_small_kinetics400/1",
    },
    "videoswin_base_kinetics400": {
        "metadata": {
            "description": (
                "Base Video Swin Transformer backbone. "
                "Pretrained on ImageNet-1K and trained on Kinetics-400. "
                "Achieves 80.6% top-1 and 94.6% top-5 accuracy on Kinetics-400."
            ),
            "params": 87_638_984,
            "official_name": "VideoSwinB",
            "path": "video_swin",
        },
        "kaggle_handle": "kaggle://keras/video_swin/keras/videoswin_base_kinetics400/1",
    },
    "videoswin_base_kinetics400_imagenet22k": {
        "metadata": {
            "description": (
                "Base Video Swin Transformer backbone. "
                "Pretrained on ImageNet-22K and trained on Kinetics-400. "
                "Achieves 82.7% top-1 and 95.5% top-5 accuracy on Kinetics-400."
            ),
            "params": 87_638_984,
            "official_name": "VideoSwinB",
            "path": "video_swin",
        },
        "kaggle_handle": "kaggle://keras/video_swin/keras/videoswin_base_kinetics400_imagenet22k/1",
    },
    "videoswin_base_kinetics600_imagenet22k": {
        "metadata": {
            "description": (
                "Base Video Swin Transformer backbone. "
                "Pretrained on ImageNet-22K and trained on Kinetics-600. "
                "Achieves 84.0% top-1 and 96.5% top-5 accuracy on Kinetics-600."
            ),
            "params": 87_638_984,
            "official_name": "VideoSwinB",
            "path": "video_swin",
        },
        "kaggle_handle": "kaggle://keras/video_swin/keras/videoswin_base_kinetics600_imagenet22k/1",
    },
    "videoswin_base_something_something_v2": {
        "metadata": {
            "description": (
                "Base Video Swin Transformer backbone. "
                "Pretrained on Kinetics-400 and trained on Something-Something V2. "
                "Achieves 69.6% top-1 and 92.7% top-5 accuracy."
            ),
            "params": 87_638_984,
            "official_name": "VideoSwinB",
            "path": "video_swin",
        },
        "kaggle_handle": "kaggle://keras/video_swin/keras/videoswin_base_something_something_v2/1",
    },
}