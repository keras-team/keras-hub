# Metadata for loading pretrained model weights.
backbone_presets = {
    "dfine_nano_coco": {
        "metadata": {
            "description": (
                "D-FINE Nano model, the smallest variant in the family, "
                "pretrained on the COCO dataset. Ideal for applications "
                "where computational resources are limited."
            ),
            "params": 3788625,
            "path": "d_fine",
        },
        "kaggle_handle": "kaggle://keras/d-fine/keras/dfine_nano_coco/1",
    },
    "dfine_small_coco": {
        "metadata": {
            "description": (
                "D-FINE Small model pretrained on the COCO dataset. Offers a "
                "balance between performance and computational efficiency."
            ),
            "params": 10329321,
            "path": "d_fine",
        },
        "kaggle_handle": "kaggle://keras/d-fine/keras/dfine_small_coco/1",
    },
    "dfine_medium_coco": {
        "metadata": {
            "description": (
                "D-FINE Medium model pretrained on the COCO dataset. A solid "
                "baseline with strong performance for general-purpose "
                "object detection."
            ),
            "params": 19621160,
            "path": "d_fine",
        },
        "kaggle_handle": "kaggle://keras/d-fine/keras/dfine_medium_coco/1",
    },
    "dfine_large_coco": {
        "metadata": {
            "description": (
                "D-FINE Large model pretrained on the COCO dataset. Provides "
                "high accuracy and is suitable for more demanding tasks."
            ),
            "params": 31344064,
            "path": "d_fine",
        },
        "kaggle_handle": "kaggle://keras/d-fine/keras/dfine_large_coco/1",
    },
    "dfine_xlarge_coco": {
        "metadata": {
            "description": (
                "D-FINE X-Large model, the largest COCO-pretrained variant, "
                "designed for state-of-the-art performance where accuracy "
                "is the top priority."
            ),
            "params": 62834048,
            "path": "d_fine",
        },
        "kaggle_handle": "kaggle://keras/d-fine/keras/dfine_xlarge_coco/1",
    },
    "dfine_small_obj365": {
        "metadata": {
            "description": (
                "D-FINE Small model pretrained on the large-scale Objects365 "
                "dataset, enhancing its ability to recognize a wider "
                "variety of objects."
            ),
            "params": 10623329,
            "path": "d_fine",
        },
        "kaggle_handle": "kaggle://keras/d-fine/keras/dfine_small_obj365/1",
    },
    "dfine_medium_obj365": {
        "metadata": {
            "description": (
                "D-FINE Medium model pretrained on the Objects365 dataset. "
                "Benefits from a larger and more diverse pretraining corpus."
            ),
            "params": 19988670,
            "path": "d_fine",
        },
        "kaggle_handle": "kaggle://keras/d-fine/keras/dfine_medium_obj365/1",
    },
    "dfine_large_obj365": {
        "metadata": {
            "description": (
                "D-FINE Large model pretrained on the Objects365 dataset for "
                "improved generalization and performance on diverse object "
                "categories."
            ),
            "params": 31858578,
            "path": "d_fine",
        },
        "kaggle_handle": "kaggle://keras/d-fine/keras/dfine_large_obj365/1",
    },
    "dfine_xlarge_obj365": {
        "metadata": {
            "description": (
                "D-FINE X-Large model pretrained on the Objects365 dataset, "
                "offering maximum performance by leveraging a vast number "
                "of object categories during pretraining."
            ),
            "params": 63348562,
            "path": "d_fine",
        },
        "kaggle_handle": "kaggle://keras/d-fine/keras/dfine_xlarge_obj365/1",
    },
    "dfine_small_obj2coco": {
        "metadata": {
            "description": (
                "D-FINE Small model first pretrained on Objects365 and then "
                "fine-tuned on COCO, combining broad feature learning with "
                "benchmark-specific adaptation."
            ),
            "params": 10329321,
            "path": "d_fine",
        },
        "kaggle_handle": "kaggle://keras/d-fine/keras/dfine_small_obj2coco/1",
    },
    "dfine_medium_obj2coco": {
        "metadata": {
            "description": (
                "D-FINE Medium model using a two-stage training process: "
                "pretraining on Objects365 followed by fine-tuning on COCO."
            ),
            "params": 19621160,
            "path": "d_fine",
        },
        "kaggle_handle": "kaggle://keras/d-fine/keras/dfine_medium_obj2coco/1",
    },
    "dfine_large_obj2coco_e25": {
        "metadata": {
            "description": (
                "D-FINE Large model pretrained on Objects365 and then "
                "fine-tuned on COCO for 25 epochs. A high-performance model "
                "with specialized tuning."
            ),
            "params": 31344064,
            "path": "d_fine",
        },
        "kaggle_handle": "kaggle://keras/d-fine/keras/dfine_large_obj2coco_e25/1",
    },
    "dfine_xlarge_obj2coco": {
        "metadata": {
            "description": (
                "D-FINE X-Large model, pretrained on Objects365 and fine-tuned "
                "on COCO, representing the most powerful model in this "
                "series for COCO-style tasks."
            ),
            "params": 62834048,
            "path": "d_fine",
        },
        "kaggle_handle": "kaggle://keras/d-fine/keras/dfine_xlarge_obj2coco/1",
    },
}
