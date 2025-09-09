#!/usr/bin/env python3
import timm
from anomalib.models import (
    Cfa,
    Cflow,
    Csflow,
    Dfkde,
    Dfm,
    Draem,
    Dsr,
    EfficientAd,
    Fastflow,
    Fre,
    Ganomaly,
    Padim,
    Patchcore,
    ReverseDistillation,
    Stfpm,
    Supersimplenet,
    Uflow,
    VlmAd,
    WinClip,
)
from anomalib.models.components import AnomalibModule
from anomalib.models.components.classification import FeatureScalingMethod
from anomalib.models.image.efficient_ad.torch_model import EfficientAdModelSize
from anomalib.models.image.reverse_distillation.anomaly_map import (
    AnomalyMapGenerationMode,
)
from anomalib.models.image.vlm_ad.utils import ModelName
from anomalib.pre_processing import PreProcessor
from torchvision.transforms.v2 import Compose, Resize


def get_model(
    model_name: str,
    backbone: str,
    image_size: int = 256,
) -> AnomalibModule:
    """
    Creates and returns an anomaly detection model based on the specified model name and backbone.
    Args:
        model_name (str): The name of the anomaly detection model to instantiate. Supported values include
            "CFA", "C-Flow", "CS-Flow", "DFKDE", "DFM", "DRAEM", "DSR", "Efficient AD", "FastFlow", "FRE",
            "GANomaly", "PaDiM", "PatchCore", "Reverse Distillation", "STFPM", "SuperSimpleNet", "U-Flow",
            "VLM-AD", "WinCLIP".
        backbone (str): The name of the backbone neural network architecture to use for feature extraction.
        image_size (int, optional): The size to which input images will be resized. Defaults to 256.
    Returns:
        AnomalibModule: An instance of the specified anomaly detection model, configured with the given backbone
        and preprocessing settings.
    Raises:
        ValueError: If an unknown model name is provided.
    """

    if backbone in ["", "mcait"]:
        layers = []
    else:
        feature_model = timm.create_model(backbone, features_only=True)
        layers = feature_model.feature_info.module_name()  # type: ignore
        print(layers)

    # 前処理の設定
    transform = Compose([Resize((image_size, image_size))])
    pre_processor = PreProcessor(transform=transform)

    # ----- モデル構築 -----
    if model_name == "CFA":
        model = Cfa(
            backbone=backbone,
            pre_processor=pre_processor,
            visualizer=False,
            gamma_c=1,
            gamma_d=1,
            num_nearest_neighbors=3,
            num_hard_negative_features=3,
            radius=1e-5,
        )
        # max_epochs = 30
        # callbacks = [EarlyStopping(patience=5, monitor="pixel_AUROC", mode="max")]
    elif model_name == "C-Flow":
        model = Cflow(
            backbone=backbone,
            pre_processor=pre_processor,
            layers=layers,
            pre_trained=True,
            fiber_batch_size=64,
            decoder="freia-cflow",
            condition_vector=128,
            coupling_blocks=8,
            clamp_alpha=1.9,
            permute_soft=False,
            lr=0.0001,
        )
        # max_epochs = 50
        # callbacks = [EarlyStopping(patience=5, monitor="pixel_AUROC", mode="max")]
    elif model_name == "CS-Flow":
        model = Csflow(
            pre_processor=pre_processor,
            cross_conv_hidden_channels=1024,
            n_coupling_blocks=4,
            clamp=3,
            num_channels=3,
        )
        # max_epochs = 240
    elif model_name == "DFKDE":
        model = Dfkde(
            backbone=backbone,
            layers=[layers[-1]],
            # pre_processor = pre_processor,
            pre_trained=True,
            # n_pca_components=16,
            n_pca_components=1,
            feature_scaling_method=FeatureScalingMethod.SCALE,
            max_training_points=40000,
        )
        # max_epochs = 1
        # callbacks = [EarlyStopping(monitor="pixel_AUROC", mode="max")]
    elif model_name == "DFM":
        model = Dfm(
            backbone=backbone,
            layer=layers[-2],
            pre_processor=pre_processor,
            pre_trained=True,
            pooling_kernel_size=4,
            pca_level=0.97,
            score_type="fre",
        )
        # max_epochs = 1
    elif model_name == "DRAEM":
        model = Draem(
            pre_processor=pre_processor,
            enable_sspcab=False,
            sspcab_lambda=0.1,
            anomaly_source_path=None,
            beta=(0.1, 1.0),
        )
        # max_epochs = 700
        # callbacks = [EarlyStopping(patience=20, monitor="pixel_AUROC", mode="max")]
    elif model_name == "DSR":
        model = Dsr(
            pre_processor=pre_processor,
            latent_anomaly_strength=0.2,
            upsampling_train_ratio=0.7,
        )
        # max_epochs = 700
    elif model_name == "Efficient AD":
        model = EfficientAd(
            pre_processor=pre_processor,
            imagenet_dir="./datasets/imagenette",
            teacher_out_channels=384,
            model_size=EfficientAdModelSize.S,
            lr=0.0001,
            weight_decay=0.00001,
            padding=False,
            pad_maps=True,
        )
        # max_epochs = 1000
    elif model_name == "FastFlow":
        model = Fastflow(
            backbone=backbone,
            pre_processor=pre_processor,
            pre_trained=True,
            flow_steps=8,
            conv3x3_only=False,
            hidden_ratio=1.0,
        )
        # max_epochs = 500
        # callbacks = [EarlyStopping(monitor="pixel_AUROC", mode="max")]
    elif model_name == "FRE":
        model = Fre(
            backbone=backbone,
            layer=layers[-2],
            pre_processor=pre_processor,
            input_dim=image_size * image_size,
            latent_dim=image_size,
            pre_trained=True,
            pooling_kernel_size=2,
        )
        # max_epochs = 1220
    elif model_name == "GANomaly":
        model = Ganomaly(
            pre_processor=pre_processor,
            batch_size=32,
            n_features=64,
            latent_vec_size=100,
            extra_layers=0,
            add_final_conv_layer=True,
            wadv=1,
            wcon=50,
            wenc=1,
            lr=0.0002,
            beta1=0.5,
            beta2=0.999,
        )
        # max_epochs = 100
        # callbacks = [EarlyStopping(monitor="image_AUROC", mode="max")]
    elif model_name == "PaDiM":
        model = Padim(
            backbone=backbone,
            layers=layers[-4:-1],
            pre_processor=pre_processor,
            n_features=100,
            pre_trained=True,
        )
        # max_epochs = 1
    elif model_name == "PatchCore":
        model = Patchcore(
            backbone=backbone,
            layers=layers[-3:-1],
            pre_processor=pre_processor,
            pre_trained=True,
            coreset_sampling_ratio=0.1,
            num_neighbors=9,
        )
        # max_epochs = 1
    elif model_name == "Reverse Distillation":
        model = ReverseDistillation(
            backbone=backbone,
            layers=layers[-4:-1],
            pre_processor=pre_processor,
            anomaly_map_mode=AnomalyMapGenerationMode.ADD,
            pre_trained=True,
        )
        # max_epochs = 200
        # callbacks = [EarlyStopping(monitor="pixel_AUROC", mode="max")]
    elif model_name == "STFPM":
        model = Stfpm(
            backbone=backbone,
            layers=layers[-4:-1],
            pre_processor=pre_processor,
        )
        # max_epochs = 100
        # callbacks = [EarlyStopping(patience=5, monitor="pixel_AUROC", mode="max")]
    elif model_name == "SuperSimpleNet":
        model = Supersimplenet(
            backbone=backbone,
            layers=layers[-3:-1],
            pre_processor=pre_processor,
            perlin_threshold=0.2,
            supervised=False,
        )
        # max_epochs = 1
    elif model_name == "U-Flow":
        model = Uflow(
            backbone=backbone,
            # pre_processor=pre_processor,
            flow_steps=4,
            affine_clamp=2.0,
            affine_subnet_channels_ratio=1.0,
            permute_soft=False,
        )
        # max_epochs = 200
        # callbacks = [EarlyStopping(patience=20, monitor="pixel_AUROC", mode="max")]
    elif model_name == "VLM-AD":
        model = VlmAd(
            model=ModelName.LLAMA_OLLAMA,
            api_key=None,
            k_shot=0,
        )
        # max_epochs = 1
    elif model_name == "WinCLIP":
        model = WinClip(
            # pre_processor=pre_processor,
            class_name="transistor",
            k_shot=0,
            scales=(2, 3),
            few_shot_source=None,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model
