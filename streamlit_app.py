import csv
import io
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from anomalib.data.datamodules import Folder
from anomalib.data.utils import ValSplitMode
from anomalib.engine.engine import Engine
from PIL import Image
from streamlit.runtime.uploaded_file_manager import UploadedFile

import constants
import models

st.set_page_config(
    page_title="異常検知",
    page_icon=":control_knobs:",
    layout="wide",
)

train_images_placeholder = st.empty()
result_images_placeholder = st.empty()
download_button_placeholder = st.empty()


def configure_sidebar() -> bool:
    """
    Configures the Streamlit sidebar UI for anomaly detection settings.

    The sidebar includes:
        - File uploaders for training and test images.
        - Selectboxes for choosing the anomaly detection model and its backbone.
        - Checkbox and number input for threshold configuration (with auto/manual option).
        - Number inputs for image resizing and training epochs.
        - Submit button to start the inspection process.
        - Buttons and dialogs providing information about available models and backbones.

    Returns:
        bool: True if the form is submitted ("検査開始" button pressed), otherwise False.
    """
    with st.sidebar:
        st.markdown("# 異常検知")
        st.markdown("## :level_slider: 検査条件")

        train_images = st.file_uploader(
            "学習画像",
            type=["png", "jpg", "jpeg", "bmp"],
            accept_multiple_files=True,
            key="train_images",
            help="学習に使用する正常画像を選択してください。",
        )

        test_images = st.file_uploader(
            "検査画像",
            type=["png", "jpg", "jpeg", "bmp"],
            accept_multiple_files=True,
            key="test_images",
            help="検査する画像を選択してください。",
        )

        col_model_name_1, col_model_name_2 = st.columns(
            [9, 1], vertical_alignment="bottom"
        )

        with col_model_name_1:
            model_name = st.selectbox(
                "検査手法",
                options=constants.MODEL_NAMES,
                index=12,
                key="model_name",
            )

        @st.dialog("検査手法について", width="large")
        def about_model_name():
            df = pd.DataFrame(constants.ABOUT_MODEL_NAMES)
            st.table(df)

        with col_model_name_2:
            if st.button("?", key="button_about_model_name"):
                about_model_name()

        col_backbone_1, col_backbone_2 = st.columns(
            [9, 1], vertical_alignment="bottom"
        )
        with col_backbone_1:
            backbone = st.selectbox(
                "モデル",
                options=constants.MODEL_BACKBONES[model_name],
                index=0,
                key="backbone",
            )

        @st.dialog("モデルについて", width="large")
        def about_backbone():
            df = pd.DataFrame(constants.ABOUT_BACKBONE)
            st.table(df)

        with col_backbone_2:
            if st.button("?", key="button_about_backbone"):
                about_backbone()

        threshold_auto = st.checkbox(
            "自動しきい値", key="threshold_auto", value=True
        )
        threshold = st.number_input(
            "しきい値", format="%0.5f", key="threshold", disabled=threshold_auto
        )

        image_size = st.number_input(
            "縮小サイズ", value=128, min_value=2, key="image_size"
        )
        epochs = st.number_input("学習回数", value=1, min_value=1, key="epochs")

        submitted = st.button(
            "検査開始", type="primary", width="content", key="submitted"
        )

    return submitted


def disp_train_images(images: list[UploadedFile]) -> None:
    """Display the training images in a grid.

    Args:
        images (list[UploadedFile]): List of uploaded training images.
    """
    if images:
        with train_images_placeholder.container(height=300):
            st.header("学習画像")
            # Create a grid of images
            cols = st.columns(3)
            for i, image in enumerate(images):
                cols[i % 3].image(
                    image,
                    caption=f"{image.name}",
                    width="stretch",
                )


def predict_to_image(pred_image: torch.Tensor):
    """
    Converts a PyTorch tensor representing an image prediction to a NumPy uint8 image array.
    The function permutes the tensor dimensions from (C, H, W) to (W, H, C), clips the values to the [0, 1] range,
    scales them to [0, 255], and converts the result to uint8 format suitable for image display or saving.
    Args:
        pred_image (torch.Tensor): The predicted image tensor with shape (C, H, W).
    Returns:
        np.ndarray: The processed image as a NumPy array with dtype uint8 and shape (W, H, C).
    """
    return (pred_image.permute(1, 2, 0).numpy().clip(0.0, 1.0) * 255).astype(
        np.uint8
    )


def disp_result_images(predictions, threshold) -> None:
    """
    Displays the result images, anomaly maps, and prediction scores in a Streamlit app, and provides a downloadable ZIP file containing the results.

    Args:
        predictions (list): A list of prediction objects, each containing image data, anomaly maps, prediction scores, and image paths.
        threshold (float): The threshold value used to determine if a prediction is normal or anomalous.

    Functionality:
        - Displays original images, their corresponding anomaly heatmaps, and prediction results in a 3-column grid layout.
        - Generates a CSV file summarizing the results (file name, score, and judgment).
        - Packages the heatmap images and CSV file into a ZIP archive.
        - Provides a download button for the ZIP file in the Streamlit interface.

    Notes:
        - Assumes the existence of several helper functions and variables (e.g., get_map_min_max, get_item, predict_to_image, superimpose_anomaly_map_g, result_images_placeholder, constants.RESULT_PATH).
        - Handles both torch.Tensor and non-tensor image types.
        - Skips predictions with missing or invalid data.
    """
    if predictions is None:
        return

    st.session_state["test_pil_images"] = []
    st.session_state["heat_maps"] = []
    st.session_state["str_results"] = []
    st.session_state["str_threshold"] = 0.0

    map_min, map_max, map_ptp = get_map_min_max(predictions)
    with result_images_placeholder.container(height=300):
        st.header("検査結果")
        str_threshold = f"しきい値: {threshold:.5f}"
        st.info(str_threshold)
        st.session_state["str_threshold"] = str_threshold

        # Create a grid of images
        cols = st.columns(3)

        # CSV用のメモリバッファを用意
        csv_buffer = io.StringIO()
        csv_writer = csv.writer(csv_buffer)
        csv_writer.writerow(
            [
                "検査手法",
                st.session_state["model_name"],
                "モデル",
                st.session_state["backbone"],
                "しきい値",
                threshold,
            ]
        )
        csv_writer.writerow(["ファイル名", "スコア", "判定"])

        # 画像とcsvファイルをzipに
        zip_path = Path(constants.RESULT_PATH) / "result.zip"
        with zipfile.ZipFile(zip_path, "w") as zipf:
            for i, prediction in enumerate(predictions):
                tile_0 = cols[0].container(height=200, border=False)
                image = get_item(prediction, "image")
                if isinstance(image, torch.Tensor):
                    np_image = predict_to_image(image)
                else:
                    continue

                pil_image = Image.fromarray(np_image)
                wh_ratio = st.session_state["test_wh_ratios"][i]
                resized_image = pil_image.resize(
                    (pil_image.width, int(pil_image.width / wh_ratio))
                )
                with tile_0:
                    st.image(resized_image, width="stretch")
                st.session_state["test_pil_images"].append(resized_image)

                image_path = Path(prediction.image_path[0]).name

                tile_1 = cols[1].container(height=200, border=False)
                anomaly_map = get_item(prediction, "anomaly_map")
                if anomaly_map is not None:
                    anomaly_map = anomaly_map.cpu().numpy().squeeze()  # type: ignore
                    heat_map = superimpose_anomaly_map_g(
                        anomaly_map=anomaly_map,
                        image=np_image,
                        map_min=map_min,
                        map_ptp=map_ptp,
                    )
                    pil_heat_map = Image.fromarray(heat_map)
                    resized_heat_map = pil_heat_map.resize(
                        (pil_heat_map.width, int(pil_heat_map.width / wh_ratio))
                    )
                    with tile_1:
                        st.image(resized_heat_map, width="stretch")
                    st.session_state["heat_maps"].append(resized_heat_map)

                    # zipに書き込み
                    # メモリ上に画像を保存
                    img_bytes = io.BytesIO()
                    resized_heat_map.save(
                        img_bytes,
                        format="JPEG",
                    )
                    img_bytes.seek(0)

                    # ZIPに画像を書き込み
                    zipf.writestr(
                        "result_" + Path(image_path).stem + ".jpg",
                        img_bytes.read(),
                    )

                tile_2 = cols[2].container(height=200, border=False)
                tile_2.write(f"{image_path}")
                pred_score = get_item(prediction, "pred_score")
                pred_score = pred_score if pred_score is not None else 0.0
                with tile_2:
                    if pred_score <= threshold:
                        judge = "正常"
                        st.success(f"score:{pred_score:.2f} [正常]")
                    else:
                        judge = "異常"
                        st.error(f"score:{pred_score:.2f} [異常]")
                st.session_state["str_results"].append(
                    f"score:{pred_score:.2f} [{judge}]"
                )

                # CSVにファイル名を追加
                csv_writer.writerow([image_path, pred_score, judge])

            # CSVをZIPに追加
            csv_buffer.seek(0)
            zipf.writestr(
                "result.csv", csv_buffer.getvalue().encode("utf-8-sig")
            )

    with download_button_placeholder.container():
        # 保存ボタン
        st.download_button(
            "結果保存",
            data=zip_path.read_bytes(),
            file_name="result.zip",
            on_click="ignore",
        )


def disp_session_images():
    """
    Displays training and test images, heat maps, and inspection results from the Streamlit session state.

    This function performs the following:
    - Displays training images using `disp_train_images`.
    - Retrieves test images, heat maps, result strings, and threshold information from the Streamlit session state.
    - Shows the inspection threshold information.
    - Arranges test images, their corresponding heat maps, and result messages in a 3-column grid layout.
    - For each test image:
        - Displays the original image.
        - Displays the corresponding heat map.
        - Shows the inspection result as a success or error message, depending on the result content.

    Assumes the following keys exist in `st.session_state`:
        - "train_images": List of training images.
        - "test_pil_images": List of test images (PIL format).
        - "heat_maps": List of heat map images.
        - "str_results": List of result strings for each test image.
        - "str_threshold": String describing the threshold used for inspection.

    Requires a global `result_images_placeholder` for displaying the results section.
    """
    disp_train_images(st.session_state["train_images"])

    test_pil_images = st.session_state["test_pil_images"]
    heat_maps = st.session_state["heat_maps"]
    str_results = st.session_state["str_results"]
    str_threshold = st.session_state["str_threshold"]

    with result_images_placeholder.container(height=300):
        st.header("検査結果")
        st.info(str_threshold)
        # Create a grid of images
        cols = st.columns(3)
        for image, heat_map, result in zip(
            test_pil_images, heat_maps, str_results
        ):
            tile_0 = cols[0].container(height=200, border=False)
            with tile_0:
                st.image(image, width="stretch")
            tile_1 = cols[1].container(height=200, border=False)
            with tile_1:
                st.image(heat_map, width="stretch")
            tile_2 = cols[2].container(height=200, border=False)
            with tile_2:
                if "正常" in result:
                    st.success(result)
                else:
                    st.error(result)

    # 保存ボタン
    zip_path = Path(constants.RESULT_PATH) / "result.zip"
    with download_button_placeholder.container():
        st.download_button(
            "結果保存",
            data=zip_path.read_bytes(),
            file_name="result.zip",
            on_click="ignore",
        )


def save_images(train_images, test_images):
    """Delete existing dataset and save images to train/test directories.

    Args:
        train_images: List of uploaded training images
        test_images: List of uploaded test images
    """

    # 1. Delete DATASET_PATH images
    dataset_path = Path(constants.DATASET_PATH)
    if dataset_path.exists():
        shutil.rmtree(dataset_path)
    result_path = Path(constants.RESULT_PATH)
    if result_path.exists():
        shutil.rmtree(result_path)

    # 2. Save train_images to DATASET_PATH/train
    train_dir = dataset_path / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    for i, image in enumerate(train_images):
        image_name = "_".join(Path(image.name).parts)
        with open(train_dir / f"{i:04}.{image_name}", "wb") as f:
            f.write(image.getvalue())

    # 3. Save test_images to DATASET_PATH/test
    test_dir = dataset_path / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    # 画像縦横比
    st.session_state.test_wh_ratios = []
    for i, image in enumerate(test_images):
        image_name = "_".join(Path(image.name).parts)
        with open(test_dir / f"{i:04}.{image_name}", "wb") as f:
            f.write(image.getvalue())

        pil_image = Image.open(image)
        arr = np.array(pil_image)
        h, w = arr.shape[:2]
        st.session_state.test_wh_ratios.append(w / h)


def get_item(prediction, key):
    """
    Retrieve the value associated with a given key from a prediction object, handling various data types.
    Parameters:
        prediction (object): The object containing the attribute to retrieve.
        key (str): The attribute name to access within the prediction object.
    Returns:
        The first element or scalar value associated with the specified key, depending on the type:
            - For lists or tuples: returns the first element if available, else None.
            - For torch.Tensor: returns None if empty, the scalar value if single element, or the first element otherwise.
            - For numpy.ndarray: returns None if empty, the scalar value if single element, or the first element otherwise.
            - For other types (int, float, etc.): returns the value directly.
        Returns None if the attribute does not exist.
    """
    if not hasattr(prediction, key):
        return None

    val = getattr(prediction, key)

    # list or tuple
    if isinstance(val, (list, tuple)):
        return val[0] if len(val) > 0 else None

    # PyTorch Tensor
    if isinstance(val, torch.Tensor):
        if val.numel() == 0:
            return None
        if val.numel() == 1:
            return val.item()
        return val[0] if val.dim() > 0 else val.item()

    # NumPy ndarray
    if isinstance(val, np.ndarray):
        if val.size == 0:
            return None
        if val.size == 1:
            return val.item()
        return val.flat[0]  # flat iteratorで最初の要素

    # それ以外（int, float, etc.）
    return val


def get_map_min_max(predictions) -> tuple[float, float, float]:
    """
    Calculates the minimum, maximum, and range (peak-to-peak) values across anomaly maps in a list of predictions.
    Args:
        predictions (Iterable): A list or iterable of prediction objects, each containing an 'anomaly_map' attribute.
            Each 'anomaly_map' is expected to be a tensor-like object where the first element (index 0) can be
            converted to a NumPy array via `.cpu().numpy()`.
    Returns:
        tuple[float, float, float]: A tuple containing:
            - map_min (float): The minimum value found across all anomaly maps.
            - map_max (float): The maximum value found across all anomaly maps.
            - map_ptp (float): The range (max - min) of the anomaly map values.
    """
    map_min = min(
        prediction.anomaly_map[0].min().cpu().numpy()
        for prediction in predictions
    )
    map_max = max(
        prediction.anomaly_map[0].max().cpu().numpy()
        for prediction in predictions
    )
    map_ptp = map_max - map_min
    return map_min, map_max, map_ptp


def superimpose_anomaly_map_g(
    anomaly_map: np.ndarray,
    image: np.ndarray,
    map_min: float,
    map_ptp: float,
    alpha: float = 0.4,
    gamma: int = 0,
) -> np.ndarray:
    """Superimpose anomaly map on image.

    Args:
        anomaly_map (np.ndarray): Anomaly map.
        image (np.ndarray): Image.
        alpha (float): Alpha value for superimposition.
        gamma (int): Gamma value for superimposition.
        map_min (float): Minimum value of the anomaly map.
        map_ptp (float): Range of the anomaly map.

    Returns:
        np.ndarray: Superimposed image.
    """
    assert anomaly_map.shape == image.shape[:2], (
        f"Anomaly map shape {anomaly_map.shape} does not match image shape "
        f"{image.shape[:2]}."
    )
    nomalized_map = (((anomaly_map - map_min) / map_ptp) * 255).astype(np.uint8)
    color_map = cv2.applyColorMap(nomalized_map, cv2.COLORMAP_JET)
    rgb_color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)
    height, width = rgb_color_map.shape[:2]
    image = cv2.resize(image, (width, height))
    return cv2.addWeighted(rgb_color_map, alpha, image, (1 - alpha), gamma)


def main_page(submitted: bool) -> None:
    """Main page layout and logic for generating images.

    Args:
        submitted (bool): Flag indicating whether the form has been submitted.
    """

    if submitted == True:
        with st.status("処理中...", expanded=True) as status:

            if (
                st.session_state["train_images"] is None
                or len(st.session_state["train_images"]) == 0
            ):
                status.update(
                    label="学習画像を選択してください。",
                    state="error",
                    expanded=False,
                )
                return
            if (
                st.session_state["test_images"] is None
                or len(st.session_state["test_images"]) == 0
            ):
                status.update(
                    label="検査画像を選択してください。",
                    state="error",
                    expanded=False,
                )
                return
            try:
                # Load the selected model
                model = models.get_model(
                    model_name=st.session_state["model_name"],
                    backbone=st.session_state["backbone"],
                    image_size=st.session_state["image_size"],
                )

                # アップロードしたファイルをフォルダに保存
                save_images(
                    st.session_state["train_images"],
                    st.session_state["test_images"],
                )

                datamodule = Folder(
                    name="custom",
                    root=constants.DATASET_PATH,
                    normal_dir="train",
                    normal_test_dir="test",
                    val_split_mode=ValSplitMode.FROM_TRAIN,
                    # 検証データを入れるとスコアが1か0になるため、検証はしない
                    # ratioを0にするとデフォルト値で分割されるため、非常に小さい値を設定
                    val_split_ratio=0.0001,
                    train_batch_size=constants.BATCH_SIZE,
                    eval_batch_size=constants.BATCH_SIZE,
                    num_workers=1,
                )
                datamodule.setup()

                # ----- 学習 -----
                engine = Engine(
                    # callbacks=callbacks,
                    max_epochs=st.session_state["epochs"],
                    accelerator="auto",
                    devices=1,
                )
                engine.fit(
                    model=model,
                    datamodule=datamodule,
                )

                # しきい値
                if st.session_state["threshold_auto"]:
                    try:
                        # trainデータのスコア
                        train_predictions = engine.predict(
                            model=model,
                            dataloaders=datamodule.train_dataloader(),
                        )
                        train_scores = [
                            get_item(prediction, "pred_score")
                            for prediction in (train_predictions or [])
                        ]
                        # 閾値（99.7％）
                        threshold = np.mean(train_scores) + 3 * np.std(train_scores)  # type: ignore
                        # # 四分位範囲×1.5の場合
                        # threshold = np.percentile(train_scores, 75) + 1.5 * (
                        #     np.percentile(train_scores, 75) - np.percentile(train_scores, 25)
                        # )
                    except Exception as e:
                        print(f"exception: {e}")
                        threshold = 0
                    print(f"threshold: {threshold}")
                else:
                    threshold = st.session_state["threshold"]

                # 予想
                predictions = engine.predict(model=model, datamodule=datamodule)

                # 結果描画
                disp_train_images(st.session_state["train_images"])
                disp_result_images(predictions, threshold=threshold)

                status.update(
                    label="処理完了",
                    state="complete",
                    expanded=False,
                )
            except Exception as e:
                print(e)
                st.error(f"Encountered an error: {e}", icon="❌")

    elif "test_pil_images" in st.session_state and 0 < len(
        st.session_state["test_pil_images"]
    ):
        # 処理済み
        disp_session_images()
    else:
        pass


def main():
    """
    Main function to run the Streamlit application.

    This function initializes the sidebar configuration and the main page layout.
    It retrieves the user inputs from the sidebar, and passes them to the main page function.
    The main page function then generates images based on these inputs.
    """
    submitted = configure_sidebar()
    main_page(submitted)


if __name__ == "__main__":
    main()
