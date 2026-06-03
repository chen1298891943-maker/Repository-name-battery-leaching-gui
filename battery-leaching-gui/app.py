# -*- coding: utf-8 -*-
"""Streamlit GUI for CatBoost prediction of Li/Mn/Ni/Co leaching efficiencies."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "saved_catboost_gui_models"
CONFIG_PATH = MODEL_DIR / "feature_config.json"
METRICS_PATH = MODEL_DIR / "CatBoost_4metals_metrics.csv"

FEATURE_ORDER = [
    "RA-C", "RA-H", "RA-N", "RA-O", "RA-S",
    "Li in Feed", "Ni in Feed", "Co in Feed", "Mn in Feed",
    "R-Temp(°C)", "R-Time (min)", "R-Atmos", "Mass Ratio",
    "pKa", "LC (mol/L)", "S/L (g/L)", "L-Time(min)", "L-Temp (°C)",
]

DISPLAY_NAME = {
    "RA-C": "RA-C",
    "RA-H": "RA-H",
    "RA-N": "RA-N",
    "RA-O": "RA-O",
    "RA-S": "RA-S",
    "Li in Feed": "Li in Feed",
    "Ni in Feed": "Ni in Feed",
    "Co in Feed": "Co in Feed",
    "Mn in Feed": "Mn in Feed",
    "R-Temp(°C)": "R-Temp(°C)",
    "R-Time (min)": "R-Time (min)",
    "R-Atmos": "R-Atmos",
    "Mass Ratio": "Mass Ratio",
    "pKa": "pKa",
    "LC (mol/L)": "LC (mol/L)",
    "S/L (g/L)": "S/L (g/L)",
    "L-Time(min)": "L-Time(min)",
    "L-Temp (°C)": "L-Temp (°C)",
}

METAL_ORDER = ["Li-Leaching", "Mn-Leaching", "Ni-Leaching", "Co-Leaching"]

MANUAL_NUMBER_INPUT_FEATURES = {
    "RA-C", "RA-H", "RA-N", "RA-O", "RA-S",
    "Li in Feed", "Ni in Feed", "Co in Feed", "Mn in Feed",
}


def load_config_and_models():
    if not CONFIG_PATH.exists():
        st.error(
            "未找到模型配置文件。请先在项目文件夹中运行：\n\n"
            "python train_catboost_gui_models.py"
        )
        st.stop()

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    models = {}
    for target, model_file in config["target_models"].items():
        path = MODEL_DIR / model_file
        if not path.exists():
            st.error(f"未找到模型文件：{path}")
            st.stop()
        models[target] = joblib.load(path)
    return config, models


def numeric_input_widget(feature, feature_info):
    min_values = feature_info.get("numeric_min", {})
    max_values = feature_info.get("numeric_max", {})
    defaults = feature_info.get("numeric_defaults", {})

    v_min = float(min_values.get(feature, 0.0))
    v_max = float(max_values.get(feature, 100.0))
    default = float(defaults.get(feature, (v_min + v_max) / 2))

    if v_min == v_max:
        v_min = min(0.0, v_min)
        v_max = max(100.0, v_max + 1.0)

    # Use a reasonable step according to range.
    span = abs(v_max - v_min)
    step = 0.01 if span <= 10 else 0.1

    # For raw material composition and reducing-agent composition,
    # use manual number input instead of sliders.
    if feature in MANUAL_NUMBER_INPUT_FEATURES:
        return st.sidebar.number_input(
            DISPLAY_NAME.get(feature, feature),
            value=float(default),
            step=float(step),
            format="%.4f" if step == 0.01 else "%.2f",
            help=(
                f"请手动输入数值。训练数据范围约为 {v_min:.4g}–{v_max:.4g}，"
                "建议不要明显超出训练数据范围。"
            ),
        )

    # Add a small margin for user extrapolation.
    lower = min(v_min, default)
    upper = max(v_max, default)

    return st.sidebar.slider(
        DISPLAY_NAME.get(feature, feature),
        min_value=float(lower),
        max_value=float(upper),
        value=float(default),
        step=float(step),
        format="%.4f" if step == 0.01 else "%.2f",
    )


def atmosphere_widget(feature_info):
    """R-Atmos fixed numeric coding.

    The model receives integer values 1/2/3/4, while the GUI displays
    human-readable atmosphere names:
    1 = Nitrogen, 2 = Argon, 3 = Vacuum, 4 = CO.
    """
    numeric_defaults = feature_info.get("numeric_defaults", {})

    atmos_map = {
        1: "1 - 氮气 N₂",
        2: "2 - 氩气 Ar",
        3: "3 - 真空 Vacuum",
        4: "4 - CO",
    }
    options = [1, 2, 3, 4]

    default = int(round(float(numeric_defaults.get("R-Atmos", 1))))
    if default not in options:
        default = 1

    return st.sidebar.selectbox(
        "R-Atmos",
        options=options,
        index=options.index(default),
        format_func=lambda x: atmos_map.get(x, str(x)),
        help="R-Atmos 编码：1=氮气，2=氩气，3=真空，4=CO。",
    )


def main():
    st.set_page_config(
        page_title="金属浸出率预测系统",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .main .block-container {padding-top: 2.2rem; max-width: 1180px;}
        div[data-testid="stMetric"] {background-color: #f7f9fc; border: 1px solid #e5e7eb; border-radius: 10px; padding: 16px;}
        .small-note {color: #6b7280; font-size: 0.92rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    config, models = load_config_and_models()
    feature_info = config["feature_info"]

    st.sidebar.title("使用说明")
    st.sidebar.write("1. 输入原料组成和焙烧剂组成。")
    st.sidebar.write("2. 调节焙烧与浸出工艺参数。")
    st.sidebar.write("3. 点击 Predict，输出 Li、Mn、Ni、Co 浸出率。")
    st.sidebar.divider()
    st.sidebar.subheader("输入参数")

    input_data = {}
    for feature in FEATURE_ORDER:
        if feature == "R-Atmos":
            input_data[feature] = atmosphere_widget(feature_info)
        else:
            input_data[feature] = numeric_input_widget(feature, feature_info)

    st.title("Predict Li-Leaching & Mn-Leaching & Ni-Leaching & Co-Leaching")
    st.caption("Final model: CatBoost. Inputs are selected core raw-material, reducing-agent, roasting and leaching parameters.")

    st.subheader("Current input data")
    input_df = pd.DataFrame([input_data], columns=FEATURE_ORDER)
    st.dataframe(input_df, use_container_width=True, hide_index=True)

    predict_clicked = st.button("Predict", type="primary")

    if predict_clicked:
        results = []
        for target in METAL_ORDER:
            if target not in models:
                continue
            model = models[target]
            # The models were trained with standardized GUI_FEATURE columns.
            pred = float(model.predict(input_df)[0])
            # Leaching efficiency normally ranges from 0 to 100; clipping avoids impossible GUI outputs.
            pred_clip = max(0.0, min(100.0, pred))
            results.append({
                "Target": target,
                "Predicted value (%)": round(pred_clip, 3),
                "Raw model output (%)": round(pred, 3),
            })

        result_df = pd.DataFrame(results)
        st.subheader("Prediction results")
        cols = st.columns(len(result_df))
        for col, (_, row) in zip(cols, result_df.iterrows()):
            metal = row["Target"].replace("-Leaching", "")
            col.metric(label=f"{metal}-Leaching", value=f"{row['Predicted value (%)']:.2f}%")
        st.dataframe(result_df, use_container_width=True, hide_index=True)

    if METRICS_PATH.exists():
        with st.expander("CatBoost model metrics", expanded=False):
            metrics_df = pd.read_csv(METRICS_PATH)
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
