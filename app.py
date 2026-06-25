"""music-mood-recs Streamlit app — track select -> mood prediction -> top-5 similar.

Loads the trained MoodCNN, precomputed mel-spectrograms, and embeddings.
Reuses review-sentiment's st.cache_resource pattern for model loading.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import streamlit as st
import torch
from sklearn.metrics.pairwise import cosine_similarity

from src.models.cnn import CNNConfig, MoodCNN
from src.recommend.similar import extract_embeddings, top_k_similar

MODEL_DIR = os.environ.get("MMR_MODEL_DIR", "models/cnn")
MELSPEC_DIR = os.environ.get("MMR_MELSPEC_DIR", "artifacts/melspecs")
MANIFEST_CSV = os.environ.get("MMR_MANIFEST", "artifacts/melspec_manifest.csv")
META_CSV = os.environ.get("MMR_META", "artifacts/subset_meta.csv")


@st.cache_resource(max_entries=1)
def load_model_artifacts(model_dir: str):
    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
        cfg_dict = json.load(f)
    with open(os.path.join(model_dir, "tags.json"), encoding="utf-8") as f:
        tags = json.load(f)
    cfg = CNNConfig(
        n_mels=cfg_dict["n_mels"],
        n_classes=cfg_dict["n_classes"],
        embedding_dim=cfg_dict["embedding_dim"],
    )
    model = MoodCNN(cfg)
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu"))
    model.eval()
    return model, tags, cfg


@st.cache_data
def load_manifest_and_meta(manifest_csv: str, meta_csv: str):
    manifest = pd.read_csv(manifest_csv)
    meta = pd.read_csv(meta_csv).set_index("TRACK_ID")
    return manifest, meta


@st.cache_data
def load_all_melspecs(manifest: pd.DataFrame):
    mels = []
    track_ids = []
    for _, row in manifest.iterrows():
        mel = np.load(row["npy_path"]).astype(np.float32)
        mels.append(mel)
        track_ids.append(row["TRACK_ID"])
    return np.stack(mels), track_ids


@st.cache_resource(max_entries=1)
def compute_all_embeddings(model: MoodCNN, mels: np.ndarray):
    return extract_embeddings(model, mels, batch_size=32, device="cpu")


def _track_display(track_id: str, meta: pd.DataFrame, tags: list[str]) -> str:
    if track_id not in meta.index:
        return track_id
    row = meta.loc[track_id]
    active = [t for t in tags if int(row.get(f"tag_{t}", 0)) == 1]
    return f"{track_id}  [{', '.join(active) if active else '-'}]"


st.set_page_config(page_title="music-mood-recs", page_icon="🎵", layout="wide")

with st.sidebar:
    st.title("🎵 음악 무드 분류 + 추천")
    st.caption("MTG-Jamendo 무드/테마 서브셋 · CNN · 임베딩 코사인 유사도")

try:
    model, tags, cfg = load_model_artifacts(MODEL_DIR)
    manifest, meta = load_manifest_and_meta(MANIFEST_CSV, META_CSV)
    mels, track_ids = load_all_melspecs(manifest)
    embeddings = compute_all_embeddings(model, mels)
except Exception as exc:
    st.error(f"모델/데이터를 불러오지 못했습니다: {exc}")
    st.info("학습을 먼저 실행해주세요: python scripts/train_cnn.py")
    st.stop()

st.success(f"모델 로드 완료 — {len(track_ids)}곡, 태그: {', '.join(tags)}")

tab_predict, tab_about = st.tabs(["🔍 무드 예측 + 추천", "ℹ️ 프로젝트 소개"])

with tab_predict:
    col_in, col_out = st.columns([1, 2])

    with col_in:
        display_options = [_track_display(tid, meta, tags) for tid in track_ids]
        selected = st.selectbox("곡 선택", range(len(display_options)), format_func=lambda i: display_options[i])
        st.caption(f"트랙 ID: {track_ids[selected]}")
        predict_clicked = st.button("예측 + 추천", use_container_width=True)

    if predict_clicked:
        mel = mels[selected]
        x = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
            probs = torch.sigmoid(logits)[0].numpy()

        with col_out:
            st.subheader("예측 무드")
            prob_df = pd.DataFrame({"무드": tags, "확률": probs}).sort_values("확률", ascending=True)
            st.bar_chart(prob_df.set_index("무드"))
            top_mood = tags[int(probs.argmax())]
            st.metric("최상위 무드", top_mood, f"{probs.max():.1%}")

            st.divider()
            st.subheader("비슷한 무드의 곡 Top-5")
            idxs, sims = top_k_similar(selected, embeddings, k=5)
            rec_rows = []
            for i, sim in zip(idxs, sims):
                tid = track_ids[i]
                rec_rows.append(
                    {
                        "곡": _track_display(tid, meta, tags),
                        "코사인 유사도": round(float(sim), 4),
                    }
                )
            st.table(pd.DataFrame(rec_rows))

with tab_about:
    st.subheader("음악 오디오 무드 분류 + 콘텐츠 기반 추천")
    st.markdown(
        "MTG-Jamendo 무드/테마 서브셋의 멜스펙트로그램으로 CNN 무드 분류 모델을 학습하고, "
        "분류 과정에서 학습된 임베딩을 코사인 유사도로 재사용해 비슷한 무드의 곡을 추천합니다. "
        "분류와 추천을 별도 파이프라인으로 이어붙이지 않고 하나의 모델로 증명하는 DL 포트폴리오 프로젝트입니다."
    )
    stat_cols = st.columns(3)
    stat_cols[0].metric("데이터", "MTG-Jamendo")
    stat_cols[1].metric("모델", "MoodCNN")
    stat_cols[2].metric("추천", "임베딩 코사인 유사도")
