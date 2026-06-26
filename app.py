"""music-mood-recs Streamlit app — track select -> mood prediction -> top-5 similar.

Loads the trained MoodCNN, precomputed mel-spectrograms, and embeddings.
Reuses review-sentiment's st.cache_resource pattern for model loading.
"""

from __future__ import annotations

import glob
import json
import os
import tempfile
from dataclasses import dataclass
from typing import Optional

import librosa
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity

# >>> AUTO-SYNCED from src/models/cnn.py (run scripts/sync_standalone_app.py) >>>
@dataclass(frozen=True)
class CNNConfig:
    n_mels: int = 128
    n_classes: int = 5
    embedding_dim: int = 64
    conv_channels: tuple[int, int, int] = (16, 32, 64)
    kernel_size: int = 3
    dropout: float = 0.3


class MoodCNN(nn.Module):
    def __init__(self, cfg: CNNConfig | None = None):
        super().__init__()
        self.cfg = cfg or CNNConfig()
        c1, c2, c3 = self.cfg.conv_channels

        self.features = nn.Sequential(
            nn.Conv2d(1, c1, self.cfg.kernel_size, padding=self.cfg.kernel_size // 2),
            nn.BatchNorm2d(c1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(c1, c2, self.cfg.kernel_size, padding=self.cfg.kernel_size // 2),
            nn.BatchNorm2d(c2),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(c2, c3, self.cfg.kernel_size, padding=self.cfg.kernel_size // 2),
            nn.BatchNorm2d(c3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embed_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3, self.cfg.embedding_dim),
            nn.ReLU(),
            nn.Dropout(self.cfg.dropout),
        )
        self.classifier = nn.Linear(self.cfg.embedding_dim, self.cfg.n_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Return the embedding vector (batch, embedding_dim) used for recommendation."""
        h = self.features(x)
        h = self.pool(h)
        return self.embed_head(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.embed(x)
        return self.classifier(z)
# <<< AUTO-SYNCED <<<


# >>> AUTO-SYNCED from src/preprocessing/melspec.py (run scripts/sync_standalone_app.py) >>>
@dataclass(frozen=True)
class MelspecConfig:
    sr: int = 22050
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    segment_seconds: float = 30.0
    offset_mode: str = "start"  # "start" or "center"
    top_db: float = 80.0

    @property
    def segment_samples(self) -> int:
        return int(self.sr * self.segment_seconds)

    @property
    def expected_frames(self) -> int:
        # librosa default center=True pads n_fft//2 on each side, so
        # n_frames = 1 + n_samples // hop_length.
        return 1 + self.segment_samples // self.hop_length


def load_segment(
    audio_path: str, cfg: MelspecConfig, duration_cap: Optional[float] = None
) -> np.ndarray:
    """Load a fixed 30-second mono segment as a 1D float32 numpy array."""
    total_duration = duration_cap
    if total_duration is None:
        try:
            total_duration = float(librosa.get_duration(path=audio_path))
        except Exception:
            total_duration = cfg.segment_seconds
    if cfg.offset_mode == "center" and total_duration > cfg.segment_seconds:
        offset = (total_duration - cfg.segment_seconds) / 2.0
    else:
        offset = 0.0
    y, _ = librosa.load(
        audio_path,
        sr=cfg.sr,
        mono=True,
        offset=offset,
        duration=cfg.segment_seconds,
    )
    target = cfg.segment_samples
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)), mode="constant")
    elif len(y) > target:
        y = y[:target]
    return y.astype(np.float32)


def compute_melspec(y: np.ndarray, cfg: MelspecConfig) -> np.ndarray:
    """Compute a log-mel spectrogram (n_mels, n_frames) from a 1D waveform."""
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=cfg.sr,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max, top_db=cfg.top_db)
    return log_mel.astype(np.float32)


def extract_melspec(
    audio_path: str, cfg: Optional[MelspecConfig] = None
) -> np.ndarray:
    """Load segment + compute log-mel spectrogram in one call."""
    cfg = cfg or MelspecConfig()
    y = load_segment(audio_path, cfg)
    return compute_melspec(y, cfg)
# <<< AUTO-SYNCED <<<


# >>> AUTO-SYNCED from src/recommend/similar.py (run scripts/sync_standalone_app.py) >>>
@torch.no_grad()
def extract_embeddings(
    model: MoodCNN,
    mels: np.ndarray,
    batch_size: int = 32,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """Compute embeddings for an array of mel-spectrograms.

    ``mels``: (N, n_mels, n_frames) float32. Returns (N, embedding_dim) float32.
    """
    model.eval()
    model.to(device)
    out: list[np.ndarray] = []
    for i in range(0, len(mels), batch_size):
        batch = mels[i : i + batch_size]
        x = torch.from_numpy(batch).unsqueeze(1).to(device)
        z = model.embed(x)
        out.append(z.cpu().numpy())
    return np.concatenate(out, axis=0)


def top_k_similar(
    query_idx: int,
    embeddings: np.ndarray,
    k: int = 5,
    exclude_query: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices, similarities) of the top-k most similar tracks.

    Cosine similarity is computed between the query embedding and all others.
    If ``exclude_query`` is True, the query itself is removed from the results.
    """
    query = embeddings[query_idx : query_idx + 1]
    sims = cosine_similarity(query, embeddings).ravel()
    order = np.argsort(-sims)
    if exclude_query:
        order = order[order != query_idx]
    top = order[:k]
    return top, sims[top]


def top_k_similar_to_vector(
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
    k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Same ranking as ``top_k_similar`` but for a query embedding that is not
    already a row of ``embeddings`` (e.g. a user-uploaded track)."""
    query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
    sims = cosine_similarity(query, embeddings).ravel()
    order = np.argsort(-sims)
    top = order[:k]
    return top, sims[top]


@torch.no_grad()
def predict_mood_probs(model: MoodCNN, embeddings: np.ndarray) -> np.ndarray:
    """Run the trained classifier head on precomputed embeddings to get each
    track's mood probabilities without re-loading any mel-spectrogram.

    ``model.embed`` already produced ``embeddings``, so this only replays the
    final linear layer (``model.classifier``) — cheap enough to run over the
    whole library at app start-up.
    """
    model.eval()
    z = torch.from_numpy(np.asarray(embeddings, dtype=np.float32))
    logits = model.classifier(z)
    return torch.sigmoid(logits).numpy()


MOOD_KEYWORDS: dict[str, list[str]] = {
    "happy": ["행복", "기쁘", "기뻐", "즐겁", "신나", "설레", "웃음", "웃기", "들뜨", "유쾌"],
    "energetic": ["에너지", "파워", "운동", "헬스", "달리", "파이팅", "격렬", "흥분", "활기", "신남"],
    "relaxing": ["편안", "휴식", "잠", "졸려", "힐링", "차분", "여유", "평화", "느긋", "안정"],
    "film": ["영화", "웅장", "드라마틱", "서사", "영상", "스토리", "장엄", "긴장감", "몰입"],
    "dark": ["우울", "슬픔", "슬프", "어둡", "외롭", "쓸쓸", "그리움", "눈물", "힘들", "지치", "불안"],
}


def infer_mood_from_text(text: str, tags: list[str]) -> tuple[str | None, dict[str, int]]:
    """Score ``text`` against each tag's keyword list and return the
    best-matching tag (``None`` if no keyword hits at all) plus the raw
    per-tag hit counts, so the caller can show why a tag was picked.
    """
    counts = {tag: 0 for tag in tags}
    for tag in tags:
        for kw in MOOD_KEYWORDS.get(tag, []):
            if kw in text:
                counts[tag] += 1
    best_tag = max(counts, key=counts.get) if counts else None
    if best_tag is not None and counts[best_tag] == 0:
        best_tag = None
    return best_tag, counts
# <<< AUTO-SYNCED <<<


# >>> AUTO-SYNCED from src/evaluation/metrics.py (run scripts/sync_standalone_app.py) >>>
def build_comparison_table(results: dict[str, dict]) -> pd.DataFrame:
    """results = {model_display_name: {"accuracy":..., "f1_micro":..., "f1_macro":..., "roc_auc":...}}."""
    if not results:
        return pd.DataFrame(columns=["Accuracy", "F1(micro)", "F1(macro)", "ROC-AUC"])

    df = pd.DataFrame(results).T
    df = df.rename(
        columns={
            "accuracy": "Accuracy",
            "f1_micro": "F1(micro)",
            "f1_macro": "F1(macro)",
            "roc_auc": "ROC-AUC",
        }
    )
    return df[["Accuracy", "F1(micro)", "F1(macro)", "ROC-AUC"]]


def load_all_metrics(models_dir: str = "models") -> dict[str, dict]:
    """Scans models/*/metrics.json. Prefers the held-out "test" entry (final
    generalization check) when present; otherwise falls back to the last training
    epoch's val metrics."""
    results: dict[str, dict] = {}
    for metrics_path in sorted(glob.glob(os.path.join(models_dir, "*", "metrics.json"))):
        with open(metrics_path, encoding="utf-8") as f:
            data = json.load(f)
        model_dir_name = os.path.basename(os.path.dirname(metrics_path))
        display_name = data.get("display_name", model_dir_name)
        history = data.get("history")
        source = data.get("test") or (history[-1] if history else data)
        results[display_name] = {
            "accuracy": source.get("accuracy"),
            "f1_micro": source.get("f1_micro"),
            "f1_macro": source.get("f1_macro"),
            "roc_auc": source.get("roc_auc"),
        }
    return results
# <<< AUTO-SYNCED <<<


MODEL_DIR = os.environ.get("MMR_MODEL_DIR", "models/cnn")
AUDIO_DIR = os.environ.get("MMR_AUDIO_DIR", "data/audio")
MELSPEC_DIR = os.environ.get("MMR_MELSPEC_DIR", "artifacts/melspecs")
MANIFEST_CSV = os.environ.get("MMR_MANIFEST", "artifacts/melspec_manifest.csv")
META_CSV = os.environ.get("MMR_META", "artifacts/subset_meta.csv")
EMBEDDINGS_NPY = os.environ.get("MMR_EMBEDDINGS", "artifacts/embeddings.npy")


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
def load_embeddings(embeddings_npy: str, manifest: pd.DataFrame) -> np.ndarray:
    """Loads precomputed embeddings (scripts/precompute_embeddings.py), aligned
    1:1 with ``manifest`` row order.

    Streamlit Community Cloud only guarantees 1GB RAM: stacking every
    mel-spectrogram (~1.4GB for 2,247 tracks) to run a forward pass at startup
    would OOM-kill the app, so that forward pass happens offline instead and
    only this small embeddings array is loaded here.
    """
    embeddings = np.load(embeddings_npy)
    if len(embeddings) != len(manifest):
        raise ValueError(
            f"embeddings ({len(embeddings)}) and manifest ({len(manifest)}) row counts differ — "
            "re-run scripts/precompute_embeddings.py"
        )
    return embeddings


@st.cache_data(max_entries=8)
def load_mel(npy_path: str) -> np.ndarray:
    """Lazily loads a single track's mel-spectrogram (only the selected track,
    never the full dataset — see load_embeddings for why)."""
    return np.load(npy_path).astype(np.float32)


def _track_display(track_id: str, meta: pd.DataFrame, tags: list[str]) -> str:
    if track_id not in meta.index:
        return track_id
    row = meta.loc[track_id]
    active = [t for t in tags if int(row.get(f"tag_{t}", 0)) == 1]
    return f"{track_id}  [{', '.join(active) if active else '-'}]"


def _audio_path(track_id: str, manifest: pd.DataFrame) -> str | None:
    rows = manifest.loc[manifest["TRACK_ID"] == track_id, "PATH"]
    if rows.empty:
        return None
    path = os.path.join(AUDIO_DIR, rows.iloc[0])
    return path if os.path.exists(path) else None


_MOOD_EMOJI = {
    "happy": "😊",
    "energetic": "⚡",
    "relaxing": "🌿",
    "film": "🎬",
    "dark": "🌑",
}


st.set_page_config(page_title="music-mood-recs", page_icon="🎵", layout="wide")

with st.sidebar:
    st.title("🎵 음악 무드 분류 + 추천")
    st.caption("MTG-Jamendo 무드/테마 서브셋 · CNN · 임베딩 코사인 유사도")

try:
    model, tags, cfg = load_model_artifacts(MODEL_DIR)
    manifest, meta = load_manifest_and_meta(MANIFEST_CSV, META_CSV)
    track_ids = manifest["TRACK_ID"].tolist()
    embeddings = load_embeddings(EMBEDDINGS_NPY, manifest)
except Exception as exc:
    st.error(f"모델/데이터를 불러오지 못했습니다: {exc}")
    st.info(
        "데모 실행 순서:\n"
        "1. `python scripts/download_audio.py --top-n 5 --max-tars 30`\n"
        "2. `python scripts/extract_melspecs.py`\n"
        "3. `python scripts/train_cnn.py`\n"
        "4. `python -m scripts.precompute_embeddings`\n"
        "5. `streamlit run app.py`\n\n"
        "또는 환경변수로 경로 지정: `MMR_MODEL_DIR`, `MMR_MANIFEST`, `MMR_META`, `MMR_EMBEDDINGS`"
    )
    st.stop()

with st.sidebar:
    st.divider()
    st.markdown(
        f"**데이터**: {len(track_ids)}곡\n\n"
        f"**태그**: {', '.join(tags)}\n\n"
        f"**모델**: MoodCNN ({sum(p.numel() for p in model.parameters()):,} params)"
    )

st.success(f"모델 로드 완료 — {len(track_ids)}곡, 태그: {', '.join(tags)}")

def _render_recommendations(idxs: np.ndarray, sims: np.ndarray, score_label: str = "코사인 유사도") -> None:
    for i, sim in zip(idxs, sims):
        tid = track_ids[i]
        with st.container(border=True):
            rec_col_info, rec_col_audio = st.columns([2, 1])
            rec_col_info.markdown(f"**{_track_display(tid, meta, tags)}**")
            rec_col_info.caption(f"{score_label} {sim:.4f}")
            rec_audio = _audio_path(tid, manifest)
            if rec_audio:
                rec_col_audio.audio(rec_audio)


def _render_mood_probs(probs: np.ndarray) -> str:
    top_mood = tags[int(probs.argmax())]
    st.metric("최상위 무드", f"{_MOOD_EMOJI.get(top_mood, '')} {top_mood}", f"{probs.max():.1%}")
    for tag, prob in sorted(zip(tags, probs), key=lambda t: -t[1]):
        st.progress(float(prob), text=f"{_MOOD_EMOJI.get(tag, '')} {tag} {prob:.0%}")
    return top_mood


tab_predict, tab_upload, tab_text, tab_compare, tab_eda, tab_about = st.tabs(
    [
        "🔍 라이브러리 곡 예측",
        "🎤 오디오 업로드",
        "💬 텍스트로 찾기",
        "📊 모델 성능",
        "📈 데이터 탐색(EDA)",
        "ℹ️ 프로젝트 소개",
    ]
)

with tab_predict:
    col_in, col_out = st.columns([1, 2])

    with col_in:
        display_options = [_track_display(tid, meta, tags) for tid in track_ids]
        selected = st.selectbox("곡 선택", range(len(display_options)), format_func=lambda i: display_options[i])
        st.caption(f"트랙 ID: {track_ids[selected]}")

        selected_audio = _audio_path(track_ids[selected], manifest)
        if selected_audio:
            st.audio(selected_audio)
        else:
            st.caption("🔇 오디오 파일을 찾을 수 없습니다.")

        predict_clicked = st.button("예측 + 추천", use_container_width=True)

    if predict_clicked:
        mel = load_mel(manifest.iloc[selected]["npy_path"])
        x = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
            probs = torch.sigmoid(logits)[0].numpy()

        with col_out:
            st.subheader("예측 무드")
            _render_mood_probs(probs)

            st.divider()
            st.subheader("비슷한 무드의 곡 Top-5")
            idxs, sims = top_k_similar(selected, embeddings, k=5)
            _render_recommendations(idxs, sims)

with tab_upload:
    st.caption("내 컴퓨터에 있는 오디오 파일을 직접 올려서 무드를 예측하고, 라이브러리에서 비슷한 곡 5개를 추천받습니다.")
    uploaded = st.file_uploader("오디오 파일 업로드 (mp3/wav/ogg/flac)", type=["mp3", "wav", "ogg", "flac", "m4a"])

    if uploaded is not None:
        st.audio(uploaded)
        with st.spinner("멜스펙트로그램 추출 + 무드 예측 중..."):
            suffix = os.path.splitext(uploaded.name)[1] or ".mp3"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name
            try:
                mel = extract_melspec(tmp_path, MelspecConfig(n_mels=cfg.n_mels))
                x = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    z = model.embed(x)
                    probs = torch.sigmoid(model.classifier(z))[0].numpy()
                query_embedding = z[0].numpy()
            finally:
                os.remove(tmp_path)

        col_in, col_out = st.columns([1, 2])
        with col_in:
            st.subheader("예측 무드")
            _render_mood_probs(probs)
        with col_out:
            st.subheader("비슷한 무드의 곡 Top-5")
            idxs, sims = top_k_similar_to_vector(query_embedding, embeddings, k=5)
            _render_recommendations(idxs, sims)

with tab_text:
    st.caption("지금 기분이나 원하는 분위기를 문장으로 입력하면, 가장 가까운 무드를 추정해 그 무드에 맞는 곡을 추천합니다.")
    text_input = st.text_input("지금 기분이 어떤가요?", placeholder="예: 오늘 너무 우울하고 힘들어서 위로받을 음악 듣고 싶어")
    text_clicked = st.button("무드 찾기", use_container_width=True)

    if text_clicked:
        if not text_input.strip():
            st.warning("문장을 입력해주세요.")
        else:
            best_tag, keyword_hits = infer_mood_from_text(text_input, tags)
            if best_tag is None:
                best_tag = tags[0]

            st.success(f"추정된 무드: {_MOOD_EMOJI.get(best_tag, '')} **{best_tag}** (키워드 매칭 {keyword_hits[best_tag]}건)")
            st.caption(f"태그별 키워드 매칭 수: {keyword_hits}")

            track_probs = predict_mood_probs(model, embeddings)
            tag_idx = tags.index(best_tag)
            order = np.argsort(-track_probs[:, tag_idx])[:5]
            sims = track_probs[order, tag_idx]
            st.subheader(f"'{best_tag}' 무드에 가장 잘 맞는 곡 Top-5")
            _render_recommendations(order, sims, score_label=f"{best_tag} 확률")

with tab_compare:
    all_metrics = load_all_metrics()
    comparison_df = build_comparison_table(all_metrics)
    if comparison_df.empty:
        st.info("아직 학습된 모델 성능 데이터가 없습니다.")
    else:
        metric_cols = st.columns(len(comparison_df))
        for col, (model_display_name, row) in zip(metric_cols, comparison_df.iterrows()):
            col.metric(model_display_name, f"{row['F1(micro)']:.1%}", help="F1(micro), held-out test 기준")

        st.caption("test split(held-out)으로 평가한 최종 일반화 성능. test가 없는 모델은 마지막 epoch 검증 성능으로 대체.")
        st.divider()
        st.dataframe(comparison_df, use_container_width=True)
        st.bar_chart(comparison_df[["Accuracy", "F1(micro)", "F1(macro)"]], stack=False)

with tab_eda:
    tag_counts = {t: int(meta[f"tag_{t}"].sum()) for t in tags}
    sum_cols = st.columns(len(tags) + 1)
    sum_cols[0].metric("전체 트랙", f"{len(meta):,}곡")
    for col, (tag, count) in zip(sum_cols[1:], tag_counts.items()):
        col.metric(f"{_MOOD_EMOJI.get(tag, '')} {tag}", f"{count:,}곡")

    st.divider()
    st.markdown("**무드 태그 분포** — 곡당 다중 태그 가능")
    tag_df = pd.DataFrame({"곡 수": tag_counts})
    st.bar_chart(tag_df)

    st.divider()
    split_counts = meta["split"].value_counts()
    st.markdown("**train/validation/test 분할**")
    st.bar_chart(split_counts)

    st.divider()
    st.markdown("**곡 길이(초) 분포**")
    st.bar_chart(pd.cut(meta["DURATION"], bins=10).value_counts().sort_index().rename(lambda i: str(i)))

with tab_about:
    st.subheader("음악 오디오 무드 분류 + 콘텐츠 기반 추천")
    st.markdown(
        "MTG-Jamendo 무드/테마 서브셋의 멜스펙트로그램으로 CNN 무드 분류 모델을 학습하고, "
        "분류 과정에서 학습된 임베딩을 코사인 유사도로 재사용해 비슷한 무드의 곡을 추천합니다. "
        "분류와 추천을 별도 파이프라인으로 이어붙이지 않고 하나의 모델로 증명하는 DL 포트폴리오 프로젝트입니다.\n\n"
        "라이브러리에 있는 곡을 고르는 것뿐 아니라, **직접 가진 오디오 파일을 업로드**해 동일한 모델로 무드를 예측하거나, "
        "**지금 기분을 문장으로 입력**해 그 무드에 맞는 곡을 찾을 수도 있습니다(🎤 오디오 업로드 / 💬 텍스트로 찾기 탭)."
    )
    stat_cols = st.columns(3)
    stat_cols[0].metric("데이터", "MTG-Jamendo")
    stat_cols[1].metric("모델", "MoodCNN")
    stat_cols[2].metric("추천", "임베딩 코사인 유사도")
