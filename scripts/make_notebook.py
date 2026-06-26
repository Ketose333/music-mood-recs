"""Generate submission/music_mood_recs.ipynb.

The notebook must not import the src package (mirrors the app.py constraint —
see scripts/sync_standalone_app.py) so it stays a single self-contained
deliverable. inline_module() pulls each src/ file's actual source (minus its
docstring and `from __future__` import) into the relevant cell at generation
time, so src/ stays the single source of truth and nothing is hand-duplicated.

Usage:
  python scripts/make_notebook.py
"""

import ast

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


def inline_module(path: str) -> str:
    """Return a module's body (functions/classes/constants/imports), with its
    docstring and `from __future__` import stripped, for embedding directly in
    a notebook cell."""
    source = open(path, encoding="utf-8").read()
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    # Consecutive import statements are joined tightly (single newline); any
    # other statement (def/class/constant) starts a new chunk separated by a
    # blank-line gap, so inlined imports don't look inconsistently spaced
    # next to the cell's own hand-written import header.
    chunks: list[str] = []
    import_buf: list[str] = []
    for i, node in enumerate(tree.body):
        if i == 0 and isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant) and isinstance(node.value.value, str):
            continue  # module docstring
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            continue
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("src."):
            continue  # internal cross-module ref (e.g. type hints) — already defined by an earlier cell
        decorators = getattr(node, "decorator_list", [])
        start = min([d.lineno for d in decorators] + [node.lineno])
        text = "".join(lines[start - 1 : node.end_lineno]).rstrip("\n")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            import_buf.append(text)
            continue
        if import_buf:
            chunks.append("\n".join(import_buf))
            import_buf = []
        chunks.append(text)
    if import_buf:
        chunks.append("\n".join(import_buf))
    return "\n\n\n".join(chunks) + "\n"


# ===== 1. 개요 및 현황 =====
md("# music-mood-recs - 음악 무드 분류 + 콘텐츠 기반 추천\n\n"
   "MTG-Jamendo 무드/테마 서브셋으로 CNN 무드 분류 모델을 학습하고, "
   "분류 임베딩을 코사인 유사도로 재사용해 비슷한 무드의 곡을 추천한다.\n\n"
   "- 데이터: MTG-Jamendo mood/theme subset (상위 5 태그)\n"
   "- 모델: 단순 CNN (log-mel spectrogram 입력)\n"
   "- 추천: 분류 임베딩 + cosine similarity Top-5")

# ===== 2. 과제 범위 =====
md("## 1. 개요 및 현황\n\n"
   "| 항목 | 값 |\n|---|---|\n"
   "| 데이터 | MTG-Jamendo mood/theme subset |\n"
   "| 태그 | happy, energetic, relaxing, film, dark |\n"
   "| 입력 | 30초 log-mel spectrogram (128 mels) |\n"
   "| 모델 | MoodCNN (3 conv blocks + embedding head) |\n"
   "| 학습 | CPU, BCEWithLogitsLoss, Adam |\n"
   "| 추천 | 임베딩 cosine similarity Top-5 |")

# ===== 3. 데이터 수집 =====
md("## 2. 데이터 수집\n\n"
   "MTG-Jamendo 메타데이터 다운로드 + 상위 5 태그 서브셋 필터링")
code(
    "import sys, os\n"
    "# 노트북이 submission/ 안에 있어 Jupyter cwd가 거기로 잡힐 수 있음 -> 레포 루트로 이동\n"
    "if not os.path.isdir('data') and os.path.isdir(os.path.join('..', 'data')):\n"
    "    os.chdir('..')\n"
    "sys.path.insert(0, os.getcwd())\n"
    "print('cwd:', os.getcwd())\n\n"
    "import urllib.request\n"
    "from collections import Counter\n"
    "from typing import Optional\n"
    "import pandas as pd\n\n"
    + inline_module("src/data/load_jamendo.py")
    + "\n"
    "subset = build_subset(top_n=5, dest_dir='data/jamendo')\n"
    "tags = subset['tags']\n"
    "print('tags:', tags)\n"
    "for s in ['train', 'validation', 'test']:\n"
    "    print(f'{s}: {len(subset[s])} tracks')\n"
    "subset['train'].head()")

md("오디오 다운로드 — MTG-Jamendo audio-low TAR을 받아 서브셋 트랙만 추출한다. "
   "이미 추출된 트랙은 건너뛰므로(증분 안전) 재실행해도 다시 받지 않는다.")
code(
    "import requests, tarfile\n"
    "from concurrent.futures import ThreadPoolExecutor, as_completed\n\n"
    + inline_module("src/data/download_audio.py")
    + "\n"
    "MAX_TARS = 30\n"
    "subset = restrict_subset_to_folders(subset, MAX_TARS)  # 이후 셀(EDA·학습)은 모두 이 제한된 서브셋을 기준으로 한다\n"
    "for s in ['train', 'validation', 'test']:\n"
    "    print(f'{s} (restricted): {len(subset[s])} tracks')\n\n"
    "download_and_extract_subset(subset, 'data/audio', MAX_TARS, parallel=3)\n\n"
    "subset_meta = pd.concat(\n"
    "    [subset[s].assign(split=s) for s in ['train', 'validation', 'test']],\n"
    "    ignore_index=True,\n"
    ")\n"
    "subset_meta.to_csv('artifacts/subset_meta.csv', index=False)\n"
    "print(f'Saved restricted subset metadata ({len(subset_meta)} rows) to artifacts/subset_meta.csv')")

# ===== 4. EDA =====
md("## 3. 탐색적 데이터 분석 (EDA)\n\n"
   "태그별 분포 및 트랙 길이 분포 확인")
code(
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n\n"
    "all_train = subset['train']\n"
    "tag_cols = [f'tag_{t}' for t in tags]\n"
    "counts = all_train[tag_cols].sum().sort_values(ascending=False)\n"
    "print('train tag counts:')\n"
    "print(counts)\n\n"
    "fig, axes = plt.subplots(1, 2, figsize=(14, 4))\n"
    "counts.plot.bar(ax=axes[0], title='train tag distribution')\n"
    "all_train['DURATION'].astype(float).hist(bins=30, ax=axes[1])\n"
    "axes[1].set_title('track duration distribution (s)')\n"
    "plt.tight_layout()\n"
    "plt.show()")

# ===== 5. 전처리 =====
md("## 4. 데이터 전처리 - 멜스펙트로그램 추출\n\n"
   "각 트랙에서 30초 세그먼트를 잘라 log-mel spectrogram 계산. "
   "`data/audio/`의 다운로드된 mp3에서 직접 추출하며, 이미 계산된 트랙은 캐시를 재사용한다(증분 안전).")
code(
    "import os\n"
    "from dataclasses import dataclass\n"
    "from typing import Optional\n"
    "import librosa\n"
    "import numpy as np\n\n"
    + inline_module("src/preprocessing/melspec.py")
    + "\n"
    "cfg = MelspecConfig()\n"
    "print(f'sr={cfg.sr}, n_mels={cfg.n_mels}, segment={cfg.segment_seconds}s, frames={cfg.expected_frames}')\n\n"
    "MANIFEST_CSV = 'artifacts/melspec_manifest.csv'\n"
    "melspec_manifest, missing = extract_subset_melspecs(\n"
    "    meta_csv='artifacts/subset_meta.csv', audio_dir='data/audio', out_dir='artifacts/melspecs', cfg=cfg)\n"
    "melspec_manifest.to_csv(MANIFEST_CSV, index=False)\n"
    "print(f'Manifest: {len(melspec_manifest)} tracks -> {MANIFEST_CSV} (missing audio: {missing})')\n"
    "missing_ratio = missing / (len(melspec_manifest) + missing) if (len(melspec_manifest) + missing) else 0\n"
    "if missing_ratio > 0.05:\n"
    "    print(f'경고: 누락 비율 {missing_ratio:.1%} — 오디오 다운로드 상태를 확인하세요.')\n\n"
    "example_mel = np.load(melspec_manifest.iloc[0]['npy_path'])\n"
    "print('melspec shape:', example_mel.shape)\n"
    "plt.imshow(example_mel, aspect='auto', origin='lower', cmap='magma')\n"
    "plt.title('log-mel spectrogram example')\n"
    "plt.xlabel('time frames'); plt.ylabel('mel bins')\n"
    "plt.show()")

# ===== 6. 모델 정의 =====
md("## 5. 모델 정의 및 컴파일\n\n"
   "MoodCNN: 3 conv blocks + embedding head + linear classifier")
code(
    "from dataclasses import dataclass\n"
    "import torch\n"
    "import torch.nn as nn\n\n"
    + inline_module("src/models/cnn.py")
    + "\n"
    "cfg_model = CNNConfig(n_mels=128, n_classes=len(tags), embedding_dim=64)\n"
    "model = MoodCNN(cfg_model)\n"
    "print(model)\n"
    "print(f'\\nParameters: {count_parameters(model):,}')\n\n"
    "criterion = torch.nn.BCEWithLogitsLoss()\n"
    "optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)")

# ===== 7. 모델 학습 =====
md("## 6. 모델 학습\n\n"
   "BCEWithLogitsLoss로 멀티라벨 무드 분류 학습")
code(
    "import os\n"
    "import json\n"
    "from typing import Sequence\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import torch\n"
    "from torch.utils.data import Dataset, DataLoader\n"
    "from sklearn.metrics import f1_score, roc_auc_score\n\n"
    + inline_module("src/data/dataset.py")
    + "\n"
    + inline_module("src/evaluation/metrics.py")
    + "\n"
    "MODEL_OUT = 'models/cnn'\n"
    "import time\n"
    "def safe_torch_save(obj, path, max_retries=5):\n"
    "    # temp file + atomic rename, retries on transient Windows file locks\n"
    "    # (e.g. antivirus briefly scanning the freshly written file)\n"
    "    tmp_path = path + '.tmp'\n"
    "    for attempt in range(1, max_retries + 1):\n"
    "        try:\n"
    "            torch.save(obj, tmp_path)\n"
    "            os.replace(tmp_path, path)\n"
    "            return\n"
    "        except RuntimeError:\n"
    "            if attempt == max_retries:\n"
    "                raise\n"
    "            time.sleep(1)\n\n"
    "manifest = 'artifacts/melspec_manifest.csv'\n"
    "subset_meta = 'artifacts/subset_meta.csv'\n"
    "if not os.path.exists(manifest) or not os.path.exists(subset_meta):\n"
    "    print('매니페스트/메타데이터가 없습니다. scripts/download_audio.py + scripts/extract_melspecs.py를 먼저 실행하세요.')\n"
    "    train_ds = val_ds = None\n"
    "else:\n"
    "    train_ds = MelspecDataset(manifest, subset_meta, tags, 'train')\n"
    "    val_ds = MelspecDataset(manifest, subset_meta, tags, 'validation')\n"
    "    print(f'train: {len(train_ds)}, val: {len(val_ds)}')\n\n"
    "if train_ds is not None:\n"
    "    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)\n"
    "    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)\n\n"
    "    os.makedirs(MODEL_OUT, exist_ok=True)\n"
    "    epochs = 15\n"
    "    history = []\n"
    "    best_val_f1 = -1.0\n"
    "    for epoch in range(1, epochs+1):\n"
    "        model.train()\n"
    "        train_loss = 0.0\n"
    "        for x, y, _ in train_loader:\n"
    "            optimizer.zero_grad()\n"
    "            logits = model(x)\n"
    "            loss = criterion(logits, y)\n"
    "            loss.backward()\n"
    "            optimizer.step()\n"
    "            train_loss += loss.item() * x.size(0)\n"
    "        train_loss /= len(train_ds)\n\n"
    "        model.eval()\n"
    "        all_logits, all_labels, val_loss = [], [], 0.0\n"
    "        with torch.no_grad():\n"
    "            for x, y, _ in val_loader:\n"
    "                logits = model(x)\n"
    "                val_loss += criterion(logits, y).item() * x.size(0)\n"
    "                all_logits.append(logits.numpy())\n"
    "                all_labels.append(y.numpy())\n"
    "        val_metrics = compute_metrics(np.concatenate(all_logits), np.concatenate(all_labels))\n"
    "        val_metrics['loss'] = round(val_loss / len(val_ds), 4)\n\n"
    "        print(f'epoch {epoch:02d}/{epochs} train_loss={train_loss:.4f} val_loss={val_metrics[\"loss\"]:.4f} '\n"
    "              f'val_f1_micro={val_metrics[\"f1_micro\"]:.4f} val_acc={val_metrics[\"accuracy\"]:.4f}')\n"
    "        history.append({'epoch': epoch, 'train_loss': round(train_loss, 4), **val_metrics})\n"
    "        if val_metrics['f1_micro'] > best_val_f1:\n"
    "            best_val_f1 = val_metrics['f1_micro']\n"
    "            safe_torch_save(model.state_dict(), os.path.join(MODEL_OUT, 'model.pt'))\n"
    "            print(f'  -> saved best model (val_f1_micro={best_val_f1:.4f})')\n\n"
    "    with open(os.path.join(MODEL_OUT, 'tags.json'), 'w', encoding='utf-8') as f:\n"
    "        json.dump(tags, f, ensure_ascii=False, indent=2)\n"
    "    with open(os.path.join(MODEL_OUT, 'config.json'), 'w', encoding='utf-8') as f:\n"
    "        json.dump({'n_mels': cfg_model.n_mels, 'n_classes': cfg_model.n_classes, 'embedding_dim': cfg_model.embedding_dim}, f, indent=2)\n"
    "    with open(os.path.join(MODEL_OUT, 'metrics.json'), 'w', encoding='utf-8') as f:\n"
    "        json.dump({'best_val_f1_micro': best_val_f1, 'history': history, 'tags': tags}, f, ensure_ascii=False, indent=2)\n"
    "    print(f'학습 완료. Best val F1(micro)={best_val_f1:.4f}. Artifacts in {MODEL_OUT}')")

# ===== 7.5 테스트셋 평가 =====
md("## 6.5 테스트셋 평가 (held-out)\n\n"
   "학습/검증에 전혀 사용되지 않은 test split으로 최종 일반화 성능을 확인한다.")
code(
    "if train_ds is not None:\n"
    "    model.load_state_dict(torch.load(os.path.join(MODEL_OUT, 'model.pt'), map_location='cpu'))\n"
    "    model.eval()\n"
    "    test_ds = MelspecDataset(manifest, subset_meta, tags, 'test')\n"
    "    print(f'test: {len(test_ds)}')\n"
    "    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)\n"
    "    all_logits, all_labels = [], []\n"
    "    with torch.no_grad():\n"
    "        for x, y, _ in test_loader:\n"
    "            all_logits.append(model(x).numpy())\n"
    "            all_labels.append(y.numpy())\n"
    "    test_metrics = compute_metrics(np.concatenate(all_logits), np.concatenate(all_labels))\n"
    "    print('test set 성능:', test_metrics)\n"
    "    with open(os.path.join(MODEL_OUT, 'metrics.json'), encoding='utf-8') as f:\n"
    "        saved_metrics = json.load(f)\n"
    "    saved_metrics['test'] = test_metrics\n"
    "    with open(os.path.join(MODEL_OUT, 'metrics.json'), 'w', encoding='utf-8') as f:\n"
    "        json.dump(saved_metrics, f, ensure_ascii=False, indent=2)")

# ===== 8. 학습 과정 시각화 =====
md("## 7. 학습 과정 시각화\n\n"
   "학습/검증 loss 및 F1 변화 (metrics.json에서 로드)")
code(
    "import json\n"
    "metrics_path = 'models/cnn/metrics.json'\n"
    "if os.path.exists(metrics_path):\n"
    "    with open(metrics_path, encoding='utf-8') as f:\n"
    "        metrics = json.load(f)\n"
    "    history = pd.DataFrame(metrics['history'])\n"
    "    fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
    "    history.plot(x='epoch', y=['train_loss','loss'], ax=axes[0], title='loss')\n"
    "    history.plot(x='epoch', y=['f1_micro','f1_macro'], ax=axes[1], title='F1')\n"
    "    plt.tight_layout(); plt.show()\n"
    "else:\n"
    "    print('metrics.json이 없습니다. 학습을 먼저 실행하세요.')")

# ===== 9. 예측 =====
md("## 8. 모델 예측 - 무드 분류 + 추천\n\n"
   "선택한 곡의 무드 예측 + 코사인 유사도 Top-5 추천")
code(
    "import os\n"
    "import pandas as pd\n"
    "import numpy as np\n"
    "import torch\n"
    "from sklearn.metrics.pairwise import cosine_similarity\n\n"
    + inline_module("src/recommend/similar.py")
    + "\n"
    "model_path = 'models/cnn/model.pt'\n"
    "manifest = 'artifacts/melspec_manifest.csv'\n"
    "if not os.path.exists(model_path):\n"
    "    print('model.pt가 없습니다. 학습을 먼저 실행하세요.')\n"
    "elif not os.path.exists(manifest):\n"
    "    print('매니페스트가 없습니다. scripts/extract_melspecs.py를 먼저 실행하세요.')\n"
    "else:\n"
    "    model.load_state_dict(torch.load(model_path, map_location='cpu'))\n"
    "    model.eval()\n"
    "    man = pd.read_csv(manifest)\n"
    "    all_mels = np.stack([np.load(p) for p in man['npy_path']])\n"
    "    track_ids = man['TRACK_ID'].tolist()\n\n"
    "    sample_x = torch.from_numpy(all_mels[0:1]).unsqueeze(1)\n"
    "    with torch.no_grad():\n"
    "        probs = torch.sigmoid(model(sample_x))[0].numpy()\n"
    "    top_moods = sorted(zip(tags, probs), key=lambda t: -t[1])[:5]\n"
    "    print(f'{track_ids[0]} 무드 예측 top-5:', [(t, round(float(p),3)) for t,p in top_moods])\n\n"
    "    embeddings = extract_embeddings(model, all_mels, batch_size=32)\n"
    "    idxs, sims = top_k_similar(0, embeddings, k=5)\n"
    "    print('top-5 similar:', [(track_ids[i], round(float(s),3)) for i,s in zip(idxs, sims)])")

# ===== 10. 프로토타입 =====
md("## 9. 프로토타입 (Streamlit)\n\n"
   "``streamlit run app.py`` 로 실행 — 곡 선택 -> 무드 예측 -> Top-5 추천")

# ===== 11. 보완사항 =====
md("## 10. 보완사항 및 개선점\n\n"
   "1. CRNN 확장 (시간적 패턴 학습으로 성능 향상 가능)\n"
   "2. 데이터 서브셋 확대 (현재 30 TAR 폴더만 사용, 전체 100폴더로 확대 시 약 3.3배 규모)\n"
   "3. 추천 정량 평가 지표 도입 (현재 정성 사례 비교만)")

# ===== 12. 후기 =====
md("## 11. 소감 및 후기\n\n"
   "오디오 모달리티와 추천 시스템을 단일 모델로 증명한 DL 포트폴리오 프로젝트. "
   "분류 임베딩을 추천에 재사용하는 가설을 검증하고, CPU 환경에서도 작은 CNN으로 "
   "합리적인 무드 분류가 가능함을 확인했다.")

nb['cells'] = cells
nb['metadata']['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
nb['metadata']['language_info'] = {'name': 'python', 'version': '3.10'}

import os
os.makedirs('submission', exist_ok=True)
with open('submission/music_mood_recs.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print('wrote submission/music_mood_recs.ipynb with', len(cells), 'cells')
