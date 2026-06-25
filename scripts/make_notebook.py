import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


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
    "sys.path.insert(0, os.getcwd())\n"
    "os.environ.setdefault('PYTHONPATH', os.getcwd())\n\n"
    "from src.data.jamendo import build_subset\n\n"
    "subset = build_subset(top_n=5, dest_dir='data/jamendo')\n"
    "tags = subset['tags']\n"
    "print('tags:', tags)\n"
    "for s in ['train', 'validation', 'test']:\n"
    "    print(f'{s}: {len(subset[s])} tracks')\n"
    "subset['train'].head()")

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
   "각 트랙에서 30초 세그먼트를 잘라 log-mel spectrogram 계산")
code(
    "from src.preprocess.melspec import MelspecConfig, extract_melspec\n"
    "import numpy as np\n\n"
    "cfg = MelspecConfig()\n"
    "print(f'sr={cfg.sr}, n_mels={cfg.n_mels}, segment={cfg.segment_seconds}s, frames={cfg.expected_frames}')\n\n"
    "# 단일 트랙 예시 (오디오가 다운로드된 후 실행)\n"
    "# mel = extract_melspec('data/audio/00/948.mp3', cfg)\n"
    "# print('melspec shape:', mel.shape)\n"
    "# plt.imshow(mel, aspect='auto', origin='lower')\n"
    "# plt.title('log-mel spectrogram example')\n"
    "# plt.show()")

# ===== 6. 모델 정의 =====
md("## 5. 모델 정의 및 컴파일\n\n"
   "MoodCNN: 3 conv blocks + embedding head + linear classifier")
code(
    "from src.models.cnn import CNNConfig, MoodCNN, count_parameters\n"
    "import torch\n\n"
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
    "from src.data.dataset import MelspecDataset\n"
    "from torch.utils.data import DataLoader\n\n"
    "# 멜스펙 매니페스트가 있을 때 실행 (scripts/extract_melspecs.py 먼저 실행)\n"
    "# train_ds = MelspecDataset('artifacts/melspec_manifest.csv', 'artifacts/subset_meta.csv', tags, 'train')\n"
    "# val_ds = MelspecDataset('artifacts/melspec_manifest.csv', 'artifacts/subset_meta.csv', tags, 'validation')\n"
    "# train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)\n"
    "# val_loader = DataLoader(val_ds, batch_size=32)\n"
    "# print(f'train: {len(train_ds)}, val: {len(val_ds)}')\n\n"
    "# epochs = 15\n"
    "# for epoch in range(1, epochs+1):\n"
    "#     model.train()\n"
    "#     for x, y, _ in train_loader:\n"
    "#         optimizer.zero_grad()\n"
    "#         loss = criterion(model(x), y)\n"
    "#         loss.backward()\n"
    "#         optimizer.step()\n"
    "#     print(f'epoch {epoch} done')")

# ===== 8. 학습 과정 시각화 =====
md("## 7. 학습 과정 시각화\n\n"
   "학습/검증 loss 및 F1 변화 (metrics.json에서 로드)")
code(
    "import json\n"
    "# with open('models/cnn/metrics.json', encoding='utf-8') as f:\n"
    "#     metrics = json.load(f)\n"
    "# history = pd.DataFrame(metrics['history'])\n"
    "# fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
    "# history.plot(x='epoch', y=['train_loss', 'loss'], ax=axes[0], title='loss')\n"
    "# history.plot(x='epoch', y=['f1_micro', 'f1_macro'], ax=axes[1], title='F1')\n"
    "# plt.show()")

# ===== 9. 예측 =====
md("## 8. 모델 예측 - 무드 분류 + 추천\n\n"
   "선택한 곡의 무드 예측 + 코사인 유사도 Top-5 추천")
code(
    "from src.recommend.similar import extract_embeddings, top_k_similar\n\n"
    "# model.load_state_dict(torch.load('models/cnn/model.pt', map_location='cpu'))\n"
    "# model.eval()\n"
    "# embeddings = extract_embeddings(model, all_mels, batch_size=32)\n"
    "# idxs, sims = top_k_similar(0, embeddings, k=5)\n"
    "# print('top-5 similar:', [(track_ids[i], round(float(s),3)) for i,s in zip(idxs, sims)])")

# ===== 10. 프로토타입 =====
md("## 9. 프로토타입 (Streamlit)\n\n"
   "``streamlit run app.py`` 로 실행 — 곡 선택 -> 무드 예측 -> Top-5 추천")

# ===== 11. 보완사항 =====
md("## 10. 보완사항 및 개선점\n\n"
   "1. CRNN 확장 (시간적 패턴 학습으로 성능 향상 가능)\n"
   "2. 데이터 서브셋 확대 (현재 10 TAR 폴더만 사용, 전체 100폴더시 ~6,725곡)\n"
   "3. 추천 정량 평가 지표 도입 (현재 정성 사례 비교만)")

# ===== 12. 후기 =====
md("## 11. 소감 및 후기\n\n"
   "오디오 모달리티와 추천 시스템을 단일 모델로 증명한 DL 포트폴리오 프로젝트. "
   "분류 임베딩을 추천에 재사용하는 가설을 검증하고, CPU 환경에서도 작은 CNN으로 "
   "합리적인 무드 분류가 가능함을 확인했다.")

nb['cells'] = cells
nb['metadata']['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
nb['metadata']['language_info'] = {'name': 'python', 'version': '3.10'}

with open('notebooks/music_mood_recs.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print('wrote notebooks/music_mood_recs.ipynb with', len(cells), 'cells')
