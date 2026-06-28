# music-mood-recs

MTG-Jamendo 오디오 데이터 기반 음악 무드 분류 웹앱. CNN(멜스펙트로그램 입력)으로 5개 무드 태그(happy/energetic/relaxing/film/dark)를 분류하고, 분류 과정에서 학습된 임베딩을 코사인 유사도로 재사용해 비슷한 무드의 곡을 추천한다. 머신러닝 수업 과제로 시작한 프로젝트이며, Streamlit Cloud에 배포되어 있다.

[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://music-mood-recs.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)

## 목차

1. [배경](#배경)
2. [데이터](#데이터)
3. [파이프라인](#파이프라인)
4. [진행 현황](#진행-현황)
5. [모델 성능](#모델-성능)
6. [기능](#기능)
7. [프로젝트 현황](#프로젝트-현황)
8. [디렉터리 구조](#디렉터리-구조)
9. [로컬 환경 셋업](#로컬-환경-셋업)
10. [모델 학습](#모델-학습-이미-학습된-아티팩트가-models에-있으면-건너뛰어도-됨)
11. [앱 실행](#앱-실행)
12. [배포 (Streamlit Cloud)](#배포-streamlit-cloud)

## 배경

음악 추천은 보통 협업 필터링(다른 사용자의 청취 기록)에 의존하지만, 신곡·롱테일 곡처럼 청취 기록이 적은 곡에는 콘텐츠 기반 접근이 보완책이 된다. 이 프로젝트는 머신러닝 수업 과제로 시작했으며, **오디오 신호 자체에서 추출한 멜스펙트로그램으로 무드를 분류**하고, 분류 모델이 학습한 임베딩을 그대로 재사용해 추천까지 보여주는 단일 모델·단일 데이터셋 파이프라인을 목표로 한다.

## 데이터

| 항목 | 값 |
| --- | --- |
| 데이터셋 | MTG-Jamendo 무드/테마 서브셋 |
| 출처 | [github.com/MTG/mtg-jamendo-dataset](https://github.com/MTG/mtg-jamendo-dataset) |
| 태그 | 상위 5 태그(happy, energetic, relaxing, film, dark) |
| 규모 | 오디오 TAR 50개 기준 3,585곡 |
| 포맷 | 오디오(저비트레이트 mp3) + 멜스펙트로그램(30초 세그먼트, 128 mels, `.npy`) |
| 라이선스 | 메타데이터 CC BY-NC-SA 4.0, 오디오 개별 CC 라이선스(비상업 연구용) |

> 멀티레이블 태그 데이터셋이라, 한 곡이 여러 무드 태그를 동시에 가질 수 있다(BCEWithLogitsLoss 사용 이유).

## 파이프라인

```
MTG-Jamendo 메타데이터
  └─ scripts/download_audio.py        상위 5 태그 서브셋 필터 + 오디오 TAR 다운로드(--max-tars 50)
       └─ scripts/extract_melspecs.py 멜스펙트로그램 추출(30초 세그먼트, log-mel, 128 mels)
            └─ scripts/train_cnn.py   MoodCNN 학습(BCEWithLogitsLoss, CPU)
                 ├─ models/cnn/             학습된 아티팩트 저장
                 ├─ src/evaluation/metrics.py   F1(micro/macro)/Accuracy/ROC-AUC 계산
                 └─ src/recommend/          임베딩 코사인 유사도 Top-5 추천
                      └─ app.py (Streamlit)  곡 선택 → 무드 예측/비교/EDA 탭 → 배포
```

## 진행 현황

- [x] MTG-Jamendo 메타데이터 로드·상위 5 태그 서브셋 필터
- [x] 오디오 다운로드 + 멜스펙트로그램 추출 (50 TAR, 3,585곡)
- [x] MoodCNN 학습·평가 (train/val/test 분리)
- [x] 임베딩 재사용 코사인 유사도 Top-5 추천
- [x] EDA (태그 분포/길이 분포/멜스펙 예시)
- [x] Streamlit Cloud 배포 (OOM 방지를 위한 임베딩 사전계산 + 지연 로딩)
- [ ] CRNN 확장 (베이스라인 성능 낮을 시)
- [ ] 추천 품질 정량 평가 지표 설계

> 상세 항목·비고는 [기능](#기능) 표, 인프라·이슈는 [docs/STATUS.md](docs/STATUS.md) 참고.

## 모델 성능

| 모델 | val F1(micro) | test F1(micro) | test Accuracy | test ROC-AUC | 특징 |
| --- | --- | --- | --- | --- | --- |
| **MoodCNN** | **0.2994** | 0.2618 | 0.1624 | 0.7593 | 단순 CNN(~28K params), 멜스펙트로그램 입력, CPU 학습 |

> 성능 수치는 50 TAR(3,585곡) 기준 **로컬/CPU 실측치**. 단순 CNN·소규모 데이터·CPU 제약으로 분류 성능 자체는 낮지만, ROC-AUC(0.7593)는 분류기로서 최소한의 변별력을 갖췄음을 보여준다. 후속 개선 방향(CRNN 확장 등)은 `docs/STATUS.md` 참고.

## 기능

| 기능 | 상태 | 비고 |
| --- | --- | --- |
| MTG-Jamendo 메타데이터 로드·상위 5 태그 필터 | ✅ | `scripts/download_audio.py` |
| 오디오 다운로드 + 멜스펙트로그램 추출 | ✅ | `scripts/extract_melspecs.py`, `src/preprocessing/melspec.py` |
| MoodCNN 학습·평가 | ✅ | **val F1(micro) 0.2977 / test F1(micro) 0.2642**, `models/cnn/` |
| 임베딩 재사용 코사인 유사도 추천 | ✅ | `src/recommend/`, `scripts/precompute_embeddings.py`로 사전계산 |
| 라이브러리 곡 선택 → 무드 예측 → 추천 5곡 + 오디오 재생 | ✅ | "🔍 예측" 탭 → 입력 방식 "📂 라이브러리 곡 선택", `st.audio` |
| **내 오디오 파일 업로드 → 무드 예측 → 추천 5곡** | ✅ | "🔍 예측" 탭 → 입력 방식 "🎤 오디오 업로드" — 업로드 파일을 같은 모델로 멜스펙 추출 + 추론(`src/preprocessing/melspec.py:extract_melspec`), 임베딩을 라이브러리 임베딩과 코사인 유사도 비교(`top_k_similar_to_vector`) |
| **텍스트로 기분 입력 → 무드 추정 → 추천 5곡** | ✅ | "🔍 예측" 탭 → 입력 방식 "💬 텍스트로 찾기" — 한국어 키워드 매칭으로 5개 태그 중 추정(`infer_mood_from_text`), 추정 무드에 대한 분류기 확률 상위 5곡 추천(`predict_mood_probs`) |
| EDA (태그 분포·재생시간 분포·멜스펙 예시) | ✅ | 앱 "데이터 탐색(EDA)" 탭, `scripts/compute_eda.py`로 사전계산 |
| 클라우드 메모리 최적화 | ✅ | 임베딩 사전계산(`artifacts/embeddings.npy`) + 멜스펙 지연 로딩 (무료 티어 1GB OOM 방지) |
| Streamlit Cloud 배포 | ✅ | Python 3.11 고정 필요 (아래 "배포" 참고) |

> 텍스트 무드 추정은 별도로 학습된 NLP 모델이 아니라, 한국어 감정 키워드를 5개 무드 태그에 매핑하는 휴리스틱이다. 프로젝트가 학습한 모델은 오디오 분류기 하나뿐이며, 텍스트 입력은 그 모델이 이미 만들어 둔 무드 확률(`predict_mood_probs`)을 사용자가 더 쉽게 트리거하는 다리 역할을 한다.

> 모델 성능 수치는 [모델 성능](#모델-성능) 참고.

## 프로젝트 현황

→ **[docs/STATUS.md](docs/STATUS.md)** — 인프라 상태, 모델 학습 완료 여부, 다음 작업, 알려진 이슈를 추적하는 작업 로그.

## 디렉터리 구조

```
app.py                      Streamlit 데모 앱 (entry point)
requirements.txt            의존성 (streamlit, scikit-learn, torch, librosa, soundfile ...)
packages.txt                Streamlit Cloud용 apt 패키지

src/
  data/                      MTG-Jamendo 메타데이터/오디오 다운로드
  preprocessing/melspec.py   멜스펙트로그램 추출 (항상 forward-slash 경로 저장 — 크로스플랫폼)
  models/                    MoodCNN 정의 + 추론 래퍼
  evaluation/metrics.py      F1(micro/macro)/Accuracy/ROC-AUC 계산
  recommend/                 임베딩 코사인 유사도 Top-5 추천

scripts/                     학습/전처리/배포 CLI 진입점
  download_audio.py / extract_melspecs.py / train_cnn.py
  precompute_embeddings.py   추천용 임베딩 사전계산 → artifacts/embeddings.npy
  compute_eda.py             EDA 통계 사전계산 → models/eda/stats.json

models/                      학습된 아티팩트 (cnn/metrics.json 포함)
artifacts/                   멜스펙트로그램(.npy) + 사전계산 임베딩
tests/                       pytest 단위 테스트

docs/
  STATUS.md                  인프라/진행상황/다음작업 작업 로그
  prd.md                     제품 요구사항 (Phase 0: 무드 분류 + 콘텐츠 기반 추천)
  prd-phase-1-streaming-integration.md   Phase 1(외부 스트리밍 연동) — Draft
```

## 로컬 환경 셋업

```bash
git clone https://github.com/Ketose333/music-mood-recs.git && cd music-mood-recs
pip install -r requirements.txt
```

## 모델 학습 (이미 학습된 아티팩트가 models/에 있으면 건너뛰어도 됨)

```bash
# 1. 데이터 다운로드 (메타데이터 + 오디오 TAR)
python scripts/download_audio.py --top-n 5 --max-tars 30

# 2. 멜스펙트로그램 일괄 추출
python scripts/extract_melspecs.py --audio-dir data/audio --out artifacts/melspecs

# 3. CNN 학습 — 완료됨 (models/cnn/, val F1(micro) 0.2977 / test F1(micro) 0.2642)
python scripts/train_cnn.py --epochs 15 --batch-size 32

# 4. 추천용 임베딩 사전계산 (Streamlit Cloud의 1GB 메모리 제한 대응)
python -m scripts.precompute_embeddings

# 5. EDA 통계 사전계산 → models/eda/stats.json (앱 "데이터 탐색" 탭이 로드)
python scripts/compute_eda.py
```

각 스크립트는 첫 실행 시 MTG-Jamendo 메타데이터/오디오를 `data/`에 자동 다운로드한다(`.gitignore` 처리됨, 매번 다시 받을 필요 없음).

## 앱 실행

```bash
streamlit run app.py
```

"🔍 예측" 탭에서 입력 방식(라이브러리 곡 선택 / 오디오 업로드 / 텍스트로 찾기)을 고르면 무드 예측 결과 + 비슷한 무드 추천 5곡(오디오 재생 포함). "📊 모델 성능" 탭에서 학습된 모델의 F1/Accuracy/ROC-AUC 확인.

## 테스트

```bash
python -m pytest tests/ -v
```

## 배포 (Streamlit Cloud)

1. 레포를 GitHub에 push
2. [share.streamlit.io](https://share.streamlit.io)에서 레포 연결, entry point = `app.py`
3. `packages.txt`로 필요 apt 패키지 자동 설치됨
4. Python 버전은 `runtime.txt`(3.11) 기준, 확실한 적용은 앱 대시보드 **⋮ → Settings → Python version**에서 재확인할 것
5. 로컬 fallback이 필요하면 `streamlit run app.py`로 실행한다

## 라이선스

- 코드: MIT
- 데이터: MTG-Jamendo — 메타데이터 CC BY-NC-SA 4.0, 오디오 개별 CC 라이선스, **비상업 연구용**
- 모델 아티팩트: 본 프로젝트 산출물, 비상업 연구용
