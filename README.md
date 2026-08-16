<a id="readme-top"></a>

# music-mood-recs

MTG-Jamendo 오디오 데이터 기반 음악 무드 분류 웹앱. CNN(멜스펙트로그램 입력)으로 5개 무드 태그(happy/energetic/relaxing/film/dark)를 분류하고, 분류 과정에서 학습된 임베딩을 코사인 유사도로 재사용해 비슷한 무드의 곡을 추천한다. **LLM 확장**으로 자연어 무드 분석(Ollama → Groq → 키워드 휴리스틱 폴백)과 실제 발매 음원 Top-5 추천(iTunes 검증 + Spotify·YouTube Music·Apple Music 링크)을 지원한다. 머신러닝 수업 과제로 시작한 프로젝트이며, Streamlit Cloud에 배포되어 있다.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)

[**라이브 데모 »**](https://music-mood-recs.streamlit.app)

<!-- PORTFOLIO:FACTS:START -->
- 기간: 2026.06.25 ~ 2026.07.04 (9일) (completed)
- 현재 스택: Python, PyTorch, Librosa, Streamlit, Ollama, Groq(`openai/gpt-oss-120b`)
- 현재 설명: CNN 무드 분류와 임베딩 추천을 결합한 Streamlit 앱
- 저장소: https://github.com/Ketose333/music-mood-recs
- 데모: https://music-mood-recs.streamlit.app
<!-- PORTFOLIO:FACTS:END -->

## 목차

1. [배경](#배경)
2. [데이터](#데이터)
3. [파이프라인](#파이프라인)
4. [로드맵](#로드맵)
5. [모델 성능](#모델-성능)
6. [기능](#기능)
7. [디렉터리 구조](#디렉터리-구조)
8. [로컬 환경 셋업](#로컬-환경-셋업)
9. [모델 학습](#모델-학습-이미-학습된-아티팩트가-models에-있으면-건너뛰어도-됨)
10. [앱 실행](#앱-실행)
11. [테스트](#테스트)
12. [배포 (Streamlit Cloud)](#배포-streamlit-cloud)
13. [상시 유지 (Keep-Alive)](#상시-유지-keep-alive)
14. [라이선스](#라이선스)
15. [연락처](#연락처)

<p align="right">(<a href="#readme-top">맨 위로</a>)</p>

## 배경

음악 추천은 보통 협업 필터링(다른 사용자의 청취 기록)에 의존하지만, 신곡·롱테일 곡처럼 청취 기록이 적은 곡에는 콘텐츠 기반 접근이 보완책이 된다. 이 프로젝트는 머신러닝 수업 과제로 시작했으며, **오디오 신호 자체에서 추출한 멜스펙트로그램으로 무드를 분류**하고, 분류 모델이 학습한 임베딩을 그대로 재사용해 추천까지 보여주는 단일 모델·단일 데이터셋 파이프라인을 목표로 한다.

<p align="right">(<a href="#readme-top">맨 위로</a>)</p>

## 데이터

| 항목 | 값 |
| --- | --- |
| 데이터셋 | MTG-Jamendo 무드/테마 서브셋 |
| 출처 | [github.com/MTG/mtg-jamendo-dataset](https://github.com/MTG/mtg-jamendo-dataset) |
| 태그 | 상위 5 태그(happy, energetic, relaxing, film, dark) |
| 규모 | 오디오 TAR 100개(전체) 기준 6,725곡 |
| 포맷 | 오디오(저비트레이트 mp3) + 멜스펙트로그램(30초 세그먼트, 128 mels, `.npy`) |
| 라이선스 | 메타데이터 CC BY-NC-SA 4.0, 오디오 개별 CC 라이선스(비상업 연구용) |

> 멀티레이블 태그 데이터셋이라, 한 곡이 여러 무드 태그를 동시에 가질 수 있다(BCEWithLogitsLoss 사용 이유).

<p align="right">(<a href="#readme-top">맨 위로</a>)</p>

## 파이프라인

```
MTG-Jamendo 메타데이터
  └─ scripts/download_audio.py        상위 5 태그 서브셋 필터 + 오디오 TAR 다운로드(--max-tars 100)
       └─ scripts/extract_melspecs.py 멜스펙트로그램 추출(30초 세그먼트, log-mel, 128 mels)
            └─ scripts/train_cnn.py   MoodCNN 학습(BCEWithLogitsLoss, CPU)
                 ├─ models/cnn/             학습된 아티팩트 저장
                 ├─ src/evaluation/metrics.py   F1(micro/macro)/Accuracy/ROC-AUC 계산
                 └─ src/recommend/          임베딩 코사인 유사도 Top-5 추천
                      └─ app.py (Streamlit)  곡 선택 → 무드 예측/비교/EDA 탭 → 배포
```

<p align="right">(<a href="#readme-top">맨 위로</a>)</p>

## 로드맵

- [x] MTG-Jamendo 메타데이터 로드·상위 5 태그 서브셋 필터
- [x] 오디오 다운로드 + 멜스펙트로그램 추출 (100 TAR 전체, 6,725곡)
- [x] MoodCNN 학습·평가 (train/val/test 분리)
- [x] 임베딩 재사용 코사인 유사도 Top-5 추천
- [x] EDA (태그 분포/길이 분포/멜스펙 예시)
- [x] Streamlit Cloud 배포 (OOM 방지를 위한 임베딩 사전계산 + 지연 로딩)
- [ ] CRNN 확장 (베이스라인 성능 낮을 시)
- [ ] 추천 품질 정량 평가 지표 설계

> 상세 항목·비고는 [기능](#기능) 표에서, 인프라 상태·다음 작업·알려진 이슈는 [진행 현황 문서](docs/STATUS.md)에서 계속 관리합니다.

<p align="right">(<a href="#readme-top">맨 위로</a>)</p>

## 모델 성능

| 모델 | val F1(micro) | test F1(micro) | test Accuracy | test ROC-AUC | 특징 |
| --- | --- | --- | --- | --- | --- |
| **MoodCNN** | **0.3477** | 0.2227 | 0.1448 | 0.7199 | 단순 CNN(~28K params), 멜스펙트로그램 입력, CPU 학습 |

> 성능 수치는 100 TAR 전체(6,725곡) 기준 **로컬/CPU 실측치**. 단순 CNN·CPU 제약으로 분류 성능 자체는 낮지만, ROC-AUC(0.7199)는 분류기로서 최소한의 변별력을 갖췄음을 보여준다. 데이터 규모가 50→100 TAR로 늘었다고 test F1(micro)이 항상 개선되지는 않았음(0.2618 → 0.2227) — 단순 CNN 용량 한계로 보이며, 후속 개선 방향(CRNN 확장 등)은 `docs/STATUS.md` 참고.

<p align="right">(<a href="#readme-top">맨 위로</a>)</p>

## 기능

| 기능 | 상태 | 비고 |
| --- | --- | --- |
| MTG-Jamendo 메타데이터 로드·상위 5 태그 필터 | ✅ | `scripts/download_audio.py` |
| 오디오 다운로드 + 멜스펙트로그램 추출 | ✅ | `scripts/extract_melspecs.py`, `src/preprocessing/melspec.py` |
| MoodCNN 학습·평가 | ✅ | **val F1(micro) 0.3477 / test F1(micro) 0.2227**(100 TAR 전체), `models/cnn/` |
| 임베딩 재사용 코사인 유사도 추천 | ✅ | `src/recommend/`, `scripts/precompute_embeddings.py`로 사전계산 |
| 라이브러리 곡 선택 → 무드 예측 → 추천 5곡 + 오디오 재생 | ✅ | "🔍 예측" 탭 → 입력 방식 "📂 라이브러리 곡 선택", `st.audio` |
| **내 오디오 파일 업로드 → 무드 예측 → 추천 5곡** | ✅ | "🔍 예측" 탭 → 입력 방식 "🎤 오디오 업로드" — 업로드 파일을 같은 모델로 멜스펙 추출 + 추론(`src/preprocessing/melspec.py:extract_melspec`), 임베딩을 라이브러리 임베딩과 코사인 유사도 비교(`top_k_similar_to_vector`) |
| **텍스트로 기분 입력 → LLM 무드 분석 → 추천 5곡** | ✅ | "🔍 예측" 탭 → 입력 방식 "💬 텍스트로 찾기" — LLM 3단 폴백 체인(Ollama 로컬 → Groq 무료 API → 키워드 휴리스틱)으로 무드 추론(`src/llm/mood_analyzer.py:analyze_mood`), 추정 무드에 대한 분류기 확률 상위 5곡 추천(`predict_mood_probs`) |
| **실제 발매곡 검색 → 무드 예측 → 실제 발매곡 추천** | ✅ | "🔍 예측" 탭 → 입력 방식 "🔎 음원 검색" — iTunes에서 실제 발매곡을 검색해 30초 프리뷰로 무드 예측, 같은 CNN 임베딩으로 다른 실제 발매곡과 유사도 비교(라이브러리 미경유) |
| **DL × LLM 결합 랭킹** | ✅ | 오디오 입력이 있는 모드에서 LLM이 찾은 추천 후보 Top-5 각각의 iTunes 프리뷰를 CNN에 통과시켜 입력 곡과의 코사인 유사도로 재정렬(`src/recommend/preview_rank.py`). 프리뷰는 특징 추출 직후 삭제(저장·재생 없음) |
| **실제 발매 음원 Top-5 (LLM + iTunes 검증)** | ✅ | 예측 탭 4개 입력 모드 전부 — LLM이 무드에 맞는 실존 곡을 제안하면 iTunes Search API로 검증(환각 차단), Spotify·YouTube Music·Apple Music **검색 링크만** 제공(직접 재생 없음 — 저작권 안전). `src/llm/music_search.py:recommend_real_tracks` |
| EDA (태그 분포·재생시간 분포·멜스펙 예시) | ✅ | 앱 "데이터 탐색(EDA)" 탭, `scripts/compute_eda.py`로 사전계산 |
| 클라우드 메모리 최적화 | ✅ | 임베딩 사전계산(`artifacts/embeddings.npy`) + 멜스펙 지연 로딩 (무료 티어 1GB OOM 방지) |
| Streamlit Cloud 배포 | ✅ | Python 3.11 고정 필요 (아래 "배포" 참고) |

> 텍스트 무드 추정은 LLM이 담당한다(별도 NLP 모델 학습 없음): Ollama 로컬 LLM을 우선 시도하고, 없으면 Groq의 `openai/gpt-oss-120b`(`GROQ_API_KEY` env/secrets), 그것도 없으면 기존 한국어 키워드 휴리스틱으로 폴백한다. 어느 경로든 결과는 같은 5개 학습 태그로 매핑되어 오디오 분류기의 무드 확률(`predict_mood_probs`)로 곡을 고르므로, DL 파이프라인은 그대로다. LLM 모듈은 태그 목록만 입력받아 데이터셋 규모(50/100 TAR)와 무관하게 동작한다.

> 모델 성능 수치는 [모델 성능](#모델-성능) 참고.

<p align="right">(<a href="#readme-top">맨 위로</a>)</p>

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
  llm/                       LLM 확장 — mood_analyzer.py(자연어 무드 분석, 3단 폴백) + music_search.py(실음원 Top-5, iTunes 검증)

scripts/                     학습/전처리/배포 CLI 진입점
  download_audio.py / extract_melspecs.py / train_cnn.py
  precompute_embeddings.py   추천용 임베딩 사전계산 → artifacts/embeddings.npy
  compute_eda.py             EDA 통계 사전계산 → models/eda/stats.json

models/                      학습된 아티팩트 (cnn/metrics.json 포함)
artifacts/                   멜스펙트로그램(.npy) + 사전계산 임베딩
tests/                       pytest 단위 테스트

docs/
  STATUS.md                  인프라/진행상황/다음작업 작업 로그
  prd.md                     제품 요구사항 (Phase 0: 무드 분류 + 콘텐츠 기반 추천) — Done
  prd-phase-2-llm-extension.md           Phase 2(LLM 확장: 자연어 무드 분석 + 실음원 추천) — Done
```

<p align="right">(<a href="#readme-top">맨 위로</a>)</p>

## 로컬 환경 셋업

```bash
git clone https://github.com/Ketose333/music-mood-recs.git && cd music-mood-recs
pip install -r requirements.txt
```

<p align="right">(<a href="#readme-top">맨 위로</a>)</p>

## 모델 학습 (이미 학습된 아티팩트가 models/에 있으면 건너뛰어도 됨)

```bash
# 1. 데이터 다운로드 (메타데이터 + 오디오 TAR)
python scripts/download_audio.py --top-n 5 --max-tars 100

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

<p align="right">(<a href="#readme-top">맨 위로</a>)</p>

## 앱 실행

```bash
streamlit run app.py
```

"🔍 예측" 탭에서 입력 방식(음원 검색 / 오디오 업로드 / 텍스트로 찾기 / 라이브러리 곡 선택)을 고르면 무드 예측 결과 + 비슷한 무드 추천 5곡(오디오 재생 포함). "📊 모델 성능" 탭에서 학습된 모델의 F1/Accuracy/ROC-AUC 확인.

<p align="right">(<a href="#readme-top">맨 위로</a>)</p>

## 테스트

```bash
python -m pytest tests/ -v
```

<p align="right">(<a href="#readme-top">맨 위로</a>)</p>

## 배포 (Streamlit Cloud)

1. 레포를 GitHub에 push
2. [share.streamlit.io](https://share.streamlit.io)에서 레포 연결, entry point = `app.py`
3. `packages.txt`로 필요 apt 패키지 자동 설치됨
4. Python 버전은 `runtime.txt`(3.11) 기준, 확실한 적용은 앱 대시보드 **⋮ → Settings → Python version**에서 재확인할 것
5. **LLM 경로 활성화(선택)**: 앱 대시보드 **⋮ → Settings → Secrets**에 `GROQ_API_KEY = "..."` 등록(Streamlit Cloud에는 Ollama가 없어 Groq이 클라우드 LLM 경로). 미등록 시 텍스트 무드 분석은 키워드 휴리스틱, 실음원 추천은 iTunes 무드 검색으로 폴백되어 앱은 정상 동작한다
6. 로컬 fallback이 필요하면 `streamlit run app.py`로 실행한다 (로컬에서는 Ollama가 떠 있으면 자동 사용 — 기본 모델 `gemma4:e2b`, `MMR_OLLAMA_MODEL`로 변경 가능)

<p align="right">(<a href="#readme-top">맨 위로</a>)</p>

## 상시 유지 (Keep-Alive)

Streamlit Community Cloud 무료 티어는 일정 기간 트래픽이 없으면 앱이 슬립 상태로 전환된다.
`.github/workflows/keep_alive.yml`이 6시간마다 GitHub-hosted Chromium으로 앱을 방문하고,
슬립 화면이면 깨우기 버튼을 누른 뒤 앱 본문 로딩까지 검증한다. 앱은 Streamlit Community Cloud에서
**Public**이어야 하며, 인증 리다이렉트나 로딩 실패는 성공으로 숨기지 않고 Actions 실패로 기록한다.

<p align="right">(<a href="#readme-top">맨 위로</a>)</p>

## 라이선스

- 코드: MIT
- 데이터: MTG-Jamendo — 메타데이터 CC BY-NC-SA 4.0, 오디오 개별 CC 라이선스, **비상업 연구용**
- 모델 아티팩트: 본 프로젝트 산출물, 비상업 연구용

<p align="right">(<a href="#readme-top">맨 위로</a>)</p>

## 연락처

- GitHub: [Ketose333](https://github.com/Ketose333)
- 문의: 이 저장소의 GitHub Issues

<p align="right">(<a href="#readme-top">맨 위로</a>)</p>
