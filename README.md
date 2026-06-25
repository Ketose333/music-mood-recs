# music-mood-recs

음악 오디오 무드 분류 모델(CNN/CRNN, 멜스펙트로그램 입력)을 학습하고, 분류 과정에서 학습된 임베딩을 코사인 유사도로 재사용해 콘텐츠 기반 추천까지 보여주는 단일 모델·단일 데이터셋·단일 도메인 DL 포트폴리오 프로젝트다.

| 항목 | 값 |
| --- | --- |
| 상태 | 구현 진행 중 (7/1 발표 데드라인) |
| 워크스페이스 | `ai-service-blueprints` 레포 바깥의 형제 디렉터리(독립 레포) |
| 데이터 | MTG-Jamendo 무드/테마 서브셋(상위 5 태그: happy, energetic, relaxing, film, dark) |
| 모델 | MoodCNN (단순 CNN, 멜스펙트로그램 입력, ~28K params) |
| 데모 | Streamlit "곡 선택 → 무드 예측 → 비슷한 무드 추천 5곡" |

## 문서

상세 문서 인덱스는 [`docs/README.md`](docs/README.md)를 참조한다.

| 문서 | 내용 |
| --- | --- |
| [`docs/prd.md`](docs/prd.md) | 제품 요구사항 — Phase 0(MVP: 무드 분류 + 콘텐츠 기반 추천) |
| [`docs/STATUS.md`](docs/STATUS.md) | 진행상황 SSOT — 인프라 표, 다음 작업, 알려진 이슈 |
| [`docs/prd-phase-1-streaming-integration.md`](docs/prd-phase-1-streaming-integration.md) | Phase 1(외부 스트리밍 서비스 연동) — Draft, 다른 기술 영역이라 분리 |

## 원천 후보

이 프로젝트는 `ai-service-blueprints` 워크스페이스의 후보 발굴·평가 절차를 거쳐 승격되었다. 후보 ID와 평가 근거는 `../ai-service-blueprints/_workspace-docs/topic-brainstorming.md`의 `T-026`을 참조한다(이 레포가 독립 GitHub 레포로 분리되면 상대 경로 대신 절대 URL로 교체가 필요하다).

## 기존 포트폴리오 연결

`review-sentiment`(NSMC 텍스트 감성분류), 무디트리(`KDigital3/AIproject`, 텍스트/UI)와 모달리티가 겹치지 않는 세 번째 DL 프로젝트로, 음악 오디오 도메인을 추가한다.

## 파이프라인

```
MTG-Jamendo 메타데이터 → 상위 5 태그 서브셋 필터
  → 오디오 다운로드(audio-low TAR, --max-tars 30)
  → 멜스펙트로그램 추출(30초 세그먼트, log-mel, 128 mels)
  → MoodCNN 학습(BCEWithLogitsLoss, CPU)
  → 임베딩 추출 + 코사인 유사도 Top-5 추천
  → Streamlit 데모
```

## 실행 가이드

```bash
pip install -r requirements.txt

# 1. 데이터 다운로드 (메타데이터 + 오디오 TAR)
python scripts/download_audio.py --top-n 5 --max-tars 30

# 2. 멜스펙트로그램 일괄 추출
python scripts/extract_melspecs.py --audio-dir data/audio --out artifacts/melspecs

# 3. CNN 학습
python scripts/train_cnn.py --epochs 15 --batch-size 32

# 4. Streamlit 데모 실행
streamlit run app.py
```

## 테스트

```bash
python -m pytest tests/ -v
```

## 제출 패키징

```bash
python scripts/package_submission.py --name 본인이름
```

## 라이선스

- 코드: MIT
- 데이터: MTG-Jamendo — 메타데이터 CC BY-NC-SA 4.0, 오디오 개별 CC 라이선스, **비상업 연구용**
- 모델 아티팩트: 본 프로젝트 산출물, 비상업 연구용
