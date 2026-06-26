# music-mood-recs — 진행상황 (STATUS)

마지막 갱신: 2026-06-26

이 문서는 music-mood-recs 프로젝트의 단일 진실 공급원(SSOT)이다. 제품 요구사항은 [`prd.md`](prd.md)를, 전체 워크스페이스 통합 상태는 `../career/docs/STATUS.md`를 참조한다.

## 인프라

| 항목 | 값 |
| --- | --- |
| DL 프레임워크 | PyTorch(CPU 빌드) |
| 오디오 전처리 | librosa + soundfile, 멜스펙트로그램 |
| 추천 | scikit-learn(cosine_similarity), 임베딩 재사용 |
| 데이터 | MTG-Jamendo 무드/테마 서브셋, `--max-tars 30` 기준으로 통일(README/app.py/노트북/보고서 전부 일치) |
| Git 추적 정책 | `models/`·`artifacts/` 직접 추적(LFS 미사용). 수십MB 이상으로 커지면 `git lfs install` 재검토 |

## 데드라인

**2026-07-01 09:00 발표·시연·제출.** 산출물: 학습 노트북(ipynb) + 소스(py) + 보고서(PPT) → zip 1개 이메일 제출(ahnhg2000@gmail.com).

## 현재 상태 (2026-06-26)

- `submission/music_mood_recs.ipynb` — 30 TAR 기준 전체 실행 완료(다운로드~멜스펙 추출~CNN 재학습~테스트셋 평가까지 사용자 직접 완주).
- 오디오 다운로드/추출: TAR 00~29 전체 완료(`subset_meta.csv`/`melspec_manifest.csv` 2,247행).
- `models/cnn/` — 30 TAR 데이터로 재학습 완료. `metrics.json` 기준 **best val F1(micro)=0.2977**, **test F1(micro)=0.2642 / accuracy=0.1659 / ROC-AUC=0.7456**(태그: happy/energetic/relaxing/film/dark). 커밋·푸시 완료(`c2c1579`).
- README/STATUS의 "30 TAR" 서술과 실제 모델 학습 데이터가 이제 일치함.

## 노트북 점검 결과 처리 완료 (2026-06-26)

`submission/music_mood_recs.ipynb` 점검 항목 4건 모두 `scripts/make_notebook.py` 수정 후 재생성 완료:

- [x] **stale 텍스트**: cell "10. 보완사항 및 개선점"의 `"현재 10 TAR 폴더만 사용, 전체 100폴더시 ~6,725곡"`을 30 TAR 기준 서술로 갱신.
- [x] **마크다운-코드 불일치**: "모델 예측" 셀에 `model(x)` forward + sigmoid 확률로 무드 예측 top-5 출력 추가(`app.py` 패턴과 동일) — 이후 추천(`top_k_similar`) 계산.
- [x] **테스트셋 미사용**: 학습 셀과 시각화 셀 사이에 "6.5 테스트셋 평가" 셀 신설. `test_ds`(615곡)로 held-out 평가 후 `metrics.json`에 `test` 키로 저장.
- [x] (낮은 우선순위) 멜스펙 추출 셀에 누락 비율 5% 초과 시 경고 출력 로직 추가.

## Streamlit 데모 (app.py) — 2026-06-26 보강 완료

- [x] UI를 review-sentiment와 1:1 대응(탭 4개: 🔍 예측 / 📊 모델 성능 / 📈 데이터 탐색(EDA) / ℹ️ 프로젝트 소개), 곡 선택·추천 결과에 `st.audio` 오디오 재생 추가.
- [x] **버그 수정 1** — `st.cache_resource`가 `MoodCNN` 인스턴스를 해싱하려다 실패(`Cannot hash argument 'model'`) → 인자명에 언더스코어(`_model`)로 수정(`2a935cb`).
- [x] **버그 수정 2(크로스플랫폼)** — `melspec_manifest.csv`의 `npy_path`가 Windows 백슬래시(`artifacts\00\12100.npy`)로 저장돼 Streamlit Cloud(Linux)에서 파일을 못 찾던 문제. `src/preprocessing/melspec.py`가 항상 forward-slash로 저장하도록 수정 + 기존 CSV/노트북 일괄 정규화(`a088d6f`, `af2b8d0`).
- [x] **버그 수정 3(OOM)** — Streamlit Community Cloud는 RAM 1GB만 보장하는데 기존 `app.py`가 시작 시 멜스펙 2,247개(~1.4GB)를 전부 메모리에 올려 OOM으로 크래시. `scripts/precompute_embeddings.py`로 임베딩을 오프라인 1회 계산해 `artifacts/embeddings.npy`(0.6MB)로 분리, 예측 시에는 선택한 1곡만 지연 로딩하도록 변경(`82fe973`). **로컬·클라우드 모두 정상 동작 확인됨.**
- [x] 죽은 잔재 정리: `src/preprocess/`(빈 디렉터리), `models/cnn_synth/`(미참조 합성 테스트 아티팩트) 삭제.

## Streamlit 데모 (app.py) — 오디오 업로드/텍스트 무드 검색 추가 (2026-06-26)

기존 데모는 라이브러리에 있는 곡을 드롭다운으로 고르기만 할 수 있어("뭔지도 모르는 곡을 골라 비슷한 곡 5개만 보여줄 뿐") 사용자가 직접 자기 음악이나 기분을 입력할 수 없다는 한계가 있었음. 이를 해결하기 위해 탭 2개 추가:

- [x] **🎤 오디오 업로드** — `st.file_uploader`로 사용자가 mp3/wav/ogg/flac/m4a 파일을 직접 올리면, 학습 때와 동일한 `extract_melspec`(`src/preprocessing/melspec.py`)으로 멜스펙트로그램을 즉석 추출 → 학습된 MoodCNN으로 무드 예측 + 임베딩 추출 → `top_k_similar_to_vector`(신규, `src/recommend/similar.py`)로 라이브러리 곡과 코사인 유사도 비교해 Top-5 추천.
- [x] **💬 텍스트로 찾기** — 사용자가 기분을 한국어 문장으로 입력하면 `infer_mood_from_text`(신규)가 키워드 매칭으로 5개 태그 중 하나를 추정 → `predict_mood_probs`(신규)가 사전계산된 임베딩에 분류기 헤드(`model.classifier`)만 다시 태워 멜스펙 재로딩 없이 라이브러리 전체의 무드 확률을 구함 → 추정 무드 확률 상위 5곡 추천. 텍스트 → 무드 매핑은 별도 학습 모델이 아니라 휴리스틱 키워드 사전(README 참고)이며, 추천 자체는 기존 오디오 분류기의 출력을 재사용한다.
- 변경 파일: `src/recommend/similar.py`(`top_k_similar_to_vector`/`predict_mood_probs`/`MOOD_KEYWORDS`/`infer_mood_from_text` 추가), `scripts/sync_standalone_app.py`(BLOCKS에 melspec/신규 similar 함수 추가), `app.py`(탭 2개 추가 + 동기화), `tests/test_similar.py`(신규 함수 4개 테스트 추가).
- 검증: `python -m pytest tests/ -v` 29개 전체 PASS, `streamlit run app.py` 로컬 기동 후 HTTP 200 확인(수동 UI 클릭 테스트는 미실시 — 다음 세션에서 실제 오디오 업로드/텍스트 입력 케이스로 한 번 더 확인 권장).

## Streamlit 데모 — 탭 통합 + 업로드 X버튼 가려지는 문제 수정 (2026-06-26)

사용자가 클라우드에서 시연해보니 1) 탭이 6개로 늘어나 review-sentiment(4탭)와 비대칭이고 너무 많다는 피드백, 2) 큰 파일 업로드 시 Streamlit이 띄우는 용량 초과 에러가 업로드된 파일 칩을 가려서 제거(×) 버튼을 누르기 어렵다는 버그 제보가 있었음.

- [x] **탭 6개 → 4개로 통합** — `🔍 라이브러리 곡 예측`/`🎤 오디오 업로드`/`💬 텍스트로 찾기`를 별도 탭으로 쪼개지 않고, 단일 `🔍 예측` 탭 안에 `st.radio`(가로형) "입력 방식" 선택자로 합침. review-sentiment와 동일하게 다시 4탭(`🔍 예측`/`📊 모델 성능`/`📈 데이터 탐색(EDA)`/`ℹ️ 프로젝트 소개`) 구조가 됐고, 탭 내부 콘텐츠(3가지 입력 방식)는 그대로 유지.
- [x] **업로드 용량 초과 에러가 ×버튼을 가리는 문제** — 근본 원인은 `.streamlit/config.toml`의 `maxUploadSize`가 **5MB**로 설정돼 있었던 것(보통 mp3 한 곡이 5~10MB라 거의 항상 초과). `maxUploadSize`를 **50MB**로 올려 정상적인 곡 업로드에서는 에러 자체가 거의 발생하지 않도록 함. 그래도 큰 파일을 올리는 경우를 대비해, Streamlit 내부 ×버튼에 의존하지 않는 **"🔄 다른 파일 선택" 버튼**을 추가(`st.session_state.uploader_reset_n`을 증가시켜 `file_uploader`의 `key`를 바꿔 강제로 새 위젯 인스턴스를 만드는 방식) — 에러 메시지가 무엇을 가리든 항상 일반 크기의 버튼으로 업로드 상태를 초기화할 수 있음.
- 변경 파일: `.streamlit/config.toml`(`maxUploadSize` 5→50), `app.py`(탭 구조를 라디오 기반 단일 탭으로 리팩터링 + 리셋 버튼 추가), `submission/music_mood_recs.py`(재생성).
- 검증: `python -m pytest tests/ -v` 29개 전체 PASS, `streamlit run app.py` 로컬 기동 HTTP 200 확인. **클라우드에서 실제 큰 파일 업로드로 ×버튼/리셋 버튼 동작 재확인은 사용자가 다음 시연 때 진행 예정.**

## 보고서 PPT 데이터 갱신 (2026-06-26)

- [x] `scripts/compute_eda.py` 버그 수정 — 함수 내 중복 `import os`로 `UnboundLocalError` 발생하던 문제 해결, 재실행해 `fig_tag_distribution.png`/`fig_duration_hist.png`/`fig_melspec_example.png` 재생성 완료.
- [x] `scripts/plot_training_curves.py` 신규 추가 — `models/cnn/metrics.json`의 `history`로 train/val loss + val F1(micro/macro) 곡선 그려 `artifacts/fig_training_curves.png` 생성(이전엔 해당 그림을 만드는 스크립트가 없어 보고서 슬라이드 13/14가 깨진 상태였음).
- [x] `scripts/make_report.py` 텍스트 갱신 — 모델학습 슬라이드(13)에 실제 학습 결과(best val F1(micro)=0.2977, test F1(micro)=0.2642/accuracy=0.1659/ROC-AUC=0.7456) 반영, 보완사항(19)·소감(20) 슬라이드의 stale한 "10/100폴더" 서술을 "30 TAR(2,247곡)" 기준으로 정정. 커밋 `ca83b5e`.
- **[!] 보고서 PPTX 실제 생성은 보류** — `make_report.py`가 의존하는 템플릿 `딥러닝 산출물_202500312(홍길동v0.1).pptx`가 이 PC(워크트리)에 없음(레포에도 커밋 안 됨, 디스크 전체 검색해도 못 찾음). 사용자가 **다른 PC에서** 템플릿을 받아 `python scripts/make_report.py` 실행해 `음악무드분류및추천_보고서.pptx` 생성 예정.

## 남은 작업 (P0, 데드라인 내 필수)

- [x] 노트북 끝까지 실행 — 멜스펙 추출 + CNN 재학습 + 테스트셋 평가(완료, `c2c1579`)
- [x] 재학습 결과로 `models/cnn/metrics.json` 갱신 확인(test 성능 포함)
- [x] `app.py` Streamlit 데모 — 로컬 + Streamlit Cloud 배포 모두 정상 동작 확인
- [x] 보고서 PPT 스크립트에 실데이터 학습 결과/그래프 반영 (`scripts/make_report.py`, `scripts/plot_training_curves.py`) — 코드는 완료, **실제 PPTX 생성은 템플릿 보유 PC에서 마무리 필요**
- [ ] (다른 PC) 템플릿 파일 확보 → `python scripts/make_report.py` 실행 → PPTX 생성 확인
- [ ] 신규 "🎤 오디오 업로드"/"💬 텍스트로 찾기" 탭을 Streamlit Cloud 배포본에서 실제 오디오 파일/문장으로 한 번 더 클릭 테스트(로컬 기동만 확인됨, librosa의 mp3 디코딩이 클라우드 환경에서도 동일하게 동작하는지 확인 필요)
- [ ] 발표 시연 리허설 + `scripts/package_submission.py`로 zip 패키징 → 이메일 제출(ahnhg2000@gmail.com, 2026-07-01 09:00)

## P1 (보고서 "보완사항"으로 서술, 후속 이월 — 미착수)

- [ ] CRNN 확장(베이스라인 성능 낮을 시)
- [ ] 추천 정량 평가 지표 설계(정성 사례 비교 위주로 보고서 작성)

## 알려진 이슈 (열린 것만)

| 이슈 | 비고 |
| --- | --- |
| CPU 학습 시간 | 6일 데드라인 내 단순 CNN만 |
| 분류 임베딩 → 추천 재사용 가정 미검증 | 재학습 후 정성 평가 필요 |
| 모델 성능 낮음(test F1-micro 0.2642) | 보고서에 "후속 개선점"으로 서술(CRNN 확장 등), 이번 제출에서는 시간상 스킵 |
