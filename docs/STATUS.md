# music-mood-recs — 진행상황 (STATUS)

마지막 갱신: 2026-06-27

이 문서는 music-mood-recs 프로젝트의 단일 진실 공급원(SSOT)이다. 제품 요구사항은 [`prd.md`](prd.md)를, 전체 워크스페이스 통합 상태는 `../career/docs/STATUS.md`를 참조한다.

## 인프라

| 항목 | 값 |
| --- | --- |
| DL 프레임워크 | PyTorch(CPU 빌드) |
| 오디오 전처리 | librosa + soundfile, 멜스펙트로그램 |
| 추천 | scikit-learn(cosine_similarity), 임베딩 재사용 |
| 데이터 | MTG-Jamendo 무드/테마 서브셋, `--max-tars 30` 기준으로 통일(README/app.py/노트북/보고서 전부 일치) |
| Git 추적 정책 | `models/`만 git 직접 추적(소형). `data/audio/`·`artifacts/melspecs/`·`artifacts/embeddings.npy`(~7GB)는 GitHub LFS 무료 한도(1GB) 초과로 제외, HF Hub 데이터셋 레포(`Ketose333/music-mood-recs-assets`)에 호스팅 — 배포 앱(`app.py`의 `_resolve()`)이 런타임에 `huggingface_hub`로 받아옴 |
| 보고서 생성 의존성 | `python-pptx` (requirements.txt엔 없음 — `scripts/make_report.py` 실행 전 `pip install python-pptx` 필요, 배포 앱은 사용 안 함) |

## 데드라인

**2026-07-01 09:00 발표·시연·제출.** 산출물: 학습 노트북(ipynb) + 소스(py) + 보고서(PPT) → zip 1개 이메일 제출(ahnhg2000@gmail.com).

## 현재 상태 (2026-06-27)

- 모델: `models/cnn/` — 30 TAR(2,247곡) 데이터로 학습 완료. **best val F1(micro)=0.2977**, **test F1(micro)=0.2642 / accuracy=0.1659 / ROC-AUC=0.7456**(태그: happy/energetic/relaxing/film/dark).
- 노트북: `submission/music_mood_recs.ipynb` 30 TAR 기준 전체 실행 완료(다운로드~멜스펙~CNN 학습~테스트셋 평가), stale 텍스트/마크다운-코드 불일치 점검 4건 모두 해결됨.
- 앱(`app.py`): 4탭(`🔍 예측`/`📊 모델 성능`/`📈 데이터 탐색(EDA)`/`ℹ️ 프로젝트 소개`) 구조. `🔍 예측` 탭 안에서 라디오로 입력 방식 3가지 선택 — **📂 라이브러리 곡 선택 / 🎤 오디오 업로드 / 💬 텍스트로 찾기**(텍스트는 키워드 휴리스틱으로 태그 추정 후 같은 분류기 확률로 추천, 별도 NLP 모델 아님). 로컬·Streamlit Cloud 모두 사용자가 직접 테스트해 정상 동작 확인됨(업로드 용량초과 에러로 ×버튼이 가려지는 문제도 `maxUploadSize` 5→50MB + "🔄 다른 파일 선택" 리셋 버튼으로 해결).
- **HF Hub 데이터 이전 완료 + 재부팅 검증** — `data/audio`·`artifacts/melspecs`·`embeddings.npy`(~7GB, GitHub LFS 무료 한도 7배 초과 상태였음)를 `Ketose333/music-mood-recs-assets`로 이전, git 히스토리에서도 완전 제거(force-push, `.git` 7.69GB→12MB). `app.py`의 `_resolve()`가 런타임에 huggingface_hub로 받아오도록 변경, Streamlit Cloud 재부팅으로 정상 동작 직접 확인됨. `submission/music_mood_recs.py`·`.ipynb`도 이 변경 반영해 재생성·푸시 완료.
- 보고서: **part1/part2 분리 구조 폐기 → `submission/보고서.pptx` 단일 파일로 완전 병합 완료**(사용자 직접 작업). 프로토타이핑 화면 3슬라이드(예측+추천 / 오디오업로드·텍스트검색 / 소개탭)는 Streamlit Cloud 재부팅 후 실제 앱을 Playwright로 캡처(1518×886px = 759×443의 2배, 브라우저/Streamlit 툴바 제거한 순수 앱 화면)해 `submission/앱 1 예측화면.png`·`앱 2 업로드텍스트.png`·`앱 3 소개탭.png`로 저장, `make_report.py`로 part2에 1차 첨부 확인 후 최종적으로 part1에 수동 병합. 병합 후 더 이상 필요 없는 옛 노트북 스크린샷 8장(`노트북 N ...png`)과 `음악무드분류및추천_보고서_part2.pptx`는 사용자가 직접 삭제함 — **`scripts/make_report.py`는 이제 사용하지 않는 레거시 스크립트**(원본 스크린샷은 `artifacts/app_screens/`에 남아있음).
- 보고서용 차트 6종(EDA 3종 + 학습곡선 + 모델예측 예시 2종)을 **실데이터로 생성**해 `artifacts/report_figures/`에 759×443 이하로 통일(신규 스크립트 `scripts/plot_prediction_examples.py` + `compute_eda.py`/`plot_training_curves.py` 출력경로·크기 갱신). 묵은 `artifacts/fig_*.png` 4종은 로컬에서도 삭제됨.
- **HF Hub 데이터셋 레포(`Ketose333/music-mood-recs-assets`) 정리**: 라이선스 카드(README.md) 추가(MTG-Jamendo 출처+트랙별 CC 라이선스 분포표+비영리 연구용 명시), `app.py`가 안 쓰는 잔여물 20개(묵은 차트 PNG 4·합성테스트데이터 16) 삭제, 커밋 히스토리를 `Initial commit` 1개로 압축. **public 유지**(현재 코드에 토큰 인증이 없어 private 전환 시 채점 PC에서 401 발생 — "어느 PC에서도 실행 가능" 조건 미충족). 정리 후 매니페스트·임베딩·멜스펙·오디오 다운로드 전부 재검증 완료, 로컬 보고서 파일과는 완전히 무관(영향 없음 확인됨).
- `submission/music_mood_recs.py`는 항상 최신 `app.py`와 동기화돼 있음(매 기능 변경 후 `python scripts/sync_standalone_app.py && python scripts/make_notebook.py`로 재생성, 최종 zip 패키징 시 `package_submission.py` 사용).

<details>
<summary>완료된 작업 이력 (펼치기)</summary>

- **모델 재학습/노트북 점검** — 30 TAR 재학습, `metrics.json` test 키 추가, 노트북 stale 텍스트/마크다운-코드 불일치 4건 수정 (`c2c1579` 등).
- **Streamlit 데모 1차 보강** — review-sentiment 1:1 대응 UI, `st.audio` 재생 추가, 버그 3건 수정(캐시 해싱 `_model`, Windows 백슬래시 경로 크로스플랫폼 정규화, OOM 방지용 임베딩 사전계산) (`2a935cb`/`a088d6f`/`af2b8d0`/`82fe973`).
- **오디오 업로드 + 텍스트 무드 검색 추가** — 라이브러리 곡만 고를 수 있던 한계 해소. `top_k_similar_to_vector`/`predict_mood_probs`/`infer_mood_from_text`(`src/recommend/similar.py`) 추가, 업로드 파일은 `extract_melspec`으로 동일 전처리 후 같은 모델로 추론 (`3fae9fd`/`dd8fa7a`).
- **탭 6개 → 4개 통합 + 업로드 ×버튼 가려짐 수정** — 3가지 입력 방식을 별도 탭이 아닌 단일 `🔍 예측` 탭의 라디오 선택으로 합침(review-sentiment와 탭 수 대칭). `maxUploadSize` 5→50MB + 리셋 버튼 추가 (`d337002`).
- **보고서 PPT 데이터/슬라이드 갱신** — EDA 그림 재생성, 학습 곡선 스크립트 신규(`plot_training_curves.py`), 실데이터 성능 반영, 오디오 업로드/텍스트 검색 슬라이드 신규 추가, stale "future work"(오디오 업로드를 미래 계획으로 적어둔 보완사항 행) 정정 (`ca83b5e` 등 + 이번 세션).
- **HF Hub 데이터 이전 + 히스토리 재작성** — `data/audio`·`artifacts/melspecs`·`embeddings.npy` git 추적 해제 후 HF Hub(`Ketose333/music-mood-recs-assets`)로 이전, `app.py`에 `_resolve()` 폴백 추가(`84828dd`). `download_audio.py`에 `--hf-repo-id` 옵션 추가해 추출 즉시 업로드 가능하도록 개선(`342ddbf`). `git-filter-repo`로 전체 히스토리에서 해당 경로 제거 후 force-push(`00f7508`, `.git` 12MB). 제출 스냅샷 재생성(`d8e5144`). Streamlit Cloud 재부팅으로 정상 동작 확인됨.
- **보고서 part1/part2 분리 + 실데이터 차트화 + HF Hub 라이선스 정리** — part1(Miricanvas)·part2(`make_report.py`) 분리 후 part1이 점진적으로 흡수, `make_report.py`도 그만큼 책임 축소(최종: 프로토타이핑 화면 3슬라이드만). EDA 3종·학습곡선·모델예측 2종 차트를 리사이즈 대신 실데이터 재생성(`scripts/plot_prediction_examples.py` 신규, `compute_eda.py`/`plot_training_curves.py` 출력경로·크기 통일), `artifacts/report_figures/`로 일원화. HF Hub 레포에 라이선스 카드 추가(MTG-Jamendo 트랙별 CC 라이선스 점검 결과 포함) + 앱 미사용 잔여물 20개 삭제 + 히스토리 `Initial commit` 1개로 압축, public 유지 결정(토큰 미지원으로 private 시 채점 PC 401).
- **프로토타이핑 스크린샷 캡처 + 보고서 단일 파일 병합** — Streamlit Cloud 배포 앱을 Playwright로 직접 조작(예측+추천 실행, 텍스트 무드 검색 실행, 소개 탭)해 3장 캡처, 1518×886px(=759×443×2, 브라우저/Streamlit 툴바 제거)로 통일해 `submission/`에 저장. 이후 part1/part2를 `submission/보고서.pptx` 단일 파일로 완전 병합(사용자 작업), 옛 노트북 스크린샷 8장과 part2 PPTX는 더 이상 필요 없어 삭제. `scripts/make_report.py`는 레거시로 전환.

</details>

## 남은 작업 (P0, 데드라인 내 필수)

- [x] **Streamlit 앱 스크린샷 3장 캡처** → `submission/앱 1 예측화면.png` / `앱 2 업로드텍스트.png` / `앱 3 소개탭.png` (1518×886px, 순수 앱 화면)
- [x] part1 + part2 최종 PPT로 통합 → `submission/보고서.pptx` 단일 파일로 완료(사용자 직접 작업, 옛 part2/노트북 스크린샷 8장 삭제)
- [ ] 발표 시연 리허설 — 입력 방식 3가지(라이브러리 선택/업로드/텍스트) 전부 시연 동선에 포함
- [ ] `scripts/package_submission.py --name 본인이름 --report "submission/보고서.pptx"`로 최종 zip(노트북+py+보고서) 패키징 → 이메일 제출(ahnhg2000@gmail.com, 2026-07-01 09:00). **`--report` 기본값이 옛 파일명이라 위처럼 인자로 명시할 것**

## P1 (보고서 "보완사항"으로 서술, 후속 이월 — 미착수)

- [ ] CRNN 확장(베이스라인 성능 낮을 시)
- [ ] 추천 정량 평가 지표 설계(정성 사례 비교 위주로 보고서 작성)
- [ ] **LLM 연동**: 텍스트 무드 검색을 키워드 휴리스틱 → LLM/임베딩 기반으로 고도화 — DL 과제 종료 후 동일 레포의 LLM 과제 단계에서 진행, 이번 제출 범위 아님(7월 1일 이전 실구현 금지)
- [ ] **LLM 연동**: 추천 곡에 메타데이터(제목/아티스트)·설명을 노출해 "데이터셋 안에서 뭔지 모르는 곡을 추천받는" 문제 해소 — 동일하게 LLM 과제 단계로 이월(7월 1일 이전 실구현 금지)

## 알려진 이슈 (열린 것만)

| 이슈 | 비고 |
| --- | --- |
| CPU 학습 시간 | 6일 데드라인 내 단순 CNN만 |
| 분류 임베딩 → 추천 재사용 가정 미검증 | 재학습 후 정성 평가 필요 |
| 모델 성능 낮음(test F1-micro 0.2642) | 보고서에 "후속 개선점"으로 서술(CRNN 확장 등), 이번 제출에서는 시간상 스킵 |
