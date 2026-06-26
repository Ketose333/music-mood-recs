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
| 보고서 생성 의존성 | `python-pptx` (requirements.txt엔 없음 — `scripts/make_report.py` 실행 전 `pip install python-pptx` 필요, 배포 앱은 사용 안 함) |

## 데드라인

**2026-07-01 09:00 발표·시연·제출.** 산출물: 학습 노트북(ipynb) + 소스(py) + 보고서(PPT) → zip 1개 이메일 제출(ahnhg2000@gmail.com).

## 현재 상태 (2026-06-26)

- 모델: `models/cnn/` — 30 TAR(2,247곡) 데이터로 학습 완료. **best val F1(micro)=0.2977**, **test F1(micro)=0.2642 / accuracy=0.1659 / ROC-AUC=0.7456**(태그: happy/energetic/relaxing/film/dark).
- 노트북: `submission/music_mood_recs.ipynb` 30 TAR 기준 전체 실행 완료(다운로드~멜스펙~CNN 학습~테스트셋 평가), stale 텍스트/마크다운-코드 불일치 점검 4건 모두 해결됨.
- 앱(`app.py`): 4탭(`🔍 예측`/`📊 모델 성능`/`📈 데이터 탐색(EDA)`/`ℹ️ 프로젝트 소개`) 구조. `🔍 예측` 탭 안에서 라디오로 입력 방식 3가지 선택 — **📂 라이브러리 곡 선택 / 🎤 오디오 업로드 / 💬 텍스트로 찾기**(텍스트는 키워드 휴리스틱으로 태그 추정 후 같은 분류기 확률로 추천, 별도 NLP 모델 아님). 로컬·Streamlit Cloud 모두 사용자가 직접 테스트해 정상 동작 확인됨(업로드 용량초과 에러로 ×버튼이 가려지는 문제도 `maxUploadSize` 5→50MB + "🔄 다른 파일 선택" 리셋 버튼으로 해결).
- 보고서: `음악무드분류및추천_보고서.pptx` **생성 완료**(템플릿 확보됨, 21슬라이드). 실데이터 학습 결과·그래프·오디오 업로드/텍스트 검색 데모 슬라이드까지 반영됨.
- `submission/music_mood_recs.py`는 항상 최신 `app.py`와 동기화돼 있음(매 기능 변경 후 `python scripts/package_submission.py`로 재생성).

<details>
<summary>완료된 작업 이력 (펼치기)</summary>

- **모델 재학습/노트북 점검** — 30 TAR 재학습, `metrics.json` test 키 추가, 노트북 stale 텍스트/마크다운-코드 불일치 4건 수정 (`c2c1579` 등).
- **Streamlit 데모 1차 보강** — review-sentiment 1:1 대응 UI, `st.audio` 재생 추가, 버그 3건 수정(캐시 해싱 `_model`, Windows 백슬래시 경로 크로스플랫폼 정규화, OOM 방지용 임베딩 사전계산) (`2a935cb`/`a088d6f`/`af2b8d0`/`82fe973`).
- **오디오 업로드 + 텍스트 무드 검색 추가** — 라이브러리 곡만 고를 수 있던 한계 해소. `top_k_similar_to_vector`/`predict_mood_probs`/`infer_mood_from_text`(`src/recommend/similar.py`) 추가, 업로드 파일은 `extract_melspec`으로 동일 전처리 후 같은 모델로 추론 (`3fae9fd`/`dd8fa7a`).
- **탭 6개 → 4개 통합 + 업로드 ×버튼 가려짐 수정** — 3가지 입력 방식을 별도 탭이 아닌 단일 `🔍 예측` 탭의 라디오 선택으로 합침(review-sentiment와 탭 수 대칭). `maxUploadSize` 5→50MB + 리셋 버튼 추가 (`d337002`).
- **보고서 PPT 데이터/슬라이드 갱신** — EDA 그림 재생성, 학습 곡선 스크립트 신규(`plot_training_curves.py`), 실데이터 성능 반영, 오디오 업로드/텍스트 검색 슬라이드 신규 추가, stale "future work"(오디오 업로드를 미래 계획으로 적어둔 보완사항 행) 정정 (`ca83b5e` 등 + 이번 세션).

</details>

## 남은 작업 (P0, 데드라인 내 필수)

- [ ] 발표 시연 리허설 — 입력 방식 3가지(라이브러리 선택/업로드/텍스트) 전부 시연 동선에 포함
- [ ] `scripts/package_submission.py --name 본인이름`으로 최종 zip(노트북+py+보고서) 패키징 → 이메일 제출(ahnhg2000@gmail.com, 2026-07-01 09:00)

## P1 (보고서 "보완사항"으로 서술, 후속 이월 — 미착수)

- [ ] CRNN 확장(베이스라인 성능 낮을 시)
- [ ] 추천 정량 평가 지표 설계(정성 사례 비교 위주로 보고서 작성)
- [ ] 텍스트 무드 추정을 키워드 휴리스틱에서 경량 NLP 분류기로 고도화

## 알려진 이슈 (열린 것만)

| 이슈 | 비고 |
| --- | --- |
| CPU 학습 시간 | 6일 데드라인 내 단순 CNN만 |
| 분류 임베딩 → 추천 재사용 가정 미검증 | 재학습 후 정성 평가 필요 |
| 모델 성능 낮음(test F1-micro 0.2642) | 보고서에 "후속 개선점"으로 서술(CRNN 확장 등), 이번 제출에서는 시간상 스킵 |
