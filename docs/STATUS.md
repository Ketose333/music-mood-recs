# music-mood-recs — 진행상황 (STATUS)

마지막 갱신: 2026-08-16

이 문서는 music-mood-recs 프로젝트의 단일 진실 공급원(SSOT)이다. 제품 요구사항은 [`prd.md`](prd.md)를, 전체 워크스페이스 통합 상태는 `../career/docs/STATUS.md`를 참조한다.

## 인프라

| 항목 | 값 |
| --- | --- |
| DL 프레임워크 | PyTorch(CPU 빌드) |
| 오디오 전처리 | librosa + soundfile, 멜스펙트로그램 |
| 추천 | scikit-learn(cosine_similarity), 임베딩 재사용 |
| 데이터 | MTG-Jamendo 무드/테마 서브셋, 100 TAR 전체(6,725곡). 로컬+HF Hub(`Ketose333/music-mood-recs-assets`) 동시 저장 |
| Git 추적 정책 | git은 무거운 파일을 추적하지 않음(LFS 미사용, `.gitattributes` 삭제). `data/audio/`·`artifacts/melspecs/`·`artifacts/embeddings.npy`·`models/cnn/model.pt`를 HF Hub에서 런타임 로드(`app.py` `_resolve()`) |
| 보고서 생성 | `submission/보고서.pptx` 수동 관리 |
| Streamlit keep-alive | 공통 canonical Playwright로 6시간마다 방문·wake·앱 본문 로딩 검증 (PR #6, #8, #10). Public 앱 auth bootstrap 지원 |

## 데드라인

- **2026-07-07 17:30 LLM 과제 제출 / 07-07 오전 발표.** 산출물: 보고서(PPT/PDF) + 소스(ipynb·py) + Streamlit Cloud 시연 → zip 1개 이메일 제출(ahnhg2000@gmail.com, 예: `홍길동_딥러닝_LLM프로젝트.zip`). 범위·요구사항은 [`prd-phase-2-llm-extension.md`](prd-phase-2-llm-extension.md) 참고.
- ~~2026-07-01 09:00 DL 과제 발표·시연·제출~~ — 완료.

## 현재 상태 (2026-08-16)

- **Groq 종료 모델 교체 완료(PR #12)** — `llama-3.3-70b-versatile` 기본값을 `openai/gpt-oss-120b`로 교체. 모듈·Streamlit 단일 배포본을 동기화하고 JSON Object Mode 실호출 및 81개 테스트 통과.
- **Git LFS 완전 미사용** — `model.pt`를 HF Hub 자산 레포로 이전하고 `.gitattributes` 삭제, git 히스토리에서도 제거. 계정 LFS 예산 상태와 무관하게 clone 가능.
- **Streamlit keep-alive 보강** — 단순 curl 거짓 성공을 제거하고 Chromium wake·앱 준비 문구를 검증. Public 앱도 거치는 중간 auth bootstrap은 허용하고, 최종 인증 화면에 머문 경우만 실패하도록 오탐 수정.
- **LLM 확장(Phase 2) 완료** — `src/llm/mood_analyzer.py`(Ollama→Groq→키워드 3단 폴백), `src/llm/music_search.py`(iTunes 검증된 실음원 Top-5), `app.py` 예측 탭 3개 모드에 "실제 음원 Top-5" 추가.
- **Ollama 모델 확정** — `gemma4:e2b` 채택(웜업 후 ~14초, `gemma2:latest` 대비 3배 빠름). 첫 로드 ~60초 대응을 위해 타임아웃 90초 분리.
- **UX 버그 3건 수정** — 업로드 용량 초과 시 ×버튼 클릭 불가, 예측 결과 소실, 업로드 모드 재연산 방지. `submission/music_mood_recs.py`·`.ipynb` 재생성 완료.
- **발표자료 최종 완료** — LLM 확장 섹션(05) 실편집 반영 + 퇴고(참고 예시 15건 대비 톤·분량 점검) 완료. 총 27슬라이드(01~04 DL 18장, 05 LLM 9장, 06 마무리 2장 구성). 편집 작업용 메모였던 `docs/llm-presentation-outline.md`는 반영 완료 후 삭제.
- **Groq 클라우드 LLM 경로 실동작 확인 완료** — Groq 로그인 시 "Continue with GitHub"가 콜백 무한 루프에 걸리는 문제 발견(InPrivate에서도 재현되어 쿠키/확장 문제 아님으로 확인) → "Continue with Google"로 우회 성공. API 키 발급 후 Streamlit Cloud Secrets + 로컬 `.streamlit/secrets.toml` 양쪽에 등록 완료.
- **실음원 추천 "다른 곡" 재시도 캐시 버그 수정** — LLM이 매번 같은 유명곡을 답해 재시도해도 목록이 안 바뀌던 문제. 이전에 보여준 곡을 exclude로 누적해 프롬프트에서 명시적으로 제외하도록 수정(`src/llm/music_search.py`).
- **실음원 추천에 국가 필터 추가** — 한국(K-pop)/일본(J-pop)/전체 선택 가능, 장르 태그+문자 체계(한글/가나) 이중 검증. 각 추천곡에 무드 매칭 이유(한국어 1문장, 사실 주장 금지)도 함께 생성.
- **예측 탭 UI 재구성** — 입력 방식/추천 결과 표시/국가 선택을 라디오·체크박스→드롭다운 전환(이모지 제거, 도움말 아이콘 통일), 사이드바의 `데이터:`/`태그:`/`모델:` 텍스트를 이 3개 드롭다운으로 교체, "곡 선택"을 최상위 섹션으로 승격, 라이브러리/업로드/텍스트 3개 입력 모드가 모두 동일한 레이아웃(입력→예측 무드→라이브러리 Top-5→실음원 Top-5) 사용.
- **git 커밋 히스토리 재작성**(3471e76~HEAD) — GitHub 웹에서 생성된 이질적인 커밋 정리, author/committer date 일치, force-push 완료.
- **DL × LLM 결합 랭킹 추가** (`src/recommend/preview_rank.py`) — 추천 Top-5 후보 각각의 iTunes 공식 30초 프리뷰(`previewUrl`)를 기존 멜스펙→CNN 경로에 통과시켜 임베딩을 뽑고, 입력 곡 임베딩과의 코사인 유사도로 재정렬(입력 오디오가 있는 모드). LLM이 후보를 찾고 학습된 DL 모델이 순위를 매기는 구조. 프리뷰는 특징 추출 직후 삭제(저장·재생 없음 — 저작권 안전). m4a 디코딩은 pip 설치형 `imageio-ffmpeg` 폴백으로 처리(Streamlit Cloud 호환). 프리뷰 없음/디코딩 실패 곡은 점수 없이 원래 순서 유지.
- **4번째 입력 모드 "음원 검색" 추가** — 사용자가 실제 발매곡을 검색하면(iTunes) 그 곡의 30초 프리뷰로 무드를 예측하고, 같은 CNN 임베딩으로 다른 실제 발매곡들과의 유사도를 계산해 추천한다. 라이브러리를 전혀 거치지 않는 실음원 대 실음원 비교 경로로, 기존 알고리즘(멜스펙→CNN 임베딩→코사인 유사도)을 100% 재사용한다. 자기 자신이 추천 목록에 나오지 않도록 검색한 곡을 제외 처리.
- **UI 정식 서비스 톤으로 정리** — "실제 음원 Top-5"류 표현을 "추천 Top-5"로 간소화, 라이브러리 섹션은 모든 모드에서 "📚 라이브러리 데모 — DL 과제 연장" 접이식 섹션으로 하위 배치(추천 Top-5가 먼저 나옴), 입력 방식 드롭다운 순서를 [음원 검색, 오디오 업로드, 텍스트로 찾기, 라이브러리 곡 선택]으로 재배열(라이브러리를 DL 과제 데모로 명확히 후순위 처리), 추천 결과 표시 옵션도 [전체, 추천만, 라이브러리 데모만]으로 리네이밍.
- **테스트 80건 전체 통과**(preview_rank 8건 포함), 실제 화면에서 "음원 검색" 모드 검색→예측→CNN 재정렬→라이브러리 데모 접이식 섹션까지 전체 플로우 확인.
- 향후 방향: DL(CNN 임베딩/코사인 유사도)과 LLM 기능이 따로 노는 것처럼 보이지 않게, 새 기능도 최대한 DL 결과를 재사용/연계하는 방향으로 진행할 것.
- **데이터 100 TAR 전체로 확장 완료** — `MAX_TARS` 50→100. TAR 90 다운로드가 mirror 네트워크 오류로 실패해 55곡이 누락됐던 것을 사용자가 수동 다운로드/추출 후 `src/data/load_jamendo.py:build_subset()`으로 정확한 대상 55곡을 재계산해 이름 정리(`.low.mp3`→`.mp3`) + HF Hub 업로드로 복구. `artifacts/subset_meta.csv`·`melspec_manifest.csv`·`embeddings.npy` 전부 6,725행/100폴더로 일치 확인.
- **100 TAR 전체로 재학습 완료** — val F1(micro) 0.3477, test F1(micro) 0.2227 / F1(macro) 0.1368 / Accuracy 0.1448 / ROC-AUC 0.7199. 50 TAR 체크포인트(test F1-micro 0.2618) 대비 test F1(micro)은 오히려 소폭 하락(0.2227) — 단순 CNN 용량 한계로 추정, 데이터 확대만으로 성능이 항상 개선되진 않음을 확인.
- **제출 노트북 슬림화** — `submission/music_mood_recs.ipynb`에서 LLM 확장(§9, §9.1)·프로토타입 데모(§10)·보완사항(§11) 섹션 제거(30→19개 셀), "소감 및 후기"를 §9로 재번호. 노트북은 이제 순수 DL 파이프라인(데이터 수집→EDA→전처리→모델→학습→평가→시각화→예측)만 다룸. `src/llm/`·`src/recommend/preview_rank.py`·`app.py` 등 실제 소스/배포 앱은 전혀 변경 없음(테스트 80건 재확인 통과).
- **git 히스토리 55개 → 2개 커밋으로 재압축**(`git commit-tree`로 작업 트리 안전하게 보존) — 1번째: DL 과제 제출본(~2026-06-30), 2번째: LLM 확장 전체(README 포함). force-push 완료, 이후 갱신은 2번째 커밋에 계속 amend.

## 남은 작업 (P0, LLM 과제 데드라인 2026-07-07 내 필수)

- [x] ~~LangChain 대비 직접 호출+검증 체인 talking point 반영~~ — 완료(슬라이드 21 환각 대응)
- [x] ~~로컬 Ollama(`gemma4:e2b`)로 발표 시연 리허설~~ — 완료
- [x] ~~`submission/music_mood_recs.pptx` 실편집(스크린샷 캡처 포함) + 퇴고~~ — 완료(27슬라이드 최종본)
- [ ] `submission/`의 ipynb + py + 보고서를 zip(`김관영_딥러닝_LLM프로젝트.zip` 형식)으로 묶어 이메일 제출(ahnhg2000@gmail.com, 2026-07-07 17:30)

## P1 (보고서 "보완사항"으로 서술, 후속 이월 — 미착수)

- [x] ~~LLM 연동: 텍스트 무드 검색 고도화~~ — 완료 (`src/llm/mood_analyzer.py`)
- [x] ~~LLM 연동: 추천 곡 메타데이터·설명 노출~~ — 완료 (`src/llm/music_search.py`)
- [ ] CRNN 확장(베이스라인 성능 낮을 시)
- [ ] 추천 정량 평가 지표 설계
- [ ] Spotify Web API 연동으로 검색 링크 → 정확한 곡 페이지 링크 고도화

## 알려진 이슈 (열린 것만)

| 이슈 | 비고 |
| --- | --- |
| CPU 학습 시간 | 6일 데드라인 내 단순 CNN만 |
| 분류 임베딩 → 추천 재사용 가정 미검증 | 재학습 후 정성 평가 필요 |
| 모델 성능 낮음(test F1-micro 0.2227, 100 TAR 전체 기준) | 보고서에 "후속 개선점"으로 서술(CRNN 확장 등), 이번 제출에서는 시간상 스킵 |
