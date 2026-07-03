# music-mood-recs — 진행상황 (STATUS)

마지막 갱신: 2026-07-03

이 문서는 music-mood-recs 프로젝트의 단일 진실 공급원(SSOT)이다. 제품 요구사항은 [`prd.md`](prd.md)를, 전체 워크스페이스 통합 상태는 `../career/docs/STATUS.md`를 참조한다.

## 인프라

| 항목 | 값 |
| --- | --- |
| DL 프레임워크 | PyTorch(CPU 빌드) |
| 오디오 전처리 | librosa + soundfile, 멜스펙트로그램 |
| 추천 | scikit-learn(cosine_similarity), 임베딩 재사용 |
| 데이터 | MTG-Jamendo 무드/테마 서브셋, 50 TAR(3,585곡). 로컬+HF Hub(`Ketose333/music-mood-recs-assets`) 동시 저장. 확장 시 `MAX_TARS`만 조정 |
| Git 추적 정책 | git은 무거운 파일을 추적하지 않음(LFS 미사용, `.gitattributes` 삭제). `data/audio/`·`artifacts/melspecs/`·`artifacts/embeddings.npy`·`models/cnn/model.pt`를 HF Hub에서 런타임 로드(`app.py` `_resolve()`) |
| 보고서 생성 | `submission/보고서.pptx` 수동 관리 |

## 데드라인

- **2026-07-07 17:30 LLM 과제 제출 / 07-07 오전 발표.** 산출물: 보고서(PPT/PDF) + 소스(ipynb·py) + Streamlit Cloud 시연 → zip 1개 이메일 제출(ahnhg2000@gmail.com, 예: `홍길동_딥러닝_LLM프로젝트.zip`). 범위·요구사항은 [`prd-phase-2-llm-extension.md`](prd-phase-2-llm-extension.md) 참고.
- ~~2026-07-01 09:00 DL 과제 발표·시연·제출~~ — 완료.

## 현재 상태 (2026-07-03)

- **Git LFS 완전 미사용** — `model.pt`를 HF Hub 자산 레포로 이전하고 `.gitattributes` 삭제, git 히스토리에서도 제거. 계정 LFS 예산 상태와 무관하게 clone 가능.
- **LLM 확장(Phase 2) 완료** — `src/llm/mood_analyzer.py`(Ollama→Groq→키워드 3단 폭백), `src/llm/music_search.py`(iTunes 검증된 실음원 Top-5), `app.py` 예측 탭 3개 모드에 "실제 음원 Top-5" 추가. 신규 테스트 25건 포함 **전체 54건 통과**.
- **Ollama 모델 확정** — `gemma4:e2b` 채택(웜업 후 ~14초, `gemma2:latest` 대비 3배 빠름). 첫 로드 ~60초 대응을 위해 타임아웃 90초 분리.
- **UX 버그 3건 수정** — 업로드 용량 초과 시 ×버튼 클릭 불가, 예측 결과 소실, 업로드 모드 재연산 방지. `submission/music_mood_recs.py`·`.ipynb` 재생성 완료.
- **발표 개요 문서 작성** — [`docs/llm-presentation-outline.md`](llm-presentation-outline.md) 10슬라이드 분량. pptx 실편집은 미반영.

## 남은 작업 (P0, LLM 과제 데드라인 2026-07-07 내 필수)

- [ ] Streamlit Cloud Secrets에 `GROQ_API_KEY` 등록 후 재부팅 → 클라우드 LLM 경로 실동작 확인 — Groq 로그인 시 "Continue with GitHub"가 콜백 무한 루프에 걸리는 문제 발견(2026-07-03, InPrivate에서도 재현되어 쿠키/확장 문제 아님으로 확인). **"Continue with Google"로 우회 성공** — 이 계정으로 API 키 발급 후 등록 진행 예정
- [ ] (선택, 저위험) `docs/llm-presentation-outline.md`에 LangChain 대비 직접 호출+검증 체인 talking point 1~2문장 추가
- [ ] 로컬 Ollama(`gemma4:e2b`)로 발표 시연 리허설 — 텍스트 무드 분석 + 실음원 Top-5 동선 포함
- [ ] 위 개요를 `submission/music_mood_recs.pptx`에 실제 반영(스크린샷 2장 캡처 포함) — 사용자 직접 편집
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
| 모델 성능 낮음(test F1-micro 0.2642) | 보고서에 "후속 개선점"으로 서술(CRNN 확장 등), 이번 제출에서는 시간상 스킵 |
