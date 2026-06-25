# music-mood-recs

음악 오디오 무드 분류 모델(CNN/CRNN, 멜스펙트로그램 입력)을 학습하고, 분류 과정에서 학습된 임베딩을 코사인 유사도로 재사용해 콘텐츠 기반 추천까지 보여주는 단일 모델·단일 데이터셋·단일 도메인 DL 포트폴리오 프로젝트다.

| 항목 | 값 |
| --- | --- |
| 상태 | 기획 (착수 전) |
| 워크스페이스 | `ai-service-blueprints` 레포 바깥의 형제 디렉터리(독립 레포) |
| 데이터 | MTG-Jamendo 또는 MagnaTagATune (무드 태그 포함 공개 데이터셋) |
| 모델 | CNN/CRNN, 멜스펙트로그램 입력 |
| 데모 | Streamlit "곡 선택 → 무드 예측 → 비슷한 무드 추천 5곡" |

## 문서

상세 문서 인덱스는 [`docs/README.md`](docs/README.md)를 참조한다.

| 문서 | 내용 |
| --- | --- |
| [`docs/prd.md`](docs/prd.md) | 제품 요구사항 — Phase 0(MVP: 무드 분류 + 콘텐츠 기반 추천) |
| [`docs/prd-phase-1-streaming-integration.md`](docs/prd-phase-1-streaming-integration.md) | Phase 1(외부 스트리밍 서비스 연동) — Draft, 다른 기술 영역이라 분리 |

## 원천 후보

이 프로젝트는 `ai-service-blueprints` 워크스페이스의 후보 발굴·평가 절차를 거쳐 승격되었다. 후보 ID와 평가 근거는 `../ai-service-blueprints/_workspace-docs/topic-brainstorming.md`의 `T-026`을 참조한다(이 레포가 독립 GitHub 레포로 분리되면 상대 경로 대신 절대 URL로 교체가 필요하다).

## 기존 포트폴리오 연결

`review-sentiment`(NSMC 텍스트 감성분류), 무디트리(`KDigital3/AIproject`, 텍스트/UI)와 모달리티가 겹치지 않는 세 번째 DL 프로젝트로, 음악 오디오 도메인을 추가한다.
