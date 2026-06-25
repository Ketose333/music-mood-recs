# music-mood-recs 문서

music-mood-recs 프로젝트의 상세 문서 모음이다. 프로젝트 개요는 [`../README.md`](../README.md)를 참조한다.

## 문서 목록

| 문서 | 상태 | 역할 |
| --- | --- | --- |
| [`prd.md`](prd.md) | Draft | Phase 0(MVP) 제품 요구사항과 의사결정 기준선 — 무드 분류 + 콘텐츠 기반 추천 |
| [`prd-phase-1-streaming-integration.md`](prd-phase-1-streaming-integration.md) | Draft | Phase 1 — 외부 스트리밍 서비스 연동(다른 기술 영역이라 Phase 0과 분리) |

## 해당 없음

- `architecture.md`: 학습 스크립트 + Streamlit 단일 페이지 앱 구조로, 별도 아키텍처 문서가 필요할 만큼 컴포넌트 경계가 복잡하지 않다. 필요해지면 추가한다.
- `backend.md`: 별도 백엔드 서버 없이 Streamlit + 로컬 모델 아티팩트(임베딩, 체크포인트)로 동작한다.
- `data-model.md`: 영속 DB 없이 오디오 파일 + 추출된 멜스펙트로그램/임베딩 캐시 파일로 동작한다.
- `api.md`: 외부 API 노출 없음(Phase 0 기준). Phase 1(스트리밍 연동)에서 외부 API 호출이 생기면 그때 작성한다.
