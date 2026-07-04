# music-mood-recs 문서

music-mood-recs 프로젝트의 상세 문서 모음이다. 프로젝트 개요는 [`../README.md`](../README.md)를 참조한다.

## 문서 목록

| 문서 | 상태 | 역할 |
| --- | --- | --- |
| [`prd.md`](prd.md) | Done | Phase 0(MVP) 제품 요구사항과 의사결정 기준선 — 무드 분류 + 콘텐츠 기반 추천 |
| [`prd-phase-2-llm-extension.md`](prd-phase-2-llm-extension.md) | Done | Phase 2 — LLM 기반 자연어 무드 분석 + 실제 발매 음원 추천 확장 |
| [`STATUS.md`](STATUS.md) | Active | 진행상황 SSOT — 인프라 표, 다음 작업 체크리스트, 알려진 이슈 |

## 해당 없음

- `architecture.md`: 학습 스크립트 + Streamlit 단일 페이지 앱 구조로, 별도 아키텍처 문서가 필요할 만큼 컴포넌트 경계가 복잡하지 않다. 필요해지면 추가한다.
- `backend.md`: 별도 백엔드 서버 없이 Streamlit + 로컬 모델 아티팩트(임베딩, 체크포인트)로 동작한다.
- `data-model.md`: 영속 DB 없이 오디오 파일 + 추출된 멜스펙트로그램/임베딩 캐시 파일로 동작한다.
- `api.md`: 외부 API 노출 없음.
