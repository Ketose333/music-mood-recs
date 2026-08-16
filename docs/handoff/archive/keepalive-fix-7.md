# 이슈 #7 — 공개 앱 인증 부트스트랩 오탐 제거

- 완료: PR #8 (`9298dba`)
- 원인: 중간 `share.streamlit.io/-/auth/app` response를 최종 인증 상태로 오인
- 해결: redirect 완료 후 최종 URL과 top-level/iframe 앱 준비 문구로 판정
- 검증: 구현자·독립 reviewer 실제 운영 Chromium smoke PASS, security P1/P2 0, auto-review PASS

