# music-mood-recs — 진행상황 (STATUS)

마지막 갱신: 2026-06-25 (Git 추적 정책 검증)

이 문서는 music-mood-recs 프로젝트의 단일 진실 공급원(SSOT)이다. 제품 요구사항은 [`prd.md`](prd.md)를, 전체 워크스페이스 통합 상태는 `../career/docs/STATUS.md`를 참조한다.

## 인프라 표

| 항목 | 값 | 비고 |
| --- | --- | --- |
| 언어/런타임 | Python 3.10(로컬) / 3.11(Streamlit Cloud) | `runtime.txt` |
| DL 프레임워크 | PyTorch(CPU 빌드) | 오디오 CNN, GPU 없음 |
| 오디오 전처리 | librosa + soundfile | 멜스펙트로그램 |
| 추천 | scikit-learn(cosine_similarity) | 임베딩 재사용 |
| 데모 | Streamlit Cloud(예정) 또는 로컬 IP | `review-sentiment` 패턴 재사용 |
| 데이터 | MTG-Jamendo 무드/테마 서브셋 | §17.1, 비상업 연구용 |
| 영속 DB | 없음 | 파일 시스템 기반 |

## 마지막 머지 PR

없음(착수 전).

## 데드라인

**2026-07-01 09:00 발표·시연·제출** (딥러닝 모델 과제, 오늘 6/25 기준 6일).

산출물: 학습 노트북(ipynb) + 소스(py) + 보고서(PPT, 양식 준수) → zip 1개 이메일 제출(ahnhg2000@gmail.com).

## 다음 작업

### P0 (7/1 데드라인 내 필수, PRD §20)

- [x] 데이터셋 확정 — MTG-Jamendo 무드/테마 서브셋 채택(§24 Q1 해결, §17.1)
- [x] PRD 6일 CPU 현실적 범위로 축소 (상위 5 태그, 6,725곡, 30초 세그먼트, CRNN·평가심화는 P1 이월)
- [x] 프로젝트 골격 세팅 완료 (.gitignore, requirements.txt, src/ 패키지)
- [x] 데이터 수집 모듈 — 메타데이터 다운로드 + 상위 5 태그 서브셋 필터(PR-001)
- [x] 멜스펙 전처리 파이프라인(PR-001)
- [ ] 오디오 다운로드 — **일시 중지(5:30 이동), 재시작 필요**
  - 재시작 명령: `set PYTHONPATH=.&&python -u scripts/download_audio.py --top-n 5 --max-tars 10 --parallel 3 --out data/audio --meta-out artifacts/subset_meta.csv`
  - 백그라운드 실행: `Start-Process cmd -ArgumentList "/c","logs\run_download.cmd" -WindowStyle Hidden -WorkingDirectory "C:\AGENTS\music-mood-recs"` (logs\run_download.cmd에 위 명령 저장됨)
  - 부분 TAR 자동 삭제 후 처음부터 받음 (재시도 로직 내장)
  - 병렬 3개 동시 다운로드, 약 205KB/s, 10 TARs에 약 6.7시간 예상
  - 완료 후 자동으로 artifacts/subset_meta.csv 저장 (train 438 / val 119 / test 200)
- [x] 단순 CNN 무드 분류 모델 구현(PR-002) — 합성 데이터로 검증 완료, 실데이터 학습 대기
- [x] 임베딩 추출 + 코사인 유사도 Top-5 추천 구현(PR-003)
- [x] Streamlit 데모 앱 구현(PR-004) — 합성 데이터로 HTTP 200 확인
- [x] 학습 노트북(ipynb) 산출(PR-006) — 19셀, 양식 구조 대응
- [x] 발표 보고서 PPT 초안(PR-007) — 20슬라이드, 실데이터 결과 슬라이드는 학습 후 갱신
- [ ] 다운로드 완료 후 실데이터 멜스펙 추출 + CNN 학습
- [ ] 보고서 PPT에 실제 학습 결과/그래프 삽입
- [ ] 발표 시연 리허설 + zip 패키징 (이메일 제출)

### P1 (보고서 "5. 보완사항"으로 서술, 후속 이월)

- [ ] 모델 성능/추천 결과 평가 및 기록(PR-005)
- [ ] CRNN 확장(베이스라인 성능 낮을 시, 전환 기준 미정 §24 Q2)
- [ ] 추천 정량 평가 지표 설계(§24 Q3)

### P2 (PRD §20)

- [ ] 추천 개수 부족 시 UX 보완

## 6일 일정 계획 (2026-06-25 ~ 07-01)

| 일자 | 작업 |
| --- | --- |
| 6/25(목) | PRD 조정 완료, 골격 세팅, 데이터 수집 모듈, 멜스펙 전처리 모듈 |
| 6/26(금) | 오디오 서브셋 다운로드 + 멜스펙 추출 실행, CNN 모델 구현 |
| 6/27(토) | CNN 학습 실행(CPU, 수시간 예상), 임베딩 추출 |
| 6/28(일) | 코사인 유사도 추천 구현, Streamlit 데모 앱 |
| 6/29(월) | 노트북(ipynb) 정리, 보고서 PPT 초안 |
| 6/30(화) | 발표 시연 리허설, 보고서 완성, zip 패키징 |
| 7/1(수) | 09:00 발표·시연·제출 |

## Git 추적 정책 (2026-06-25 검증)

오디오 다운로드 재개 전, `models/`·`artifacts/` 추적 가능 여부를 점검.

| 항목 | 결과 |
| --- | --- |
| `git check-ignore models artifacts` | 무시 대상 아님 (정상 추적 가능) |
| `.gitignore` | `/data/`, `/artifacts/`, `/models/` 통째 무시 → `/data/audio/`, `/data/jamendo/`, `/data/synth_audio/`만 무시로 좁힘 (artifacts/models은 추적 허용) |
| `models/cnn_synth/` | `model.pt`(120KB) + `config.json`+`metrics.json`+`tags.json` (합성 모델, 실데이터 학습 전) |
| `models/` 전체 크기 | 130KB |
| `artifacts/` 전체 크기 | 9.6MB (합성 멜스펙 15개 + EDA 그래프 2개 + 메타 CSV) |
| `.gitattributes` (LFS 필터) | **제거함** — git-lfs 바이너리는 설치돼 있으나 이 레포에 `git lfs install`(pre-push 훅) 미실행 상태였음. 이대로 `*.pt` 커밋 시 로컬엔 LFS 포인터로 저장되나 push 시 실제 바이너리가 업로드 안 되어 원격에 깨진 포인터만 남을 위험. 현재 모델·아티팩트 크기가 매우 작아(130KB/9.6MB) LFS 불필요 |

**최종 정책**: Git 직접 추적 유지(LFS 미사용). 추후 실데이터 학습으로 모델/아티팩트가 크게 커지면(예: 수십MB 이상) 그때 `git lfs install` 정식 실행 후 `.gitattributes` 재도입.

## 알려진 이슈

| 이슈 | 상태 | 비고 |
| --- | --- | --- |
| MTG-Jamendo 오디오 용량(무드/테마 저품질 46GB) | 완화됨 | 상위 5~8 태그 서브셋(1,000~2,000곡, 30초 세그먼트)으로 필요분만 추출(§22) |
| CPU 학습 시간 | 관리 필요 | 6일 데드라인 내 단순 CNN만, CRNN은 P1 이월 |
| 분류 임베딩 → 추천 재사용 가정 미검증 | 열림 | 베이스라인 학습 후 정성 평가(§21, Medium 근거) |
| Streamlit Cloud 무료 티어 메모리 | 열림 | `st.cache_resource` 캐싱 전략 재사용 검토(§21, Low 근거) |
| CNN→CRNN 전환 기준 미정 | 열림 | §24 Q2, P1 이월 |
| 추천 정량 평가 지표 부재 | 열림 | §24 Q3, P1 이월, 정성 사례 비교 위주 |
