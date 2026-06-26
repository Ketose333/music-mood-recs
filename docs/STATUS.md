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

- `submission/music_mood_recs.ipynb` — 30 TAR 기준으로 재생성 완료(자체 완결, src import 없음). 학습 셀의 `torch.save`는 Windows 파일잠금(에러 1224) 재발 방지를 위해 `safe_torch_save`(임시파일+원자적 교체+재시도)로 교체됨. 아래 "노트북 점검 결과" 항목 모두 반영 완료. **사용자가 직접 실행해 다운로드~재학습까지 완료할 예정.**
- 오디오 다운로드/추출: TAR 00~29 전체 완료 확인(`data/audio/26~29`, `artifacts/melspecs/26~29`에 파일 존재, `subset_meta.csv`/`melspec_manifest.csv` 2,247행으로 갱신됨). 더 이상 재시도 불필요.
- `models/cnn/` — 아직 20 TAR(1,508곡) 기준 모델(최신 아님). 30 TAR 데이터로 재학습 필요(사용자 직접 수행).
- 노트북을 끝까지 실행하기 전까지는 README/STATUS의 "30 TAR" 서술과 실제 모델 학습 데이터가 다를 수 있음 — 완주 후 일치.

## 노트북 점검 결과 처리 완료 (2026-06-26)

`submission/music_mood_recs.ipynb` 점검 항목 4건 모두 `scripts/make_notebook.py` 수정 후 재생성 완료:

- [x] **stale 텍스트**: cell "10. 보완사항 및 개선점"의 `"현재 10 TAR 폴더만 사용, 전체 100폴더시 ~6,725곡"`을 30 TAR 기준 서술로 갱신.
- [x] **마크다운-코드 불일치**: "모델 예측" 셀에 `model(x)` forward + sigmoid 확률로 무드 예측 top-5 출력 추가(`app.py` 패턴과 동일) — 이후 추천(`top_k_similar`) 계산.
- [x] **테스트셋 미사용**: 학습 셀과 시각화 셀 사이에 "6.5 테스트셋 평가" 셀 신설. `test_ds`(615곡)로 held-out 평가 후 `metrics.json`에 `test` 키로 저장.
- [x] (낮은 우선순위) 멜스펙 추출 셀에 누락 비율 5% 초과 시 경고 출력 로직 추가.

## 남은 작업 (P0, 데드라인 내 필수)

- [ ] 노트북 끝까지 실행 — 멜스펙 추출 + CNN 재학습(사용자 직접 수행)
- [ ] 재학습 결과로 `models/cnn/metrics.json` 갱신 확인(test 성능 포함)
- [ ] 보고서 PPT에 실데이터 학습 결과/그래프 삽입 (`scripts/make_report.py`)
- [ ] 발표 시연 리허설 + `scripts/package_submission.py`로 zip 패키징 → 이메일 제출

## P1 (보고서 "보완사항"으로 서술, 후속 이월 — 미착수)

- [ ] CRNN 확장(베이스라인 성능 낮을 시)
- [ ] 추천 정량 평가 지표 설계(정성 사례 비교 위주로 보고서 작성)

## 알려진 이슈 (열린 것만)

| 이슈 | 비고 |
| --- | --- |
| CPU 학습 시간 | 6일 데드라인 내 단순 CNN만 |
| 분류 임베딩 → 추천 재사용 가정 미검증 | 재학습 후 정성 평가 필요 |
| Streamlit Cloud 무료 티어 메모리 | 배포 시 `st.cache_resource` 캐싱 전략 점검 필요 |
