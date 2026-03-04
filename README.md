# blog_engine

자동 블로그 글 생성 + 밴딧(ε-greedy) 학습 + 백테스트 파이프라인.

---

## 파이프라인 개요

```
collect → trends → summarize → write → quality_gate → select → render → learn → backtest
```

| 단계 | 역할 |
|---|---|
| collect | Google News RSS, Reddit JSON, 외부 RSS 수집 |
| trends | 트렌드 키워드 추출 + topic_pool 업데이트 |
| summarize | 수집 아이템 정제 및 요약 (top 10) |
| write | (Arm × Lens) 조합으로 초안 3개 생성 |
| quality_gate | 제목/본문 품질 검사, 미달 시 재생성 |
| select | ε-greedy 밴딧으로 최적 초안 선택 |
| render | 마크다운 최종 렌더링 + out/ 저장 |
| learn | 성과 지표 입력 → 보상 계산 → 밴딧 상태 업데이트 |
| backtest | 과거 실행 데이터 분석 → arm/lens/term 성과 리포트 |

---

## 디렉터리 구조

```
blog_engine/
├── src/
│   ├── run_once.py      # 메인 파이프라인 진입점
│   ├── collect.py
│   ├── trends.py
│   ├── summarize.py
│   ├── write.py
│   ├── select.py
│   ├── render.py
│   ├── learn.py
│   └── backtest.py
├── data/
│   ├── google_news_rss.txt   # Google News RSS URL 목록
│   ├── reddit_sources.txt    # Reddit JSON URL 목록
│   └── sources_rss.txt       # 외부 RSS URL 목록
└── out/                      # 생성된 글 저장 위치
```

---

## Quick Start

```bash
# 아침 실행
python3 src/run_once.py --slot morning

# 저녁 실행
python3 src/run_once.py --slot evening

# 시드 고정 (재현성)
python3 src/run_once.py --slot morning --seed 42
```

결과 파일:
- `out/YYYY-MM-DD_HHMM_final.md` — 최종 발행 글
- `out/YYYY-MM-DD_HHMM_candidates/` — 선택지 초안들

---

## 학습 루프 (수동 지표 입력)

글을 발행하고 24시간 뒤 조회/좋아요/댓글 수를 입력한다.

```bash
python3 src/learn.py \
  --output_file out/2026-03-04_1100_final.md \
  --views 1200 \
  --likes 30 \
  --comments 8
```

보상 공식 (현재 구현):

```
reward = log1p(views) * 0.25 + likes * 0.05 + comments * 0.1
```

- `data/bandit_state.json` — arm/lens/term별 누적 보상 저장
- `data/metrics.csv` — 지표 이력
- `data/runs.jsonl` — 실행 이력

---

## 백테스트

```bash
python3 src/backtest.py
```

- `data/runs.jsonl` + `data/metrics.csv` 조인
- arm / lens / arm\|lens / term 별 평균 보상 계산
- 결과: 콘솔 요약 + `out/backtest_report.md` 생성

---

## 운영 루틴 (개인 기준)

| 주기 | 작업 |
|---|---|
| 매일 아침 | `run_once.py --slot morning` |
| 매일 저녁 | `run_once.py --slot evening` |
| 24시간 뒤 | `learn.py`로 지표 입력 |
| 주 1회 | `backtest.py` 실행 후 `out/backtest_report.md` 확인 |

---

## 트러블슈팅

**RSS WARN (308/406/404)**

```
[collect] WARN  rss fetch failed: https://... – HTTP Error 404
```

일부 소스 URL이 이동/삭제된 경우. 파이프라인은 계속 진행되며 나머지 소스로 수집한다.  
해당 URL을 `data/sources_rss.txt`에서 제거하거나 올바른 URL로 교체한다.

**out/ 파일 누적**

`out/` 디렉터리에 실행마다 파일이 쌓인다. 주기적으로 오래된 파일을 삭제한다:

```bash
# 7일 이상 된 out/ 파일 삭제
find out/ -name "*.md" -mtime +7 -delete
find out/ -type d -empty -delete
```

**data/ 파일 누적**

- `data/runs.jsonl`, `data/metrics.csv` — 지속적으로 추가됨 (삭제 시 학습 이력 초기화)
- `data/bandit_state.json` — 현재 학습 상태. 초기화하려면 삭제

---

## 의존성

Python 3.9+ 표준 라이브러리만 사용. 외부 패키지 없음.
