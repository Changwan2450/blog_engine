# Blog Automation Orchestrator

이 문서는 블로그 자동화 루프의 전체 설계도다.

## Schedule

- Run twice per day
- 09:00 KST
- 21:00 KST

## Pipeline

1. **Collect sources**
   - Google News RSS
   - Reddit JSON
   - External blog RSS
2. **Filter & Deduplicate**
   - Remove duplicates
   - Remove non-English/Korean
   - Select top 10 topics
3. **Summarize**
   - Extract key points (5-7 lines)
4. **Generate Drafts**
   - Create 3 drafts with different strategy arms
5. **Select Draft**
   - Bandit selector chooses best arm
6. **Render**
   - Convert to blog markdown
   - Save to `/out` folder
7. **Publish**
   - Manual upload to Naver blog
8. **Measure**
   - Collect next-day metrics
9. **Learn**
   - Update bandit weights
