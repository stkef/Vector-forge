# Filtered search and the query planner

Code: `vectorforge/engine/filters.py`, `planner.py`.
Benchmark: `scripts/bench_filtered.py`.

```python
col.search(v, k=10, filter={"lang": "te", "year": {"gt": 2024}})
col.explain(k=10, filter={"lang": "te"})      # the plan, without running it
```

Conditions: equality, `in`, `gt/gte/lt/lte`, ANDed across fields. A list
value matches if any element matches.

## Where the filter is evaluated

SQLite stores payloads durably, but querying it per search would scan rows
on every call. An in-memory payload index keeps postings (value -> slots)
and numeric columns (field -> float array), and turns a filter into a
boolean mask over slots in microseconds to milliseconds. Slots are
append-only, so the index never deletes anything: stale entries point at
dead slots, which the live mask removes.

## Three ways to use the mask (100k x 128, 200 queries, single thread)

| selectivity | points | post-filter recall | graph + mask QPS | subset scan QPS | planner picks |
|---:|---:|---:|---:|---:|---|
| 0.1% | 100 | **0.009** | 18 | 62,794 | subset scan |
| 1% | 1,000 | 0.118 | 81 | 32,253 | subset scan |
| 5% | 5,000 | 0.534 | 304 | 12,681 | subset scan |
| 20% | 20,000 | 0.989 | 930 | 3,386 | subset scan |
| 50% | 50,000 | 1.000 | 1,614 | 1,214 | graph |
| 90% | 90,000 | 1.000 | 3,963 | 724 | graph |

Graph + mask and subset scan both have recall >= 0.995 everywhere.

- **Post-filtering** (search unfiltered for 10k, drop non-matches) is what
  you get bolting a filter onto any index. With a 0.1% filter, 100 candidates
  contain on average 0.1 matches: recall collapses.
- **Graph + mask** walks through non-matching nodes but never returns them.
  It keeps walking until it has `ef` matches, so recall holds, but at 1%
  selectivity it visits ~100x more nodes: 18 QPS.
- **Subset scan** brute-forces only the matching points. Exact, and the
  cheapest option while the filter is narrow.

## The cost model

    subset scan  = candidates
    graph + mask = 5 x ef x 2M / selectivity     (5 fitted from the table)

The planner takes the cheaper. At 1M points the crossover moves to ~10%:
narrow filters stay exact, broad ones use the graph.

I first raised `ef` for filtered searches (ef / sqrt(selectivity)). The
benchmark showed recall was already >= 0.995 without it and the boost only
slowed things down, so it was removed.
