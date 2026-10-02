"""Query planner: pick an execution strategy from how many points can match.

Cost model, in units of "one vector scored exactly" (calibrated with
scripts/bench_filtered.py; see docs/filtering.md):

  subset scan   cost = candidates
                Exact brute force over the matching slots only.

  HNSW + mask   cost ~ GRAPH_COST_FACTOR * ef * 2M / selectivity
                The walk keeps going until ef *matching* nodes are found, so
                it visits about 1/selectivity times more nodes than an
                unfiltered search. Fitted factor: 4-7 across 1-90% selectivity
                on 100k x 128; 5 is used.

The planner takes the cheaper one. Measured at 100k x 128: subset scan wins
up to ~20% selectivity, the graph wins from ~50%. Post-filtering (search
unfiltered, drop non-matches) is never used: its recall collapses to ~1% at
0.1% selectivity.

Raising ef for narrow filters was tried and dropped: the masked walk already
reaches recall >= 0.995 at every selectivity, so a larger ef only cost speed.

IVF-PQ is not calibrated yet and uses a fixed rule: subset scan when the
filter keeps <= 20,000 points or <= 2%, else probe more lists (nprobe/selectivity).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

GRAPH_COST_FACTOR = 5.0
IVF_SUBSET_MAX_POINTS = 20_000
IVF_SUBSET_MAX_SELECTIVITY = 0.02


@dataclass
class Plan:
    strategy: str              # empty | flat_scan | subset_scan | hnsw | ivfpq
    reason: str
    candidates: int            # points that pass the live + filter mask
    live: int
    selectivity: float
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def choose_plan(*, index_type: str, index_ready: bool, candidates: int, live: int,
                filtered: bool, exact: bool, ef: int, nprobe: int, k: int,
                graph_degree: int = 32) -> Plan:
    sel = candidates / live if live else 0.0

    def plan(strategy, reason, **params):
        return Plan(strategy, reason, candidates, live, round(sel, 6), params)

    if candidates == 0:
        return plan("empty", "no point matches")
    if exact or index_type == "flat":
        why = "exact search requested" if exact else "collection uses the flat index"
        return plan("subset_scan" if filtered else "flat_scan", why)
    if not index_ready:
        return plan("subset_scan" if filtered else "flat_scan",
                    f"{index_type} index not trained yet; searching exactly")

    if index_type == "hnsw":
        ef = max(ef, k)
        if filtered:
            graph_cost = GRAPH_COST_FACTOR * ef * graph_degree / sel
            if candidates <= graph_cost:
                return plan("subset_scan",
                            f"filter keeps {candidates:,} points ({sel:.2%}): scanning them "
                            f"(cost {candidates:,}) beats the masked graph walk "
                            f"(est. {graph_cost:,.0f})")
            return plan("hnsw", f"graph walk with filter mask (est. cost {graph_cost:,.0f} "
                                f"< scanning {candidates:,} points)", ef=ef)
        return plan("hnsw", "approximate graph search", ef=ef)

    if filtered:
        if candidates <= IVF_SUBSET_MAX_POINTS or sel <= IVF_SUBSET_MAX_SELECTIVITY:
            return plan("subset_scan", f"filter keeps {candidates:,} points ({sel:.2%}); "
                                       "scanning them exactly is cheaper")
        return plan("ivfpq", "IVF-PQ with filter mask, more lists probed",
                    nprobe=math.ceil(nprobe / sel))
    return plan("ivfpq", "approximate IVF-PQ search", nprobe=nprobe)
