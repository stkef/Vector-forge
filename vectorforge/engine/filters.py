"""Payload filters and the in-memory index that evaluates them.

Filter syntax: a dict of field -> condition, all conditions ANDed.

    {"lang": "te"}                              equals
    {"lang": {"in": ["te", "en"]}}              any of
    {"year": {"gt": 2024}}                      range: gt, gte, lt, lte (combinable)
    {"year": {"gte": 2020, "lt": 2025}, "lang": "te"}

Only top-level fields are indexed. A list value matches eq/in if any element
does (tags: ["a", "b"] matches {"tags": "a"}). Range works on scalar numbers.

SQLite stays the durable store for payloads. Evaluating filters against it
per query would mean scanning rows on every search, so this index keeps
what filtering needs in memory:

- keyword postings: (field, value) -> slots holding that value
- numeric columns:  field -> float64 array over slots, NaN where missing

Slots are append-only, so postings never need deleting: a stale entry points
at a dead slot, and dead slots are removed by ANDing with the live mask.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

RANGE_OPS = ("gt", "gte", "lt", "lte")
_SCALAR = (str, int, float, bool)


def _key(v: Any):
    """Hashable typed key so True, 1 and 1.0 don't collide."""
    if isinstance(v, bool):
        return ("b", v)
    if isinstance(v, (int, float)):
        return ("n", float(v))
    if isinstance(v, str):
        return ("s", v)
    return None


def validate_filter(flt: Any) -> dict:
    if not isinstance(flt, dict) or not flt:
        raise ValueError("filter must be a non-empty dict of field -> condition")
    for field, cond in flt.items():
        if not isinstance(field, str) or not field:
            raise ValueError("filter field names must be non-empty strings")
        if isinstance(cond, dict):
            ops = set(cond)
            if not ops or not (ops <= {"eq", "in"} | set(RANGE_OPS)):
                raise ValueError(f"bad condition for {field!r}: use eq, in, gt, gte, lt, lte")
            if ("eq" in ops or "in" in ops) and len(ops) > 1:
                raise ValueError(f"{field!r}: eq/in can't be combined with other operators")
            if "eq" in ops and _key(cond["eq"]) is None:
                raise ValueError(f"{field!r}: eq needs a string, number or bool")
            if "in" in ops and (not isinstance(cond["in"], list)
                                or any(_key(v) is None for v in cond["in"])):
                raise ValueError(f"{field!r}: in needs a list of strings, numbers or bools")
            for op in ops & set(RANGE_OPS):
                v = cond[op]
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                    raise ValueError(f"{field!r}: {op} needs a finite number")
        elif _key(cond) is None:
            raise ValueError(f"{field!r}: equality needs a string, number or bool")
    return flt


class _Slots:
    """Append-only growable int64 array."""
    __slots__ = ("a", "n")

    def __init__(self):
        self.a = np.empty(8, np.int64)
        self.n = 0

    def append(self, slot: int) -> None:
        if self.n == len(self.a):
            self.a = np.resize(self.a, 2 * len(self.a))
        self.a[self.n] = slot
        self.n += 1

    def view(self) -> np.ndarray:
        return self.a[: self.n]


class PayloadIndex:
    def __init__(self):
        self._postings: dict[str, dict[tuple, _Slots]] = {}
        self._numeric: dict[str, np.ndarray] = {}
        self._cap = 0

    def _ensure(self, n: int) -> None:
        if n <= self._cap:
            return
        cap = max(n, 2 * self._cap, 1024)
        for f, col in self._numeric.items():
            grown = np.full(cap, np.nan)
            grown[: len(col)] = col
            self._numeric[f] = grown
        self._cap = cap

    def add(self, slot: int, payload: dict) -> None:
        self._ensure(slot + 1)
        for field, value in payload.items():
            values = value if isinstance(value, list) else [value]
            post = self._postings.setdefault(field, {})
            for v in values:
                k = _key(v)
                if k is not None:
                    post.setdefault(k, _Slots()).append(slot)
            if isinstance(value, (int, float)) and not isinstance(value, bool) \
                    and math.isfinite(value):
                col = self._numeric.get(field)
                if col is None:
                    col = self._numeric[field] = np.full(self._cap, np.nan)
                col[slot] = value

    def _postings_mask(self, field: str, values: list, n: int) -> np.ndarray:
        out = np.zeros(n, bool)
        post = self._postings.get(field, {})
        for v in values:
            s = post.get(_key(v))
            if s is not None:
                slots = s.view()
                out[slots[slots < n]] = True
        return out

    def mask(self, flt: dict, n: int) -> np.ndarray:
        """Boolean mask over the first n slots: which ones match the filter."""
        out = np.ones(n, bool)
        for field, cond in flt.items():
            if not isinstance(cond, dict):
                out &= self._postings_mask(field, [cond], n)
            elif "eq" in cond:
                out &= self._postings_mask(field, [cond["eq"]], n)
            elif "in" in cond:
                out &= self._postings_mask(field, cond["in"], n)
            else:
                col = self._numeric.get(field)
                if col is None:
                    return np.zeros(n, bool)
                vals = np.full(n, np.nan)
                m = min(n, len(col))
                vals[:m] = col[:m]
                with np.errstate(invalid="ignore"):
                    if "gt" in cond:
                        out &= vals > cond["gt"]
                    if "gte" in cond:
                        out &= vals >= cond["gte"]
                    if "lt" in cond:
                        out &= vals < cond["lt"]
                    if "lte" in cond:
                        out &= vals <= cond["lte"]
            if not out.any():
                break
        return out

    def fields(self) -> dict:
        return {f: {"values": len(p), "numeric": f in self._numeric}
                for f, p in self._postings.items()}
