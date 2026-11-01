"""Rebuild a day-by-day burndown for every TAWOS sprint.

The paper we critique (Perez Castillo et al., Information 2024) reads its
features off the Worki burndown chart and the "completed vs not completed"
chart. TAWOS has no such charts, but our Phase 1-3 code can rebuild the sprint
membership and the resolution state of every issue at any moment from the Jira
change log. This script does that once per day of each sprint and writes one
row per (sprint, day) to data/burndown_daily.csv.

Copied from our paper critique (github.com/hani-ziad/AdvAI_critique). It uses
the reconstruction code of the pipeline copied in ../pipeline.

Only issue counts are used. Story points are missing for many TAWOS issues and
Worki's "hours" do not exist here.

    python -m advai.burndown
"""
from __future__ import annotations

from bisect import bisect_right

import pandas as pd

import advai
import phase3
from phase2.data import load_phase2_frame, population, time_ordered_split
from phase3.data import _issue_timelines
from src.data.load_tawos import load_change_log, load_issues, load_sprints
from src.data.planning_snapshot import _field_kind, _parse_change_sprint_set, _project_maps

DAY_NS = 86_400 * 10**9
OUT = advai.DATA / "burndown_daily.csv"


def daily_times(start_ns: int, stop_ns: int) -> list[int]:
    """Start, then every 24 h, then the stop time (if not already on a day boundary)."""
    n = int((stop_ns - start_ns) // DAY_NS)
    times = [start_ns + k * DAY_NS for k in range(n + 1)]
    if times[-1] < stop_ns:
        times.append(stop_ns)
    return times


def build() -> pd.DataFrame:
    sprints = load_sprints(phase3.RAW_DIR, modeling_only=True)
    issues = load_issues(phase3.RAW_DIR)
    change_log = load_change_log(phase3.RAW_DIR, required=True)
    base = load_phase2_frame()

    # Same cohort as the main project: the 1,691 sprints that committed at least one issue.
    ne = population(base, "nonempty")
    tr, te = time_ordered_split(ne)
    side = pd.concat([pd.Series("train", index=tr["sprint_id"]), pd.Series("test", index=te["sprint_id"])])
    cohort = sprints.merge(ne[["sprint_id", "project"]], on="sprint_id", how="inner", suffixes=("_raw", ""))

    maps = _project_maps(sprints)
    jira_maps, raw_maps, name_maps = maps
    events = _issue_timelines(issues, change_log, maps)

    # The chart a team could look at runs from the start to the planned end,
    # or to the moment the sprint was closed if that came first.
    times, start_ns = {}, {}
    for r in cohort.itertuples(index=False):
        s = r.start_date.value
        stop = min(r.end_date.value, r.complete_date.value)
        start_ns[r.sprint_id] = s
        times[r.sprint_id] = daily_times(s, stop)
    ids = set(times)
    acc = {sid: [[0, 0, 0, 0] for _ in times[sid]] for sid in ids}  # scope, done, committed_in, added_in

    for issue in issues.itertuples(index=False):
        project = issue.project
        if project not in jira_maps:
            continue
        ev = events.get(issue.raw_issue_id, [])
        jira_map, raw_map, name_map = jira_maps[project], raw_maps[project], name_maps[project]
        sp_ev = [e for e in ev if _field_kind(e.field) == "sprint"]
        sp_times = [e.change_date.value for e in sp_ev]
        sp_from = [_parse_change_sprint_set(e.from_value, e.from_string, jira_map=jira_map, name_map=name_map)[0]
                   for e in sp_ev]
        if sp_ev:
            terminal = _parse_change_sprint_set(sp_ev[-1].to_value, sp_ev[-1].to_string,
                                                jira_map=jira_map, name_map=name_map)[0]
            candidates = set(terminal)
            for e in sp_ev:
                for val, sv in ((e.from_value, e.from_string), (e.to_value, e.to_string)):
                    candidates |= _parse_change_sprint_set(val, sv, jira_map=jira_map, name_map=name_map)[0]
        else:
            terminal = set()
            if not pd.isna(issue.raw_sprint_id):
                key = str(int(float(issue.raw_sprint_id)))
                if key in raw_map:
                    terminal = {raw_map[key]}
            candidates = set(terminal)
        candidates &= ids
        if not candidates:
            continue

        def members(t):
            i = bisect_right(sp_times, t)
            return sp_from[i] if i < len(sp_times) else terminal

        created = None if pd.isna(issue.creation_date) else issue.creation_date.value
        resolved = None if pd.isna(issue.resolution_date) else issue.resolution_date.value

        for sid in candidates:
            t0 = start_ns[sid]
            if resolved is not None and resolved < t0:
                continue  # finished before the sprint began: never part of its work
            in0 = sid in members(t0) and (created is None or created <= t0)
            for k, tk in enumerate(times[sid]):
                if sid in members(tk) and (created is None or created <= tk):
                    a = acc[sid][k]
                    a[0] += 1
                    if resolved is not None and resolved <= tk:
                        a[1] += 1
                    if in0:
                        a[2] += 1
                    else:
                        a[3] += 1

    rows = []
    for r in cohort.itertuples(index=False):
        t = times[r.sprint_id]
        for k, tk in enumerate(t):
            scope, done, com_in, add_in = acc[r.sprint_id][k]
            rows.append({
                "sprint_id": r.sprint_id, "project": r.project, "start_date": r.start_date,
                "split": side[r.sprint_id], "day": k, "elapsed_days": (tk - t[0]) / DAY_NS,
                "n_days": (t[-1] - t[0]) / DAY_NS, "scope_n": scope, "done_n": done,
                "remaining_n": scope - done, "committed_in_n": com_in, "added_in_n": add_in,
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    d = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT, index=False)
    print(f"{d['sprint_id'].nunique()} sprints, {len(d)} sprint-days -> {OUT}")
