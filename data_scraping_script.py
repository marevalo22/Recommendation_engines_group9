"""
gather_data.py
Raw data gathering for the football recommender-system project.

Pulls FBref, Transfermarkt (via a pluggable adapter), and StatsBomb open data
into data/raw/ as partitioned Parquet files, plus a manifest.json.

NO preprocessing: no filtering, renaming, joining, imputation, or type coercion
beyond what each upstream library returns.

Usage
-----
    python gather_data.py                 # run everything available
    python gather_data.py --skip-tm       # skip Transfermarkt
    python gather_data.py --skip-fbref    # skip FBref
    python gather_data.py --skip-sb       # skip StatsBomb
    python gather_data.py --sources fbref # run only one source
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

# ----------------------------------------------------------------------------
# Paths and configuration
# ----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data" / "raw"

FBREF_DIR       = DATA_ROOT / "fbref"
FBREF_PLAYER    = FBREF_DIR / "player_season_stats"
FBREF_TEAM      = FBREF_DIR / "team_season_stats"
FBREF_SCHEDULE  = FBREF_DIR / "schedule"

TM_DIR          = DATA_ROOT / "transfermarkt"
TM_PLAYER       = TM_DIR / "player_season"
TM_TRANSFERS    = TM_DIR / "transfers"
TM_MARKET       = TM_DIR / "market_values"

SB_DIR          = DATA_ROOT / "statsbomb"
SB_EVENTS       = SB_DIR / "events"
SB_LINEUPS      = SB_DIR / "lineups"
SB_MATCHES      = SB_DIR / "matches"

MANIFEST_PATH   = DATA_ROOT / "manifest.json"

# Nine leagues, as spec'd. soccerdata league keys use the "COUNTRY-Competition" form.
LEAGUES = [
    "ENG-Premier League",
    "ESP-La Liga",
    "ITA-Serie A",
    "GER-Bundesliga",
    "FRA-Ligue 1",
    "ENG-Championship",
    "NED-Eredivisie",
    "POR-Primeira Liga",
    "BEL-Pro League",
]

# Six seasons, as spec'd. soccerdata accepts YYYY-YY strings for European football.
SEASONS = ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]

# soccerdata 1.9.0 public API exposes only these five stat_types via
# read_player_season_stats / read_team_season_stats.
# (The original spec also lists passing, passing_types, goal_shot_creation,
# defense, possession, keeper_adv — the library no longer surfaces those.)
FBREF_STAT_TYPES = ["standard", "keeper", "shooting", "playing_time", "misc"]

# Transfer windows the spec asks for.
TM_WINDOWS = ["summer", "winter"]

# Batch size for market-value history pulls.
TM_MARKET_BATCH = 100

# Rate-limit retry knobs.
RATE_LIMIT_SLEEP = 60
RATE_LIMIT_RETRIES = 3

# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gather")


# ----------------------------------------------------------------------------
# Manifest tracking
# ----------------------------------------------------------------------------

@dataclass
class Manifest:
    run_started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    run_finished_at: str | None = None
    package_versions: dict[str, str] = field(default_factory=dict)
    files: dict[str, list[dict[str, Any]]] = field(default_factory=lambda: {
        "fbref": [], "transfermarkt": [], "statsbomb": []
    })
    failures: list[dict[str, Any]] = field(default_factory=list)

    def record_file(self, source: str, path: Path, rows: int) -> None:
        self.files[source].append({
            "path": str(path.relative_to(ROOT)),
            "rows": int(rows),
            "bytes": int(path.stat().st_size),
        })

    def record_failure(self, source: str, key: dict[str, Any], exc: BaseException) -> None:
        self.failures.append({
            "source": source,
            "key": key,
            "exception": type(exc).__name__,
            "message": str(exc),
        })

    def write(self, path: Path) -> None:
        self.run_finished_at = datetime.now(timezone.utc).isoformat()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.__dict__, indent=2, default=str))


def _pkg_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not_installed"


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def ensure_dirs() -> None:
    for d in (
        FBREF_PLAYER, FBREF_TEAM, FBREF_SCHEDULE,
        TM_PLAYER, TM_TRANSFERS, TM_MARKET,
        SB_EVENTS, SB_LINEUPS, SB_MATCHES,
    ):
        d.mkdir(parents=True, exist_ok=True)


def safe_name(s: str) -> str:
    """Convert a league/season label into a filesystem-safe slug."""
    return s.replace(" ", "_").replace("/", "_")


def with_rate_limit_retry(fn: Callable[[], Any], label: str) -> Any:
    """Call fn(); on HTTP 429 sleep 60s and retry up to RATE_LIMIT_RETRIES times."""
    last_exc: BaseException | None = None
    for attempt in range(1, RATE_LIMIT_RETRIES + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            if "429" in msg or "Too Many Requests" in msg:
                log.warning("%s: rate-limited (attempt %d/%d), sleeping %ds",
                            label, attempt, RATE_LIMIT_RETRIES, RATE_LIMIT_SLEEP)
                time.sleep(RATE_LIMIT_SLEEP)
                last_exc = exc
                continue
            raise
    if last_exc:
        raise last_exc


def write_parquet(df, path: Path) -> int:
    """Write a DataFrame to Parquet preserving every column; returns row count.

    Flattens MultiIndex columns by joining levels with '__' because Parquet
    cannot store tuple column names. This is a format constraint, not a
    transformation: every column and every value is preserved.
    """
    import pandas as pd
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = ["__".join(str(c) for c in tup).strip("_") for tup in out.columns]
    if isinstance(out.index, pd.MultiIndex) or out.index.name is not None:
        out = out.reset_index()
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, engine="pyarrow", index=False)
    return len(out)


# ----------------------------------------------------------------------------
# FBref
# ----------------------------------------------------------------------------

def run_fbref(manifest: Manifest) -> None:
    import soccerdata as sd

    log.info("FBref: %d leagues x %d seasons x %d stat_types",
             len(LEAGUES), len(SEASONS), len(FBREF_STAT_TYPES))

    for league in LEAGUES:
        for season in SEASONS:
            try:
                fb = sd.FBref(leagues=league, seasons=season, no_cache=False, no_store=False)
            except Exception as exc:  # noqa: BLE001
                manifest.record_failure("fbref", {"league": league, "season": season,
                                                  "stage": "init"}, exc)
                log.error("FBref init failed for %s %s: %s", league, season, exc)
                continue

            # Schedule
            sched_path = FBREF_SCHEDULE / f"{safe_name(league)}_{season}.parquet"
            try:
                df = with_rate_limit_retry(lambda: fb.read_schedule(),
                                           f"schedule {league} {season}")
                rows = write_parquet(df, sched_path)
                manifest.record_file("fbref", sched_path, rows)
                log.info("  schedule %s %s -> %d rows", league, season, rows)
            except Exception as exc:  # noqa: BLE001
                manifest.record_failure("fbref", {"league": league, "season": season,
                                                  "stat_type": "schedule"}, exc)
                log.error("  schedule %s %s FAILED: %s", league, season, exc)

            # Player season stats
            for stat_type in FBREF_STAT_TYPES:
                out = FBREF_PLAYER / f"{safe_name(league)}_{season}_{stat_type}.parquet"
                try:
                    df = with_rate_limit_retry(
                        lambda s=stat_type: fb.read_player_season_stats(stat_type=s),
                        f"player {league} {season} {stat_type}")
                    rows = write_parquet(df, out)
                    manifest.record_file("fbref", out, rows)
                    log.info("  player %s %s %s -> %d rows", league, season, stat_type, rows)
                except Exception as exc:  # noqa: BLE001
                    manifest.record_failure("fbref",
                        {"league": league, "season": season, "kind": "player",
                         "stat_type": stat_type}, exc)
                    log.error("  player %s %s %s FAILED: %s", league, season, stat_type, exc)

            # Team season stats
            for stat_type in FBREF_STAT_TYPES:
                out = FBREF_TEAM / f"{safe_name(league)}_{season}_{stat_type}.parquet"
                try:
                    df = with_rate_limit_retry(
                        lambda s=stat_type: fb.read_team_season_stats(stat_type=s),
                        f"team {league} {season} {stat_type}")
                    rows = write_parquet(df, out)
                    manifest.record_file("fbref", out, rows)
                    log.info("  team   %s %s %s -> %d rows", league, season, stat_type, rows)
                except Exception as exc:  # noqa: BLE001
                    manifest.record_failure("fbref",
                        {"league": league, "season": season, "kind": "team",
                         "stat_type": stat_type}, exc)
                    log.error("  team   %s %s %s FAILED: %s", league, season, stat_type, exc)


# ----------------------------------------------------------------------------
# Transfermarkt
# ----------------------------------------------------------------------------

def run_transfermarkt(manifest: Manifest) -> None:
    """
    soccerdata 1.9.0 no longer ships a TransferMarkt scraper, so this function
    delegates to a pluggable adapter defined in tm_source.py. The adapter must
    expose three callables:

        read_players(league, season)   -> pandas.DataFrame
        read_transfers(league, season, window) -> pandas.DataFrame
        read_market_values(player_ids) -> pandas.DataFrame

    If tm_source.py returns None (the default stub), this source is skipped and
    the gap is logged to the manifest. See tm_source.py for implementation
    guidance.
    """
    try:
        import tm_source  # type: ignore
    except ImportError as exc:
        manifest.record_failure("transfermarkt",
            {"stage": "import_adapter"}, exc)
        log.error("Transfermarkt adapter not found (tm_source.py). Skipping.")
        return

    if not getattr(tm_source, "ENABLED", False):
        manifest.record_failure("transfermarkt",
            {"stage": "adapter_disabled"},
            RuntimeError("tm_source.ENABLED is False. Configure tm_source.py to "
                         "point at a live Transfermarkt source before re-running."))
        log.warning("tm_source.ENABLED is False. Skipping Transfermarkt entirely.")
        return

    # Player-season snapshots
    all_player_ids: set[str] = set()
    for league in LEAGUES:
        for season in SEASONS:
            out = TM_PLAYER / f"{safe_name(league)}_{season}.parquet"
            try:
                df = with_rate_limit_retry(
                    lambda: tm_source.read_players(league, season),
                    f"tm players {league} {season}")
                rows = write_parquet(df, out)
                manifest.record_file("transfermarkt", out, rows)
                log.info("  tm player %s %s -> %d rows", league, season, rows)
                if "player_id" in df.columns:
                    all_player_ids.update(map(str, df["player_id"].dropna().unique().tolist()))
            except Exception as exc:  # noqa: BLE001
                manifest.record_failure("transfermarkt",
                    {"league": league, "season": season, "kind": "players"}, exc)
                log.error("  tm player %s %s FAILED: %s", league, season, exc)

    # Transfers per window
    for league in LEAGUES:
        for season in SEASONS:
            for window in TM_WINDOWS:
                out = TM_TRANSFERS / f"{safe_name(league)}_{season}_{window}.parquet"
                try:
                    df = with_rate_limit_retry(
                        lambda w=window: tm_source.read_transfers(league, season, w),
                        f"tm transfers {league} {season} {window}")
                    rows = write_parquet(df, out)
                    manifest.record_file("transfermarkt", out, rows)
                    log.info("  tm transf %s %s %s -> %d rows",
                             league, season, window, rows)
                except Exception as exc:  # noqa: BLE001
                    manifest.record_failure("transfermarkt",
                        {"league": league, "season": season, "window": window,
                         "kind": "transfers"}, exc)
                    log.error("  tm transf %s %s %s FAILED: %s",
                              league, season, window, exc)

    # Market value histories, batched
    player_ids = sorted(all_player_ids)
    log.info("tm market values: %d unique player_ids to fetch", len(player_ids))
    for i in range(0, len(player_ids), TM_MARKET_BATCH):
        batch = player_ids[i:i + TM_MARKET_BATCH]
        batch_no = i // TM_MARKET_BATCH
        out = TM_MARKET / f"batch_{batch_no:03d}.parquet"
        try:
            df = with_rate_limit_retry(
                lambda b=batch: tm_source.read_market_values(b),
                f"tm market batch_{batch_no:03d}")
            rows = write_parquet(df, out)
            manifest.record_file("transfermarkt", out, rows)
            log.info("  tm market batch_%03d (%d players) -> %d rows",
                     batch_no, len(batch), rows)
        except Exception as exc:  # noqa: BLE001
            manifest.record_failure("transfermarkt",
                {"batch": batch_no, "size": len(batch), "kind": "market_values"}, exc)
            log.error("  tm market batch_%03d FAILED: %s", batch_no, exc)


# ----------------------------------------------------------------------------
# StatsBomb (open data, validation-only)
# ----------------------------------------------------------------------------

def run_statsbomb(manifest: Manifest) -> None:
    from statsbombpy import sb

    # Competitions
    comps_path = SB_DIR / "competitions.parquet"
    try:
        comps = sb.competitions()
        rows = write_parquet(comps, comps_path)
        manifest.record_file("statsbomb", comps_path, rows)
        log.info("statsbomb competitions -> %d rows", rows)
    except Exception as exc:  # noqa: BLE001
        manifest.record_failure("statsbomb", {"stage": "competitions"}, exc)
        log.error("statsbomb competitions FAILED: %s", exc)
        return

    # Matches per (competition, season) for every row statsbombpy returns.
    # statsbombpy only exposes open data, so "where data_version is open" is
    # effectively every row here. If an 'available' or similar flag is present
    # we filter on it to be safe.
    import pandas as pd
    filt = comps
    for candidate in ("match_available", "match_available_360"):
        if candidate in comps.columns:
            # keep rows where this field is non-null / truthy
            filt = filt[comps[candidate].notna()]
            break

    keys = filt[["competition_id", "season_id"]].drop_duplicates().to_records(index=False)
    log.info("statsbomb matches: %d competition-season pairs", len(keys))
    for cid, sid in keys:
        out = SB_MATCHES / f"comp_{cid}_season_{sid}.parquet"
        try:
            df = sb.matches(competition_id=int(cid), season_id=int(sid))
            rows = write_parquet(df, out)
            manifest.record_file("statsbomb", out, rows)
            log.info("  matches cid=%s sid=%s -> %d rows", cid, sid, rows)
        except Exception as exc:  # noqa: BLE001
            manifest.record_failure("statsbomb",
                {"competition_id": int(cid), "season_id": int(sid)}, exc)
            log.error("  matches cid=%s sid=%s FAILED: %s", cid, sid, exc)

    # events/ and lineups/ remain empty by design — second explicit task.


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Raw football data pull.")
    p.add_argument("--skip-fbref", action="store_true")
    p.add_argument("--skip-tm",    action="store_true")
    p.add_argument("--skip-sb",    action="store_true")
    p.add_argument("--sources", nargs="+",
                   choices=["fbref", "tm", "sb"],
                   help="Restrict to these sources (overrides --skip-*).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dirs()
    manifest = Manifest(package_versions={
        "soccerdata":  _pkg_version("soccerdata"),
        "statsbombpy": _pkg_version("statsbombpy"),
        "pandas":      _pkg_version("pandas"),
        "pyarrow":     _pkg_version("pyarrow"),
    })

    if args.sources:
        run_set = set(args.sources)
    else:
        run_set = {"fbref", "tm", "sb"}
        if args.skip_fbref: run_set.discard("fbref")
        if args.skip_tm:    run_set.discard("tm")
        if args.skip_sb:    run_set.discard("sb")

    log.info("Run plan: %s", sorted(run_set))
    log.info("Package versions: %s", manifest.package_versions)

    # Make sure tm_source is importable from the script's own directory.
    sys.path.insert(0, str(ROOT))

    try:
        if "fbref" in run_set:
            run_fbref(manifest)
        if "tm" in run_set:
            run_transfermarkt(manifest)
        if "sb" in run_set:
            run_statsbomb(manifest)
    except KeyboardInterrupt:
        log.warning("Interrupted by user. Writing partial manifest.")
    except Exception as exc:  # noqa: BLE001
        log.exception("Unexpected top-level failure: %s", exc)
        manifest.record_failure("runner", {"stage": "top_level"}, exc)
    finally:
        manifest.write(MANIFEST_PATH)
        log.info("Wrote manifest to %s", MANIFEST_PATH)
        log.info("Files produced: fbref=%d, transfermarkt=%d, statsbomb=%d",
                 len(manifest.files["fbref"]),
                 len(manifest.files["transfermarkt"]),
                 len(manifest.files["statsbomb"]))
        log.info("Failures recorded: %d", len(manifest.failures))

    return 0 if not manifest.failures else 0  # failures are logged, not fatal


if __name__ == "__main__":
    raise SystemExit(main())
