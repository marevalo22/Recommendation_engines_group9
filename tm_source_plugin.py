"""
tm_source.py
Pluggable Transfermarkt adapter for gather_data.py.

Background
----------
The original task spec assumes `soccerdata.TransferMarkt`, but that scraper has
been removed from the soccerdata package (absent in 1.9.0, the current release).
To keep the pipeline decoupled from that deprecation, gather_data.py calls this
adapter instead.

Flip ENABLED to True once you have filled in the three functions below with a
working Transfermarkt source. Until then, run_transfermarkt() will record a
clean "adapter_disabled" entry in the manifest and move on.

Three options for implementing this adapter
-------------------------------------------
1. ScraperFC (pip install ScraperFC). Has a Transfermarkt module that scrapes
   the public site. Lowest-friction, no API key needed, but subject to anti-bot
   measures. Example below.

2. transfermarkt-api (community FastAPI wrapper at
   https://github.com/felipeall/transfermarkt-api). Self-host it or point at a
   deployed instance. Clean JSON responses.

3. RapidAPI Transfermarkt endpoints. Paid. Stable.

The return contract matters more than the source you pick: each function must
return a pandas.DataFrame preserving every field the source provides, with no
filtering or renaming. gather_data.py will write it straight to Parquet.
"""

from __future__ import annotations

from typing import Iterable
import pandas as pd

# Set this to True once the three functions below are implemented.
ENABLED = False


def read_players(league: str, season: str) -> pd.DataFrame:
    """
    Player-season snapshot for one (league, season).

    Must include at minimum: player_id, player name, DOB, position,
    market value, club, minutes, appearances — plus anything else the source
    exposes. Preserve every column.

    Parameters
    ----------
    league : str
        soccerdata-style key, e.g. "ENG-Premier League".
    season : str
        Season string in the form "2022-23".
    """
    raise NotImplementedError(
        "Implement read_players using ScraperFC, transfermarkt-api, or another "
        "source. See the docstring at the top of tm_source.py."
    )


def read_transfers(league: str, season: str, window: str) -> pd.DataFrame:
    """
    Transfers for the window leading into this season.

    Parameters
    ----------
    window : str
        Either "summer" or "winter".
    """
    raise NotImplementedError(
        "Implement read_transfers for the summer and winter windows."
    )


def read_market_values(player_ids: Iterable[str]) -> pd.DataFrame:
    """
    Market-value history for a batch of players.

    gather_data.py batches player_ids in groups of 100. The returned frame
    should contain one row per (player_id, valuation_date) with the value,
    currency, and the club at that date — plus anything else the source
    provides.
    """
    raise NotImplementedError(
        "Implement read_market_values as a batched historical pull."
    )


# ---------------------------------------------------------------------------
# Reference sketch using ScraperFC. Uncomment, fill in, and set ENABLED=True.
# ---------------------------------------------------------------------------
#
# from ScraperFC import Transfermarkt
# _tm = Transfermarkt()
#
# _LEAGUE_TO_TM = {
#     "ENG-Premier League":  "Premier League",
#     "ESP-La Liga":         "La Liga",
#     "ITA-Serie A":         "Serie A",
#     "GER-Bundesliga":      "Bundesliga",
#     "FRA-Ligue 1":         "Ligue 1",
#     "ENG-Championship":    "Championship",
#     "NED-Eredivisie":      "Eredivisie",
#     "POR-Primeira Liga":   "Primeira Liga",
#     "BEL-Pro League":      "Pro League",
# }
#
# def _season_start_year(season: str) -> int:
#     return int(season.split("-")[0])
#
# def read_players(league, season):
#     return _tm.scrape_players(
#         year=_season_start_year(season),
#         league=_LEAGUE_TO_TM[league],
#     )
#
# def read_transfers(league, season, window):
#     return _tm.scrape_transfers(
#         year=_season_start_year(season),
#         league=_LEAGUE_TO_TM[league],
#         window=window,
#     )
#
# def read_market_values(player_ids):
#     frames = []
#     for pid in player_ids:
#         try:
#             frames.append(_tm.scrape_player_market_value_history(pid))
#         except Exception:
#             continue
#     return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
