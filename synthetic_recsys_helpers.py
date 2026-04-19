
"""Helper functions for the football transfer recommender synthetic augmentation layer.

This file was generated as part of an augmented proof-of-concept workflow.
It keeps the real-only benchmark separate from the synthetic / label-informed scenario track.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans

SEED = 42
TRAIN_SEASONS = ['2018-19', '2019-20', '2020-21', '2021-22']
VAL_SEASON = '2022-23'
TEST_SEASON = '2023-24'
PREV_SEASON = {
    '2019-20': '2018-19',
    '2020-21': '2019-20',
    '2021-22': '2020-21',
    '2022-23': '2021-22',
    '2023-24': '2022-23',
}
OUTFIELD_FEATURES = ['standard__Per 90 Minutes__Gls', 'standard__Per 90 Minutes__Ast', 'standard__Per 90 Minutes__G-PK', 'shooting__Standard__Sh/90', 'shooting__Standard__SoT/90', 'shooting__Standard__G/Sh', 'shooting__Standard__SoT%', 'misc__Performance__Int_p90', 'misc__Performance__TklW_p90', 'misc__Performance__Fls_p90', 'misc__Performance__Fld_p90', 'misc__Performance__Crs_p90', 'misc__Performance__CrdY_p90', 'misc__Performance__CrdR_p90', 'misc__Performance__Off_p90', 'playing_time__Team Success__PPM', 'playing_time__Team Success__+/-90', 'playing_time__Team Success__On-Off', 'playing_time__Playing Time__Min%']
GK_FEATURES = ['keeper__Performance__GA90', 'keeper__Performance__Save%', 'keeper__Performance__CS%', 'keeper__Performance__SoTA', 'keeper__Performance__Saves', 'keeper__Penalty Kicks__Save%']
STYLE_FEATS_RAW = [('standard__Poss', None), ('standard__Per 90 Minutes__Gls', None), ('standard__Per 90 Minutes__G+A', None), ('shooting__Standard__Sh/90', None), ('shooting__Standard__SoT%', None), ('misc__Performance__Crs', 'p90'), ('misc__Performance__Int', 'p90'), ('misc__Performance__TklW', 'p90'), ('misc__Performance__Fls', 'p90'), ('misc__Performance__CrdY', 'p90')]
ROLE_PROFILE = {'central_forward': {'attack': 1.0, 'create': 0.35, 'defend': 0.1, 'cross': 0.1, 'carry': 0.55, 'aerial': 0.75, 'progress': 0.35, 'receive': 1.0}, 'wide_creator_forward': {'attack': 0.6, 'create': 0.9, 'defend': 0.25, 'cross': 0.85, 'carry': 0.95, 'aerial': 0.25, 'progress': 0.7, 'receive': 0.85}, 'all_action_forward': {'attack': 0.85, 'create': 0.55, 'defend': 0.25, 'cross': 0.35, 'carry': 0.75, 'aerial': 0.5, 'progress': 0.55, 'receive': 0.9}, 'advanced_creator': {'attack': 0.35, 'create': 0.95, 'defend': 0.4, 'cross': 0.55, 'carry': 0.45, 'aerial': 0.15, 'progress': 0.9, 'receive': 0.6}, 'controller_midfielder': {'attack': 0.2, 'create': 0.7, 'defend': 0.55, 'cross': 0.2, 'carry': 0.25, 'aerial': 0.2, 'progress': 0.95, 'receive': 0.65}, 'ball_winning_midfielder': {'attack': 0.15, 'create': 0.35, 'defend': 0.95, 'cross': 0.15, 'carry': 0.25, 'aerial': 0.4, 'progress': 0.5, 'receive': 0.55}, 'attacking_full_back': {'attack': 0.2, 'create': 0.65, 'defend': 0.65, 'cross': 1.0, 'carry': 0.8, 'aerial': 0.25, 'progress': 0.75, 'receive': 0.65}, 'front_foot_defender': {'attack': 0.08, 'create': 0.25, 'defend': 0.9, 'cross': 0.1, 'carry': 0.2, 'aerial': 0.75, 'progress': 0.45, 'receive': 0.35}, 'centre_back': {'attack': 0.05, 'create': 0.1, 'defend': 0.85, 'cross': 0.05, 'carry': 0.1, 'aerial': 0.9, 'progress': 0.3, 'receive': 0.25}, 'goalkeeper': {'attack': 0.01, 'create': 0.05, 'defend': 0.7, 'cross': 0.0, 'carry': 0.02, 'aerial': 0.6, 'progress': 0.2, 'receive': 0.1}}
ROLE_BONUS = {'central_forward': {'finishing': 0.35, 'offball': 0.3, 'aerial': 0.2}, 'wide_creator_forward': {'creativity': 0.35, 'carrying': 0.35, 'progression': 0.2, 'offball': 0.15}, 'all_action_forward': {'finishing': 0.15, 'creativity': 0.1, 'carrying': 0.2, 'offball': 0.2}, 'advanced_creator': {'creativity': 0.35, 'progression': 0.3, 'press_resistance': 0.2}, 'controller_midfielder': {'progression': 0.25, 'press_resistance': 0.3, 'reliability': 0.15}, 'ball_winning_midfielder': {'defensive': 0.35, 'reliability': 0.1, 'aerial': 0.1}, 'attacking_full_back': {'carrying': 0.25, 'creativity': 0.15, 'progression': 0.2, 'defensive': 0.1}, 'front_foot_defender': {'defensive': 0.25, 'aerial': 0.2, 'progression': 0.1}, 'centre_back': {'defensive': 0.25, 'aerial': 0.3, 'reliability': 0.15}, 'goalkeeper': {'reliability': 0.2}}
SYNTH_PLAYER_VECTOR_NUMERIC = ['syn_xg_p90', 'syn_xa_p90', 'syn_shot_creating_actions_p90', 'syn_key_passes_p90', 'syn_progressive_passes_p90', 'syn_progressive_carries_p90', 'syn_successful_crosses_p90', 'syn_touches_in_box_p90', 'syn_off_ball_runs_p90', 'syn_pass_completion_pct', 'syn_line_breaking_passes_p90', 'syn_tackles_won_p90', 'syn_interceptions_p90', 'syn_aerial_duels_won_p90', 'syn_recoveries_p90', 'syn_pressures_applied_p90', 'syn_ball_retention_pct', 'syn_availability_pct', 'syn_trait_finishing', 'syn_trait_creativity', 'syn_trait_progression', 'syn_trait_ball_carrying', 'syn_trait_defensive_intensity', 'syn_trait_press_resistance', 'syn_trait_aerial_physicality', 'syn_trait_offball_threat', 'syn_trait_upside', 'syn_gk_shot_stopping', 'syn_gk_distribution', 'syn_gk_sweeper_actions', 'syn_gk_cross_claims']
TEAM_ARCHETYPE_MAP = {
    0: 'aggressive_vertical_press',
    1: 'patient_possession_control',
    2: 'reactive_midblock_transition',
    3: 'dominant_wide_control',
}
train_pop = {}


def season_z(df: pd.DataFrame, col: str) -> pd.Series:
    grp = df.groupby('season_label')[col]
    return grp.transform(lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) > 0 else 0).fillna(0)




class SimpleImplicitALS:
    def __init__(self, factors=32, regularization=0.1, iterations=12, random_state=42):
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.random_state = random_state

    def fit(self, Cui):
        Cui = Cui.tocsr().astype(np.float64)
        self.Cui = Cui
        users, items = Cui.shape
        rng = np.random.default_rng(self.random_state)
        self.user_factors = rng.normal(scale=0.01, size=(users, self.factors))
        self.item_factors = rng.normal(scale=0.01, size=(items, self.factors))
        Cui_csc = Cui.tocsc()
        I = np.eye(self.factors)

        for _ in range(self.iterations):
            YtY = self.item_factors.T @ self.item_factors
            for u in range(users):
                start, end = Cui.indptr[u], Cui.indptr[u + 1]
                idx = Cui.indices[start:end]
                data = Cui.data[start:end]
                if len(idx) == 0:
                    A = YtY + self.regularization * I
                    b = np.zeros(self.factors)
                else:
                    Y = self.item_factors[idx]
                    CuI = data
                    A = YtY + (Y.T * CuI) @ Y + self.regularization * I
                    b = (Y.T * (1.0 + CuI)) @ np.ones(len(idx))
                self.user_factors[u] = np.linalg.solve(A, b)

            XtX = self.user_factors.T @ self.user_factors
            for i in range(items):
                start, end = Cui_csc.indptr[i], Cui_csc.indptr[i + 1]
                idx = Cui_csc.indices[start:end]
                data = Cui_csc.data[start:end]
                if len(idx) == 0:
                    A = XtX + self.regularization * I
                    b = np.zeros(self.factors)
                else:
                    X = self.user_factors[idx]
                    CiI = data
                    A = XtX + (X.T * CiI) @ X + self.regularization * I
                    b = (X.T * (1.0 + CiI)) @ np.ones(len(idx))
                self.item_factors[i] = np.linalg.solve(A, b)
        return self

    def recommend_all_scores_for_user(self, user_idx):
        return self.item_factors @ self.user_factors[user_idx]


def primary_position(p):
    if pd.isna(p):
        return "UNK"
    return str(p).split(",")[0].strip()


def group_median_impute(df, feats, group_cols):
    df = df.copy()
    for f in feats:
        df[f] = df.groupby(group_cols)[f].transform(lambda s: s.fillna(s.median()))
        df[f] = df[f].fillna(df[f].median())
    return df


def zscore_within(df, feats, group_cols, suffix="_z"):
    df = df.copy()
    for f in feats:
        g = df.groupby(group_cols)[f]
        mu = g.transform("mean")
        sd = g.transform("std").replace(0, 1e-6)
        df[f+suffix] = (df[f] - mu) / sd
    return df


def preprocess_players(players):
    players = players.copy()
    players["pos_primary"] = players["pos"].apply(primary_position)
    players["pos_family"] = players["pos_primary"].map({"GK":"GK","DF":"DF","MF":"MF","FW":"FW"}).fillna("UNK")
    outfield_mask = players["pos_family"].isin(["DF","MF","FW"]) & (players["minutes_played"] >= 600)
    gk_mask = (players["pos_family"] == "GK") & (players["minutes_played"] >= 900)
    players_f = players[outfield_mask | gk_mask].copy().reset_index(drop=True)
    for c in RAW_COUNTS:
        players_f[c + "_p90"] = players_f[c] / players_f["nineties"].clip(lower=0.1)
    players_f = group_median_impute(players_f, OUTFIELD_FEATURES, ["league","season_label","pos_family"])
    players_f = group_median_impute(players_f, GK_FEATURES, ["league","season_label"])
    players_f = zscore_within(players_f, OUTFIELD_FEATURES, ["pos_family","league","season_label"])
    players_f = zscore_within(players_f, GK_FEATURES, ["league","season_label"])
    return players_f


def assign_roles(players_f):
    players_f = players_f.copy()
    players_f["role_cluster"] = players_f["pos_family"]
    centroids = {}
    for fam in ["DF","MF","FW"]:
        mask = players_f["pos_family"]==fam
        X = players_f.loc[mask, [f+"_z" for f in OUTFIELD_FEATURES]].fillna(0).values
        km = KMeans(n_clusters=2, random_state=42, n_init=10)
        labs = km.fit_predict(X)
        players_f.loc[mask,"role_cluster"] = [f"{fam}_r{lab}" for lab in labs]
        centroids[fam]=pd.DataFrame(km.cluster_centers_, columns=[f+"_z" for f in OUTFIELD_FEATURES])
    # heuristics for interpretable role archetype
    def role_label(row):
        fam=row['pos_family']
        if fam=='GK':
            return 'goalkeeper'
        cr=row.get("misc__Performance__Crs_p90_z",0)
        ast=row.get("standard__Per 90 Minutes__Ast_z",0)
        gpk=row.get("standard__Per 90 Minutes__G-PK_z",0)
        shot=row.get("shooting__Standard__Sh/90_z",0)
        sot=row.get("shooting__Standard__SoT/90_z",0)
        inter=row.get("misc__Performance__Int_p90_z",0)
        tkl=row.get("misc__Performance__TklW_p90_z",0)
        fld=row.get("misc__Performance__Fld_p90_z",0)
        off=row.get("misc__Performance__Off_p90_z",0)
        onoff=row.get("playing_time__Team Success__On-Off_z",0)
        if fam=='DF':
            if cr + ast + fld > 0.6:
                return 'attacking_full_back'
            elif inter + tkl + onoff > 1.0:
                return 'front_foot_defender'
            else:
                return 'centre_back'
        if fam=='MF':
            if ast + cr + fld > inter + tkl + 0.3:
                return 'advanced_creator'
            elif inter + tkl > ast + cr + 0.5:
                return 'ball_winning_midfielder'
            else:
                return 'controller_midfielder'
        if fam=='FW':
            if gpk + shot + sot > ast + cr + 0.5:
                return 'central_forward'
            elif ast + cr + fld > gpk + shot:
                return 'wide_creator_forward'
            else:
                return 'all_action_forward'
    players_f['role_archetype'] = players_f.apply(role_label, axis=1)
    return players_f, centroids


def build_real_feature_vectors(players_f):
    players_f = players_f.copy()
    role_dummies = pd.get_dummies(players_f["role_cluster"], prefix="role").astype(float)
    players_f["age_centred"] = (players_f["age"] - 26) / 5.0
    players_f["age_sq"] = players_f["age_centred"]**2
    z_cols_out = [f+"_z" for f in OUTFIELD_FEATURES]
    feature_cols = z_cols_out + list(role_dummies.columns) + ["age_centred","age_sq"]
    feat_df = pd.concat([
        players_f[["player_key","team","league","season_label","pos_family","role_cluster","role_archetype",
                   "player","age","minutes_played","tm_market_value_eur_resolved"]].reset_index(drop=True),
        players_f[z_cols_out].reset_index(drop=True).fillna(0),
        role_dummies.reset_index(drop=True),
        players_f[["age_centred","age_sq"]].reset_index(drop=True),
    ], axis=1)
    return feat_df, feature_cols


def build_real_team_style(teams):
    teams_f=teams.copy()
    style_feats=[]
    for col, transform in STYLE_FEATS_RAW:
        if transform=="p90":
            new_col=col+"_p90"
            teams_f[new_col]=teams_f[col]/teams_f["misc__90s"].clip(lower=1)
            style_feats.append(new_col)
        else:
            style_feats.append(col)
    for f in style_feats:
        teams_f[f]=teams_f[f].fillna(teams_f[f].median())
        g=teams_f.groupby("season_label")[f]
        teams_f[f+"_z"]=(teams_f[f]-g.transform("mean"))/g.transform("std").replace(0,1e-6)
    return teams_f, style_feats


def l2_normalise(m):
    n=np.linalg.norm(m, axis=1, keepdims=True)
    n[n==0]=1.0
    return m/n


def composite_vectorised(df):
    s = pd.Series(0.0, index=df.index)
    fw = df["pos_family"] == "FW"
    s[fw] = (
        0.40 * df.loc[fw, "standard__Per 90 Minutes__G-PK_z"]
        + 0.20 * df.loc[fw, "standard__Per 90 Minutes__Ast_z"]
        + 0.20 * df.loc[fw, "shooting__Standard__SoT/90_z"]
        + 0.10 * df.loc[fw, "misc__Performance__Fld_p90_z"]
        + 0.10 * df.loc[fw, "playing_time__Team Success__+/-90_z"]
    )
    mf = df["pos_family"] == "MF"
    s[mf] = (
        0.30 * df.loc[mf, "standard__Per 90 Minutes__Ast_z"]
        + 0.20 * df.loc[mf, "standard__Per 90 Minutes__G-PK_z"]
        + 0.20 * df.loc[mf, "misc__Performance__Int_p90_z"]
        + 0.15 * df.loc[mf, "misc__Performance__TklW_p90_z"]
        + 0.15 * df.loc[mf, "playing_time__Team Success__+/-90_z"]
    )
    dfm = df["pos_family"] == "DF"
    s[dfm] = (
        0.35 * df.loc[dfm, "misc__Performance__Int_p90_z"]
        + 0.30 * df.loc[dfm, "misc__Performance__TklW_p90_z"]
        + 0.20 * df.loc[dfm, "playing_time__Team Success__+/-90_z"]
        + 0.15 * df.loc[dfm, "playing_time__Team Success__On-Off_z"]
    )
    gk = df["pos_family"] == "GK"
    s[gk] = (
        0.40 * df.loc[gk, "keeper__Performance__Save%_z"].fillna(0)
        + 0.30 * df.loc[gk, "keeper__Performance__CS%_z"].fillna(0)
        - 0.30 * df.loc[gk, "keeper__Performance__GA90_z"].fillna(0)
    )
    return s.fillna(0)


def build_cf_model_simple(players_f, seasons_list, factors=24, reg=0.1, iters=8):
    df = players_f[players_f["season_label"].isin(seasons_list)]
    mins_agg = df.groupby(["team","player_key"])["minutes_played"].sum().reset_index()
    t_ids = sorted(mins_agg["team"].unique())
    p_ids = sorted(mins_agg["player_key"].unique())
    t_idx = {t: i for i, t in enumerate(t_ids)}
    p_idx = {p: i for i, p in enumerate(p_ids)}
    r = mins_agg["team"].map(t_idx).values
    c = mins_agg["player_key"].map(p_idx).values
    v = np.log1p(mins_agg["minutes_played"].values).astype(np.float64)
    mat = csr_matrix((v, (r, c)), shape=(len(t_ids), len(p_ids)), dtype=np.float64)
    model = SimpleImplicitALS(factors=factors, regularization=reg, iterations=iters, random_state=42).fit(mat)
    return model, mat, t_ids, p_ids, t_idx, p_idx


def build_team_profile_vec(feat_df, feature_cols, team_name, season, pos_family):
    mask = (feat_df["team"]==team_name) & (feat_df["season_label"]==season) & (feat_df["pos_family"]==pos_family)
    squad = feat_df[mask]
    if len(squad)==0:
        return None
    mins=squad["minutes_played"].values
    w = mins / mins.sum() if mins.sum()>0 else np.ones(len(squad))/len(squad)
    v = (squad[feature_cols].values.astype(np.float32).T @ w).astype(np.float32)
    n=np.linalg.norm(v)
    return v if n==0 else v/n


def get_candidates_from_feat(feat_df, target_season):
    prev = PREV_SEASON[target_season]
    cand = feat_df[feat_df["season_label"]==prev].copy()
    cand = cand.sort_values("minutes_played", ascending=False).drop_duplicates("player_key")
    return cand


def content_scores_for_teams(feat_df, feature_cols, target_season, target_teams=None):
    prev = PREV_SEASON[target_season]
    candidates = get_candidates_from_feat(feat_df, target_season)
    if target_teams is None:
        target_teams = sorted(players_f[players_f["season_label"]==target_season]["team"].unique())
    out=[]
    cand_by_fam={fam: candidates[candidates["pos_family"]==fam].copy() for fam in ["DF","MF","FW","GK"]}
    for t in target_teams:
        for fam, cand_fam in cand_by_fam.items():
            profile=build_team_profile_vec(feat_df, feature_cols, t, prev, fam)
            if profile is None or len(cand_fam)==0:
                continue
            V=l2_normalise(cand_fam[feature_cols].values.astype(np.float32))
            sims=V @ profile
            sub=pd.DataFrame({
                "team": t,
                "player_key": cand_fam["player_key"].values,
                "pos_family": fam,
                "cb_score": sims
            })
            out.append(sub)
    if not out:
        return pd.DataFrame(columns=["team","player_key","pos_family","cb_score"])
    return pd.concat(out, ignore_index=True)


def get_cf_scores(target_season, cf_model, cf_team_idx, cf_player_ids):
    target_teams = sorted(players_f[players_f["season_label"]==target_season]["team"].unique())
    rows=[]
    for t in target_teams:
        if t not in cf_team_idx:
            continue
        scores = cf_model.recommend_all_scores_for_user(cf_team_idx[t])
        rows.append(pd.DataFrame({"team":t, "player_key":cf_player_ids, "cf_score":scores}))
    return pd.concat(rows, ignore_index=True)


def get_team_style_vec(teams_f, team, season, style_feats):
    r = teams_f[(teams_f["team"]==team) & (teams_f["season_label"]==season)]
    if len(r)==0:
        return None
    v = np.nan_to_num(r[[f+"_z" for f in style_feats]].values[0]).astype(np.float32)
    n=np.linalg.norm(v)
    return v if n==0 else v/n


def candidate_style_vec(players_f, teams_f, player_key, prev_season, style_feats):
    rows = players_f[(players_f["player_key"]==player_key) & (players_f["season_label"]==prev_season)]
    if len(rows)==0: return None
    top = rows.sort_values("minutes_played", ascending=False).iloc[0]
    return get_team_style_vec(teams_f, top["team"], prev_season, style_feats)


def zscore_within_team(series, team_series):
    out = pd.Series(index=series.index, dtype=float)
    for t in team_series.unique():
        mask = team_series == t
        v = series[mask].values.astype(float)
        mu = np.nanmean(v); sd=np.nanstd(v)
        out.loc[mask] = 0.0 if (sd==0 or np.isnan(sd)) else (v-mu)/sd
    return out


def compute_real_context_features(target_season, ca_df):
    prev = PREV_SEASON[target_season]
    tvecs = {t:get_team_style_vec(teams_f, t, prev, real_style_feats) for t in ca_df["team"].unique()}
    # perhaps expensive to compute candidate style vector for each unique pid
    cvecs={}
    for p in ca_df["player_key"].unique():
        cvecs[p]=candidate_style_vec(players_f, teams_f, p, prev, real_style_feats)
    sf = np.array([float(tvecs[t] @ cvecs[p]) if tvecs[t] is not None and cvecs[p] is not None else 0.0
                   for t,p in zip(ca_df["team"].values, ca_df["player_key"].values)])
    prev_df = players_f[players_f["season_label"]==prev]
    tfm = prev_df.groupby(["team","pos_family"])["minutes_played"].sum().unstack(fill_value=0)
    tfs = tfm.div(tfm.sum(axis=1).replace(0,1), axis=0)
    tfg = 1 - tfs
    gf = np.array([tfg.loc[t,fam] if (t in tfg.index and fam in tfg.columns) else 0.5
                   for t,fam in zip(ca_df["team"].values, ca_df["pos_family"].values)])
    team_band = prev_df.groupby("team")["tm_market_value_eur_resolved"].median().to_dict()
    cand_mv = prev_df.groupby("player_key")["tm_market_value_eur_resolved"].max().to_dict()
    pp=[]
    for t,p in zip(ca_df["team"].values, ca_df["player_key"].values):
        band, mv = team_band.get(t, np.nan), cand_mv.get(p, np.nan)
        if np.isnan(band) or np.isnan(mv) or band==0:
            pp.append(0.0)
        else:
            pp.append(1/(1+np.exp(-(np.log(mv/band) - np.log(3)))))
    pp=np.array(pp)
    team_age = prev_df.groupby("team")["age"].median().to_dict()
    cand_age = prev_df.groupby("player_key")["age"].max().to_dict()
    am = np.array([max(0, abs(cand_age.get(p,26)-team_age.get(t,26))-6)/10
                   for t,p in zip(ca_df["team"].values, ca_df["player_key"].values)])
    return sf, gf, pp, am


def build_real_scores_for_season(target_season, cf_model, cf_team_idx, cf_player_ids, feature_df=None, feature_cols=None):
    if feature_df is None:
        raise ValueError('feature_df must be provided explicitly.')
    if feature_cols is None:
        feature_cols = [c for c in feature_df.columns if c.startswith('f_') or c.startswith('role_') or c in ['age_centered','age_sq']]
    candidates = get_candidates_from_feat(feature_df, target_season)
    cb = content_scores_for_teams(feature_df, feature_cols, target_season)
    cf = get_cf_scores(target_season, cf_model, cf_team_idx, cf_player_ids)
    ca = cb.merge(cf, on=["team","player_key"], how="left").fillna({"cf_score":0.0})
    prev = PREV_SEASON[target_season]
    prev_team_player = set(zip(players_f[players_f["season_label"]==prev]["team"].values,
                               players_f[players_f["season_label"]==prev]["player_key"].values))
    ca = ca[~ca.apply(lambda r: (r["team"], r["player_key"]) in prev_team_player, axis=1)].reset_index(drop=True)
    sf,gf,pp,am = compute_real_context_features(target_season, ca)
    ca["style_fit"]=sf; ca["gap_fit"]=gf; ca["price_penalty"]=pp; ca["age_mismatch"]=am
    ca["cf_z"]=zscore_within_team(ca["cf_score"], ca["team"])
    ca["cb_z"]=zscore_within_team(ca["cb_score"], ca["team"])
    ca["context_score"]=(REAL_WEIGHTS["cf"]*ca["cf_z"] + REAL_WEIGHTS["cb"]*ca["cb_z"] +
                         REAL_WEIGHTS["style"]*ca["style_fit"] + REAL_WEIGHTS["gap"]*ca["gap_fit"] -
                         REAL_WEIGHTS["price"]*ca["price_penalty"] - REAL_WEIGHTS["age"]*ca["age_mismatch"])
    return ca, candidates


def get_transfer_ground_truth(players_f, season):
    arr = players_f[(players_f["season_label"]==season) & (players_f["inferred_transfer_arrival_this_team"]==True)]
    gt = arr.groupby("team")["player_key"].apply(set).to_dict()
    gt_mins = dict(zip(zip(arr["team"], arr["player_key"]), arr["minutes_played"]))
    return gt, gt_mins


def precision_at_k(rec, gt, k):
    if not gt: return np.nan
    return sum(1 for p in rec[:k] if p in gt) / k


def recall_at_k(rec, gt, k):
    if not gt: return np.nan
    return sum(1 for p in rec[:k] if p in gt) / len(gt)


def ndcg_at_k(rec, gt, gt_mins, team, k):
    if not gt: return np.nan
    dcg = sum(np.log1p(gt_mins.get((team,p),0))/np.log2(i+2) for i,p in enumerate(rec[:k]) if p in gt)
    ideal = sorted([np.log1p(gt_mins.get((team,p),0)) for p in gt], reverse=True)[:k]
    idcg = sum(g/np.log2(i+2) for i,g in enumerate(ideal))
    return 0.0 if idcg==0 else dcg/idcg


def rmse_mae(scores_df, score_col, gt, gt_mins):
    preds=[]; actuals=[]
    for team in gt:
        sub = scores_df[scores_df["team"]==team]
        if len(sub)==0: continue
        s = sub[score_col].values.astype(float)
        scaled = np.zeros_like(s) if s.max()==s.min() else (s-s.min())/(s.max()-s.min())
        for i,p in enumerate(sub["player_key"].values):
            label = np.log1p(gt_mins.get((team,p),0)) if p in gt[team] else 0.0
            preds.append(scaled[i]); actuals.append(label)
    preds=np.array(preds); actuals=np.array(actuals)
    if actuals.max()>0:
        actuals = actuals / actuals.max()
    return float(np.sqrt(np.mean((preds-actuals)**2))), float(np.mean(np.abs(preds-actuals)))


def serendipity_at_k(rec, gt, k, pop):
    if not gt: return np.nan
    s=0.0
    for p in rec[:k]:
        rel = 1 if p in gt else 0
        s += rel * (1-pop.get(p,0.5))
    return s/k


def diversity_at_k(rec, k, vector_lookup):
    vs=[vector_lookup[p] for p in rec[:k] if p in vector_lookup]
    if len(vs)<2: return 0.0
    V=np.vstack(vs)
    sim=V@V.T
    n=len(vs)
    avg=(sim.sum()-n)/(n*(n-1))
    return float(1-avg)


def evaluate_model(scores_df, score_col, gt, gt_mins, candidates_df, vector_lookup, k_list=(10,25,50), pop_lookup=None):
    if pop_lookup is None:
        pop_lookup=train_pop
    prec={k:[] for k in k_list}; rec_={k:[] for k in k_list}; ndcg={k:[] for k in k_list}; ser={k:[] for k in k_list}; div={k:[] for k in k_list}
    all_rec_10=set(); teams_with_hit=0
    for team, gt_set in gt.items():
        if not gt_set: continue
        team_scores = scores_df[scores_df["team"]==team].sort_values(score_col, ascending=False)
        rec_list = team_scores["player_key"].tolist()
        all_rec_10.update(rec_list[:10])
        got_hit=False
        for k in k_list:
            p=precision_at_k(rec_list, gt_set, k)
            r=recall_at_k(rec_list, gt_set, k)
            n=ndcg_at_k(rec_list, gt_set, gt_mins, team, k)
            s=serendipity_at_k(rec_list, gt_set, k, pop_lookup)
            d=diversity_at_k(rec_list, k, vector_lookup)
            if not np.isnan(p):
                prec[k].append(p); rec_[k].append(r); ndcg[k].append(n); ser[k].append(s); div[k].append(d)
            if k==50 and any(pp in gt_set for pp in rec_list[:k]):
                got_hit=True
        if got_hit:
            teams_with_hit += 1
    rmse,mae = rmse_mae(scores_df, score_col, gt, gt_mins)
    out={"rmse":rmse, "mae":mae}
    for k in k_list:
        out[f"precision@{k}"]=float(np.mean(prec[k])) if prec[k] else 0.0
        out[f"recall@{k}"]=float(np.mean(rec_[k])) if rec_[k] else 0.0
        out[f"ndcg@{k}"]=float(np.mean(ndcg[k])) if ndcg[k] else 0.0
        out[f"serendipity@{k}"]=float(np.mean(ser[k])) if ser[k] else 0.0
        out[f"diversity@{k}"]=float(np.mean(div[k])) if div[k] else 0.0
    out["coverage_catalogue@10"] = len(all_rec_10) / max(len(candidates_df), 1)
    out["coverage_teams@50"] = teams_with_hit / max(len(gt), 1)
    return out


def sigmoid(x):
    return 1/(1+np.exp(-x))


def build_synthetic_team_features(teams_raw, seed=42):
    rng=np.random.default_rng(seed)
    teams=teams_raw.copy()
    teams["crosses_p90"] = teams["misc__Performance__Crs"] / teams["misc__90s"].clip(lower=1)
    teams["ints_p90"] = teams["misc__Performance__Int"] / teams["misc__90s"].clip(lower=1)
    teams["tklw_p90"] = teams["misc__Performance__TklW"] / teams["misc__90s"].clip(lower=1)
    teams["fls_p90"] = teams["misc__Performance__Fls"] / teams["misc__90s"].clip(lower=1)
    teams["cards_p90"] = teams["misc__Performance__CrdY"] / teams["misc__90s"].clip(lower=1)
    teams["offsides_p90"] = teams["misc__Performance__Off"] / teams["misc__90s"].clip(lower=1)
    teams["goal_diff_p90"] = teams["schedule_goal_difference"] / teams["schedule_matches"].clip(lower=1)
    teams["points_p90"] = teams["schedule_points"] / teams["schedule_matches"].clip(lower=1)
    teams["squad_log_mv"] = np.log1p(teams["squad_market_value_sum_eur"].fillna(teams["squad_market_value_sum_eur"].median()))
    teams["squad_mv_change_pct"] = teams["squad_market_value_change_avg_pct"].fillna(0)
    anchor_cols = [
        "standard__Poss","shooting__Standard__Sh/90","shooting__Standard__SoT%","crosses_p90",
        "ints_p90","tklw_p90","fls_p90","cards_p90","offsides_p90","goal_diff_p90",
        "points_p90","squad_log_mv","keeper__Performance__GA90","squad_mv_change_pct"
    ]
    for c in anchor_cols:
        teams[c] = teams[c].fillna(teams[c].median())
        teams[c+"_z"] = season_z(teams, c)
    n=len(teams)
    eps = lambda s: rng.normal(0,s,n)
    poss = sigmoid(0.90*teams["standard__Poss_z"] + 0.35*teams["squad_log_mv_z"] + 0.25*teams["points_p90_z"] - 0.10*teams["fls_p90_z"] + eps(0.18))
    direct = sigmoid(-0.65*teams["standard__Poss_z"] + 0.45*teams["crosses_p90_z"] + 0.35*teams["offsides_p90_z"] + 0.20*teams["shooting__Standard__Sh/90_z"] + eps(0.22))
    press = sigmoid(0.55*teams["ints_p90_z"] + 0.55*teams["tklw_p90_z"] + 0.20*teams["fls_p90_z"] + 0.20*teams["goal_diff_p90_z"] + eps(0.20))
    width = sigmoid(0.85*teams["crosses_p90_z"] + 0.20*direct - 0.15*poss + eps(0.18))
    tempo = sigmoid(0.55*teams["shooting__Standard__Sh/90_z"] - 0.25*teams["standard__Poss_z"] + 0.25*direct + 0.15*teams["offsides_p90_z"] + eps(0.20))
    territorial = sigmoid(0.55*teams["standard__Poss_z"] + 0.45*teams["goal_diff_p90_z"] + 0.35*teams["points_p90_z"] + 0.30*teams["squad_log_mv_z"] + eps(0.18))
    line = sigmoid(0.45*press + 0.35*territorial - 0.20*teams["keeper__Performance__GA90_z"] + eps(0.18))
    transition = sigmoid(0.65*direct + 0.25*tempo - 0.45*poss + eps(0.18))
    central = sigmoid(0.55*(1-width) + 0.25*poss + 0.15*territorial + eps(0.16))
    development = sigmoid(-0.45*teams["squad_log_mv_z"] + 0.25*teams["squad_mv_change_pct_z"] + 0.15*(1-territorial) + eps(0.18))
    teams["syn_latent_possession_orientation"] = poss
    teams["syn_latent_directness"] = direct
    teams["syn_latent_pressing_intensity"] = press
    teams["syn_latent_width_crossing"] = width
    teams["syn_latent_tempo"] = tempo
    teams["syn_latent_territorial_dominance"] = territorial
    teams["syn_latent_line_aggression"] = line
    teams["syn_latent_transition_dependence"] = transition
    teams["syn_latent_central_combination"] = central
    teams["syn_latent_development_orientation"] = development
    noise = lambda s: rng.normal(0,s,n)
    teams["syn_possession_pct"] = np.clip(38 + 28*poss + noise(1.5), 34, 70)
    teams["syn_passes_per_possession"] = np.clip(3.0 + 6.5*poss + 1.2*central - 1.0*transition + noise(0.4), 2.0, 12.5)
    teams["syn_build_up_speed_index"] = np.clip(25 + 55*direct + 15*tempo - 15*poss + noise(3.0), 5, 95)
    teams["syn_average_pass_length_m"] = np.clip(12.5 + 8.0*direct + 2.5*width - 2.5*poss + noise(0.7), 10.0, 27.0)
    teams["syn_directness_index"] = np.clip(20 + 70*direct + noise(4.0), 5, 98)
    teams["syn_tempo_index"] = np.clip(20 + 70*tempo + noise(4.0), 5, 98)
    teams["syn_width_crossing_tendency"] = np.clip(15 + 75*width + noise(4.0), 5, 98)
    teams["syn_central_combination_tendency"] = np.clip(15 + 75*central + noise(4.0), 5, 98)
    teams["syn_attacking_territory_pct"] = np.clip(30 + 35*territorial + 5*press - 3*transition + noise(1.8), 25, 80)
    teams["syn_final_third_entries_p90"] = np.clip(14 + 18*(0.45*territorial + 0.25*poss + 0.20*tempo + 0.10*width) + noise(1.3), 8, 40)
    teams["syn_box_entries_p90"] = np.clip(4 + 10*(0.40*territorial + 0.25*central + 0.20*width + 0.15*tempo) + noise(0.8), 2, 18)
    teams["syn_transition_attacks_p90"] = np.clip(0.5 + 8*transition + 1.5*tempo + noise(0.8), 0.0, 14)
    teams["syn_shot_creation_actions_p90"] = np.clip(10 + 16*(0.25*territorial + 0.20*tempo + 0.20*width + 0.20*central + 0.15*poss) + noise(1.2), 6, 35)
    teams["syn_xg_per_match"] = np.clip(0.65 + 1.75*(0.35*territorial + 0.20*tempo + 0.20*central + 0.15*width + 0.10*poss) + noise(0.10), 0.35, 3.2)
    teams["syn_set_piece_xg_share"] = np.clip(0.10 + 0.16*(1-central) + 0.08*width + 0.05*direct + noise(0.015), 0.05, 0.42)
    teams["syn_pressing_index"] = np.clip(10 + 85*press + noise(4.0), 5, 98)
    teams["syn_ppda"] = np.clip(22 - 16*press + 3*(1-territorial) + noise(1.0), 4.5, 24.0)
    teams["syn_recovery_height_index"] = np.clip(15 + 75*(0.55*press + 0.45*line) + noise(4.0), 5, 98)
    teams["syn_defensive_line_height_index"] = np.clip(12 + 78*line + noise(3.5), 5, 98)
    teams["syn_compactness_index"] = np.clip(25 + 55*(0.55*press + 0.45*(1-width)) + noise(3.0), 10, 98)
    teams["syn_defensive_block_depth_index"] = np.clip(100 - teams["syn_defensive_line_height_index"] + noise(2.0), 5, 95)
    teams["syn_xga_per_match"] = np.clip(1.95 - 0.95*territorial - 0.25*press + 0.20*transition + noise(0.10), 0.45, 2.60)
    teams["syn_budget_band_index"] = np.clip(20 + 60*sigmoid(teams["squad_log_mv_z"]) + noise(3), 5, 98)
    km_cols = [
        "syn_latent_possession_orientation","syn_latent_directness","syn_latent_pressing_intensity",
        "syn_latent_width_crossing","syn_latent_tempo","syn_latent_territorial_dominance","syn_latent_transition_dependence"
    ]
    km = KMeans(n_clusters=4, random_state=42, n_init=20)
    labels = km.fit_predict(teams[km_cols].values)
    teams["syn_tactical_cluster_id"] = labels
    cent = pd.DataFrame(km.cluster_centers_, columns=km_cols)
    cluster_names={}
    for cid,row in cent.iterrows():
        if row["syn_latent_possession_orientation"] > 0.62 and row["syn_latent_pressing_intensity"] > 0.58:
            name = "dominant_possession_press"
        elif row["syn_latent_width_crossing"] > 0.60 and row["syn_latent_directness"] > 0.50:
            name = "wide_crossing_vertical"
        elif row["syn_latent_directness"] > 0.58 and row["syn_latent_possession_orientation"] < 0.48:
            name = "deep_block_counter"
        else:
            name = "balanced_control"
        cluster_names[cid]=name
    teams["syn_tactical_archetype"] = teams["syn_tactical_cluster_id"].map(cluster_names)
    return teams


def build_synthetic_player_features(players_f, teams_syn, seed=42):
    rng=np.random.default_rng(seed)
    df=players_f.copy()
    merge_cols=["team","season_label","syn_latent_possession_orientation","syn_latent_directness","syn_latent_pressing_intensity",
                "syn_latent_width_crossing","syn_latent_tempo","syn_latent_territorial_dominance","syn_latent_line_aggression",
                "syn_latent_transition_dependence","syn_latent_central_combination","syn_latent_development_orientation",
                "syn_tactical_archetype","syn_budget_band_index","syn_xg_per_match","syn_xga_per_match"]
    df = df.merge(teams_syn[merge_cols], on=["team","season_label"], how="left")
    # market value anchors
    df["log_mv"] = np.log1p(df["tm_market_value_eur_resolved"].fillna(df["tm_market_value_eur_resolved"].median()))
    df["mv_change_pct"] = df["tm_market_value_change_pct_in_season"].replace([np.inf,-np.inf], np.nan).fillna(0)
    for c in ["log_mv","mv_change_pct","tm_valuation_observations_resolved"]:
        if c not in df.columns:
            continue
        df[c+"_z"] = df.groupby(["season_label","pos_family"])[c].transform(lambda s: (s - s.mean())/s.std(ddof=0) if s.std(ddof=0)>0 else 0).fillna(0)
    # age curves
    age=df["age"].fillna(df["born"].map(lambda x: np.nan))
    df["syn_age_attack_curve"] = np.exp(-((df["age"]-26)**2)/(2*4.5**2))
    df["syn_age_def_curve"] = np.exp(-((df["age"]-27)**2)/(2*5.0**2))
    df["syn_youth_upside"] = sigmoid((23 - df["age"]) / 2.5)
    df["syn_maturity"] = sigmoid((df["age"] - 24) / 3.0)
    # role profile maps
    prof = pd.DataFrame(df["role_archetype"].map(ROLE_PROFILE).tolist()).fillna(0).add_prefix("rp_")
    df = pd.concat([df.reset_index(drop=True), prof.reset_index(drop=True)], axis=1)
    # deterministic noise
    n=len(df)
    eps=lambda s: rng.normal(0,s,n)
    bonus = lambda key: df["role_archetype"].map(lambda r: ROLE_BONUS.get(r,{}).get(key,0.0)).fillna(0).values
    # latent traits 0-1
    z = lambda c: df.get(c, pd.Series(0,index=df.index)).fillna(0).values
    finishing = sigmoid(0.55*z("standard__Per 90 Minutes__G-PK_z") + 0.45*z("shooting__Standard__G/Sh_z") + 0.25*z("shooting__Standard__SoT%_z") + 0.15*z("shooting__Standard__SoT/90_z") + 0.12*df["syn_latent_territorial_dominance"].values + bonus("finishing") + eps(0.22))
    shot_quality = sigmoid(0.45*z("shooting__Standard__SoT/90_z") + 0.30*z("shooting__Standard__G/Sh_z") + 0.20*z("misc__Performance__Off_p90_z") + 0.15*df["rp_attack"].values + eps(0.20))
    creativity = sigmoid(0.50*z("standard__Per 90 Minutes__Ast_z") + 0.35*z("misc__Performance__Crs_p90_z") + 0.20*z("misc__Performance__Fld_p90_z") + 0.15*df["syn_latent_possession_orientation"].values + bonus("creativity") + eps(0.22))
    progression = sigmoid(0.35*z("misc__Performance__Crs_p90_z") + 0.25*z("standard__Per 90 Minutes__Ast_z") + 0.20*z("misc__Performance__Fld_p90_z") + 0.20*z("playing_time__Team Success__On-Off_z") + 0.15*df["syn_latent_possession_orientation"].values + bonus("progression") + eps(0.22))
    carrying = sigmoid(0.45*z("misc__Performance__Fld_p90_z") + 0.20*z("misc__Performance__Off_p90_z") + 0.20*z("shooting__Standard__Sh/90_z") + 0.10*df["syn_latent_transition_dependence"].values + bonus("carrying") + eps(0.22))
    defensive = sigmoid(0.55*z("misc__Performance__Int_p90_z") + 0.55*z("misc__Performance__TklW_p90_z") + 0.20*z("misc__Performance__Fls_p90_z") + 0.15*df["syn_latent_pressing_intensity"].values + bonus("defensive") + eps(0.22))
    aerial = sigmoid(0.30*z("shooting__Standard__Sh/90_z") + 0.25*z("standard__Per 90 Minutes__G-PK_z") + 0.25*z("misc__Performance__TklW_p90_z") + 0.20*df["syn_latent_directness"].values + bonus("aerial") + eps(0.24))
    press_res = sigmoid(0.35*z("playing_time__Playing Time__Min%_z") + 0.25*z("playing_time__Team Success__On-Off_z") + 0.25*df["syn_latent_possession_orientation"].values + 0.20*z("standard__Per 90 Minutes__Ast_z") - 0.10*z("misc__Performance__Fls_p90_z") + bonus("press_resistance") + eps(0.20))
    reliability = sigmoid(0.55*z("playing_time__Playing Time__Min%_z") + 0.20*z("playing_time__Team Success__On-Off_z") + 0.20*df["syn_maturity"].values + 0.10*z("log_mv_z") + bonus("reliability") + eps(0.18))
    offball = sigmoid(0.50*z("misc__Performance__Off_p90_z") + 0.35*z("shooting__Standard__Sh/90_z") + 0.25*z("standard__Per 90 Minutes__G-PK_z") + 0.20*df["syn_latent_directness"].values + bonus("offball") + eps(0.22))
    distribution = sigmoid(0.35*creativity + 0.35*progression + 0.30*press_res + 0.10*df["rp_progress"].values + eps(0.15))
    upside = sigmoid(0.40*z("mv_change_pct_z") + 0.30*df["syn_youth_upside"].values + 0.20*reliability + 0.10*shot_quality + eps(0.18))
    availability = np.clip(sigmoid(0.8*z("playing_time__Playing Time__Min%_z") + 0.2*reliability + eps(0.10)), 0.15, 0.99)
    df["syn_trait_finishing"] = finishing
    df["syn_trait_shot_quality"] = shot_quality
    df["syn_trait_creativity"] = creativity
    df["syn_trait_progression"] = progression
    df["syn_trait_ball_carrying"] = carrying
    df["syn_trait_defensive_intensity"] = defensive
    df["syn_trait_aerial_physicality"] = aerial
    df["syn_trait_press_resistance"] = press_res
    df["syn_trait_reliability"] = reliability
    df["syn_trait_offball_threat"] = offball
    df["syn_trait_distribution"] = distribution
    df["syn_trait_upside"] = upside
    df["syn_trait_availability"] = availability
    # feature generation helpers
    def feat_noise(s): return rng.normal(0,s,n)
    atk, cre, dfn, crs, car, air, prog, rec = [df[c].values for c in ["rp_attack","rp_create","rp_defend","rp_cross","rp_carry","rp_aerial","rp_progress","rp_receive"]]
    team_pos = df["syn_latent_possession_orientation"].values
    team_dir = df["syn_latent_directness"].values
    team_prs = df["syn_latent_pressing_intensity"].values
    team_wid = df["syn_latent_width_crossing"].values
    team_tmp = df["syn_latent_tempo"].values
    team_ter = df["syn_latent_territorial_dominance"].values
    team_ctr = df["syn_latent_central_combination"].values
    # outfield observable synthetic features
    npxg = np.clip(atk * (0.02 + 0.78*(0.45*finishing + 0.30*offball + 0.15*shot_quality + 0.10*team_ter)) + feat_noise(0.03), 0.0, 0.95)
    xg = np.clip(npxg + atk * (0.01 + 0.07*sigmoid(z("misc__Performance__Fld_p90_z"))) + feat_noise(0.01), 0.0, 1.05)
    xa = np.clip(cre * (0.03 + 0.55*(0.45*creativity + 0.20*progression + 0.15*distribution + 0.10*team_ter)) + feat_noise(0.03), 0.0, 0.75)
    xgi = np.clip(xg + xa, 0.0, 1.50)
    sca = np.clip(0.5 + 6.5*(0.25*creativity*cre + 0.20*progression*prog + 0.15*carrying*car + 0.20*team_ter + 0.20*team_tmp) + feat_noise(0.4), 0.1, 12.0)
    gca = np.clip(0.08 + 0.45*xgi + 0.08*sca + feat_noise(0.05), 0.0, 1.8)
    key_passes = np.clip(cre * (0.10 + 3.5*(0.50*creativity + 0.20*progression + 0.10*team_ter)) + feat_noise(0.2), 0.0, 5.8)
    prog_passes = np.clip(prog * (0.20 + 7.0*(0.40*progression + 0.25*distribution + 0.15*team_pos)) + feat_noise(0.5), 0.0, 12.5)
    prog_carries = np.clip(car * (0.10 + 6.5*(0.45*carrying + 0.20*offball + 0.15*team_tmp + 0.10*team_dir)) + feat_noise(0.5), 0.0, 10.5)
    prog_received = np.clip(rec * (0.40 + 9.5*(0.40*offball + 0.15*team_ter + 0.15*atk + 0.10*team_dir)) + feat_noise(0.8), 0.0, 16.0)
    p_final3 = np.clip(prog * (0.30 + 8.0*(0.45*progression + 0.20*distribution + 0.15*team_pos)) + feat_noise(0.6), 0.0, 14.0)
    p_box = np.clip(np.maximum(cre, prog*0.6) * (0.05 + 4.2*(0.45*creativity + 0.20*progression + 0.10*team_ter)) + feat_noise(0.25), 0.0, 6.0)
    succ_crosses = np.clip(crs * (0.02 + 2.0*(0.45*creativity + 0.25*distribution + 0.15*team_wid)) + feat_noise(0.15), 0.0, 4.5)
    chances_created = np.clip(0.55*key_passes + 0.35*succ_crosses + 0.08*sca + feat_noise(0.15), 0.0, 7.0)
    shot_acc = np.clip(18 + 55*(0.45*shot_quality + 0.30*finishing + 0.10*atk) + feat_noise(3.5), 15, 78)
    shot_conv = np.clip(4 + 22*(0.50*finishing + 0.25*shot_quality + 0.10*atk) + feat_noise(1.8), 2, 34)
    touches_box = np.clip(atk * (1.0 + 12*(0.40*offball + 0.20*atk + 0.15*team_ter + 0.10*team_dir)) + feat_noise(0.8), 0.2, 18)
    carries_box = np.clip(car * (0.02 + 4.5*(0.50*carrying + 0.20*offball + 0.10*team_tmp)) + feat_noise(0.25), 0.0, 5.5)
    succ_dribbles = np.clip(car * (0.05 + 4.8*(0.50*carrying + 0.15*creativity + 0.10*df["syn_youth_upside"].values)) + feat_noise(0.25), 0.0, 6.5)
    offball_runs = np.clip(rec * (0.3 + 8.5*(0.45*offball + 0.15*atk + 0.15*team_dir)) + feat_noise(0.5), 0.0, 13.0)
    pass_comp = np.clip(62 + 27*(0.40*distribution + 0.25*press_res + 0.15*team_pos + 0.10*reliability) + feat_noise(2.0), 58, 95)
    fwd_pass_comp = np.clip(pass_comp - 4 + 7*(0.35*progression + 0.20*distribution - 0.05*team_dir) + feat_noise(1.5), 50, 92)
    passes_press = np.clip(prog * (0.30 + 6.2*(0.45*press_res + 0.30*distribution + 0.10*team_pos)) + feat_noise(0.4), 0.0, 10.0)
    line_break = np.clip(prog * (0.10 + 5.0*(0.45*progression + 0.25*press_res + 0.10*team_ctr)) + feat_noise(0.3), 0.0, 7.5)
    xt_pass = np.clip(0.03*line_break + 0.05*p_box + 0.03*succ_crosses + feat_noise(0.02), 0.0, 1.0)
    tackles = np.clip(dfn * (0.15 + 4.5*(0.45*defensive + 0.15*reliability + 0.10*team_prs)) + feat_noise(0.3), 0.0, 6.0)
    interceptions = np.clip(dfn * (0.15 + 4.2*(0.50*defensive + 0.15*press_res + 0.10*team_prs)) + feat_noise(0.3), 0.0, 5.5)
    clearances = np.clip(dfn * air * (0.05 + 8.0*(0.45*aerial + 0.20*defensive + 0.10*team_dir)) + feat_noise(0.5), 0.0, 9.5)
    aerial_won = np.clip(air * (0.05 + 6.0*(0.50*aerial + 0.15*reliability + 0.10*team_dir)) + feat_noise(0.35), 0.0, 8.0)
    recoveries = np.clip(dfn * (0.40 + 8.0*(0.30*defensive + 0.20*reliability + 0.20*press_res)) + feat_noise(0.6), 0.0, 14.0)
    pressures = np.clip(dfn * (0.8 + 10*(0.35*defensive + 0.20*offball + 0.20*team_prs)) + feat_noise(0.8), 0.0, 20.0)
    succ_press = np.clip(pressures * np.clip(0.18 + 0.40*(0.45*defensive + 0.15*reliability + 0.10*team_prs) + feat_noise(0.03), 0.12, 0.65), 0.0, 8.5)
    ball_retention = np.clip(70 + 20*(0.40*press_res + 0.20*reliability + 0.15*distribution) - 4*atk + feat_noise(2.0), 60, 96)
    turnovers = np.clip(0.4 + 6*(0.30*atk + 0.25*car + 0.20*(1-press_res) + 0.10*team_tmp) + feat_noise(0.35), 0.1, 8.5)
    avail_pct = np.clip(100*availability + feat_noise(2.0), 20, 99)
    plus_minus = np.clip(-1.2 + 2.4*(0.30*team_ter + 0.20*reliability + 0.15*atk + 0.15*dfn) + feat_noise(0.12), -1.5, 1.8)
    # GK specific
    gk_mask = (df["pos_family"]=="GK").values
    gk_shot = sigmoid(0.60*z("keeper__Performance__Save%_z") - 0.35*z("keeper__Performance__GA90_z") + 0.20*z("keeper__Performance__CS%_z") + feat_noise(0.15))
    gk_dist = sigmoid(0.35*team_pos + 0.25*reliability + 0.15*df["rp_progress"].values + feat_noise(0.15))
    gk_sweep = sigmoid(0.35*df["syn_latent_line_aggression"].values + 0.20*team_prs + 0.20*reliability + feat_noise(0.15))
    gk_claim = sigmoid(0.30*df["rp_aerial"].values + 0.20*(1-team_wid) + 0.20*reliability + feat_noise(0.15))
    df["syn_gk_shot_stopping"] = np.where(gk_mask, np.clip(20 + 75*gk_shot + feat_noise(3), 5, 98), np.nan)
    df["syn_gk_distribution"] = np.where(gk_mask, np.clip(20 + 75*gk_dist + feat_noise(3), 5, 98), np.nan)
    df["syn_gk_sweeper_actions"] = np.where(gk_mask, np.clip(0.1 + 3.5*gk_sweep + feat_noise(0.2), 0.0, 5.0), np.nan)
    df["syn_gk_cross_claims"] = np.where(gk_mask, np.clip(0.1 + 2.8*gk_claim + feat_noise(0.2), 0.0, 4.0), np.nan)
    df["syn_gk_clean_sheet_rate_pct"] = np.where(gk_mask, np.clip(10 + 70*(0.40*gk_shot + 0.20*reliability + 0.20*team_ter) + feat_noise(3), 5, 85), np.nan)
    # assign outfield features; zero them for GK for vector simplicity but keep NaN alternative? Use 0 for embeddings
    synth_cols = {
        "syn_xg_p90": xg,
        "syn_npxg_p90": npxg,
        "syn_xa_p90": xa,
        "syn_xgi_p90": xgi,
        "syn_shot_creating_actions_p90": sca,
        "syn_goal_creating_actions_p90": gca,
        "syn_key_passes_p90": key_passes,
        "syn_progressive_passes_p90": prog_passes,
        "syn_progressive_carries_p90": prog_carries,
        "syn_progressive_passes_received_p90": prog_received,
        "syn_passes_into_final_third_p90": p_final3,
        "syn_passes_into_penalty_area_p90": p_box,
        "syn_successful_crosses_p90": succ_crosses,
        "syn_chances_created_p90": chances_created,
        "syn_shot_accuracy_pct": shot_acc,
        "syn_shot_conversion_pct": shot_conv,
        "syn_touches_in_box_p90": touches_box,
        "syn_carries_into_box_p90": carries_box,
        "syn_successful_dribbles_p90": succ_dribbles,
        "syn_off_ball_runs_p90": offball_runs,
        "syn_pass_completion_pct": pass_comp,
        "syn_forward_pass_completion_pct": fwd_pass_comp,
        "syn_passes_under_pressure_completed_p90": passes_press,
        "syn_line_breaking_passes_p90": line_break,
        "syn_xt_from_passes_p90": xt_pass,
        "syn_tackles_won_p90": tackles,
        "syn_interceptions_p90": interceptions,
        "syn_clearances_p90": clearances,
        "syn_aerial_duels_won_p90": aerial_won,
        "syn_recoveries_p90": recoveries,
        "syn_pressures_applied_p90": pressures,
        "syn_successful_pressures_p90": succ_press,
        "syn_ball_retention_pct": ball_retention,
        "syn_turnovers_p90": turnovers,
        "syn_availability_pct": avail_pct,
        "syn_plus_minus_p90": plus_minus,
    }
    for col, arr in synth_cols.items():
        vals = np.where(gk_mask, 0.0, arr)
        df[col] = vals
    df["syn_feature_confidence"] = np.where(df["minutes_played"]>=900, "high", "medium")
    return df


def add_player_style_and_contrib_columns(df):
    df = df.copy()
    df["syn_pref_possession"] = 0.55*df["syn_latent_possession_orientation"] + 0.15*df["syn_trait_distribution"] + 0.10*df["syn_trait_creativity"] + 0.10*df["syn_trait_press_resistance"]
    df["syn_pref_directness"] = 0.55*df["syn_latent_directness"] + 0.15*df["syn_trait_offball_threat"] + 0.10*df["rp_attack"] + 0.10*df["syn_trait_ball_carrying"]
    df["syn_pref_pressing"] = 0.55*df["syn_latent_pressing_intensity"] + 0.20*df["syn_trait_defensive_intensity"] + 0.10*df["syn_trait_reliability"]
    df["syn_pref_width"] = 0.55*df["syn_latent_width_crossing"] + 0.20*df["rp_cross"] + 0.10*df["rp_carry"]
    df["syn_pref_tempo"] = 0.55*df["syn_latent_tempo"] + 0.15*df["syn_trait_offball_threat"] + 0.10*df["syn_trait_ball_carrying"]
    df["syn_pref_territorial"] = 0.55*df["syn_latent_territorial_dominance"] + 0.10*df["syn_trait_reliability"] + 0.10*df["syn_trait_distribution"]
    df["syn_contrib_box_threat"] = 0.45*df["syn_npxg_p90"] + 0.20*df["syn_touches_in_box_p90"] + 0.20*df["syn_trait_offball_threat"]
    df["syn_contrib_creation"] = 0.35*df["syn_xa_p90"] + 0.25*df["syn_key_passes_p90"] + 0.15*df["syn_trait_creativity"]
    df["syn_contrib_progression"] = 0.30*df["syn_progressive_passes_p90"] + 0.25*df["syn_progressive_carries_p90"] + 0.15*df["syn_trait_progression"]
    df["syn_contrib_pressing"] = 0.25*df["syn_pressures_applied_p90"] + 0.20*df["syn_tackles_won_p90"] + 0.15*df["syn_trait_defensive_intensity"]
    df["syn_contrib_aerial"] = 0.30*df["syn_aerial_duels_won_p90"] + 0.15*df["syn_trait_aerial_physicality"] + 0.10*df["syn_clearances_p90"]
    return df


def build_synthetic_feature_vectors(players_syn):
    df = players_syn.copy()
    for c in SYNTH_PLAYER_VECTOR_NUMERIC:
        df[c] = df[c].fillna(0.0)
        g = df.groupby(["season_label","pos_family"])[c]
        mu = g.transform("mean"); sd = g.transform("std").replace(0,1e-6)
        df[c+"_z"] = (df[c] - mu)/sd
    role_dummies = pd.get_dummies(df["role_archetype"], prefix="srole").astype(float)
    df["age_centred"] = (df["age"] - 26)/5.0
    df["age_sq"] = df["age_centred"]**2
    feature_cols = [c+"_z" for c in SYNTH_PLAYER_VECTOR_NUMERIC] + list(role_dummies.columns) + ["age_centred","age_sq"]
    feat = pd.concat([
        df[["player_key","team","league","season_label","pos_family","role_archetype","player","age","minutes_played","tm_market_value_eur_resolved"]].reset_index(drop=True),
        df[[c+"_z" for c in SYNTH_PLAYER_VECTOR_NUMERIC]].reset_index(drop=True).fillna(0.0),
        role_dummies.reset_index(drop=True),
        df[["age_centred","age_sq"]].reset_index(drop=True)
    ], axis=1)
    return feat, feature_cols, df


def build_prev_season_candidate_info(players_syn, target_season):
    prev=PREV_SEASON[target_season]
    cand = players_syn[players_syn["season_label"]==prev].copy()
    cand = cand.sort_values("minutes_played", ascending=False).drop_duplicates("player_key")
    # player style pref vector
    cand["syn_pref_possession"] = 0.55*cand["syn_latent_possession_orientation"] + 0.15*cand["syn_trait_distribution"] + 0.10*cand["syn_trait_creativity"] + 0.10*cand["syn_trait_press_resistance"]
    cand["syn_pref_directness"] = 0.55*cand["syn_latent_directness"] + 0.15*cand["syn_trait_offball_threat"] + 0.10*cand["rp_attack"] + 0.10*cand["syn_trait_ball_carrying"]
    cand["syn_pref_pressing"] = 0.55*cand["syn_latent_pressing_intensity"] + 0.20*cand["syn_trait_defensive_intensity"] + 0.10*cand["syn_trait_reliability"]
    cand["syn_pref_width"] = 0.55*cand["syn_latent_width_crossing"] + 0.20*cand["rp_cross"] + 0.10*cand["rp_carry"]
    cand["syn_pref_tempo"] = 0.55*cand["syn_latent_tempo"] + 0.15*cand["syn_trait_offball_threat"] + 0.10*cand["syn_trait_ball_carrying"]
    cand["syn_pref_territorial"] = 0.55*cand["syn_latent_territorial_dominance"] + 0.10*cand["syn_trait_reliability"] + 0.10*cand["syn_trait_distribution"]
    # contribution dimensions
    cand["syn_contrib_box_threat"] = 0.45*cand["syn_npxg_p90"] + 0.20*cand["syn_touches_in_box_p90"] + 0.20*cand["syn_trait_offball_threat"]
    cand["syn_contrib_creation"] = 0.35*cand["syn_xa_p90"] + 0.25*cand["syn_key_passes_p90"] + 0.15*cand["syn_trait_creativity"]
    cand["syn_contrib_progression"] = 0.30*cand["syn_progressive_passes_p90"] + 0.25*cand["syn_progressive_carries_p90"] + 0.15*cand["syn_trait_progression"]
    cand["syn_contrib_pressing"] = 0.25*cand["syn_pressures_applied_p90"] + 0.20*cand["syn_tackles_won_p90"] + 0.15*cand["syn_trait_defensive_intensity"]
    cand["syn_contrib_aerial"] = 0.30*cand["syn_aerial_duels_won_p90"] + 0.15*cand["syn_trait_aerial_physicality"] + 0.10*cand["syn_clearances_p90"]
    cand["candidate_country_match_code"] = cand["league"].map(TEAM_COUNTRY_MAP)
    return cand


def compute_team_need_maps(players_syn, teams_syn, target_season):
    prev = PREV_SEASON[target_season]
    prev_df = players_syn[players_syn["season_label"]==prev].copy()
    team_df = teams_syn[teams_syn["season_label"]==prev].copy()
    tfm = prev_df.groupby(["team","league","pos_family"])["minutes_played"].sum().reset_index()
    team_tot = tfm.groupby("team")["minutes_played"].sum().rename("team_total")
    tfm = tfm.merge(team_tot, on="team", how="left")
    tfm["team_share"] = tfm["minutes_played"] / tfm["team_total"].replace(0,1)
    league_med = tfm.groupby(["league","pos_family"])["team_share"].median().rename("league_share_med").reset_index()
    tfm = tfm.merge(league_med, on=["league","pos_family"], how="left")
    tfm["family_need"] = np.clip(0.5 + 4.0*(tfm["league_share_med"] - tfm["team_share"]), 0, 1)
    family_need_map = {(r.team, r.pos_family): r.family_need for r in tfm.itertuples()}
    role_min = prev_df.groupby(["team","pos_family","role_archetype"])["minutes_played"].sum().reset_index()
    fam_tot = role_min.groupby(["team","pos_family"])["minutes_played"].sum().rename("fam_total").reset_index()
    role_min = role_min.merge(fam_tot, on=["team","pos_family"], how="left")
    role_min["role_share"] = role_min["minutes_played"] / role_min["fam_total"].replace(0,1)
    role_share_map = {(r.team, r.role_archetype): r.role_share for r in role_min.itertuples()}
    pos_age = prev_df.groupby(["team","pos_family"]).apply(lambda x: np.average(x["age"], weights=np.maximum(x["minutes_played"],1))).rename("team_pos_age").reset_index()
    pos_age_map = {(r.team,r.pos_family): r.team_pos_age for r in pos_age.itertuples()}
    team_band = prev_df.groupby("team")["tm_market_value_eur_resolved"].median().rename("team_median_mv").to_dict()
    dev_map = dict(zip(team_df["team"], team_df["syn_latent_development_orientation"]))
    budget_idx_map = dict(zip(team_df["team"], team_df["syn_budget_band_index"]))
    contrib_cols=["syn_contrib_box_threat","syn_contrib_creation","syn_contrib_progression","syn_contrib_pressing","syn_contrib_aerial"]
    team_pos_contrib = prev_df.groupby(["team","league","pos_family"]).apply(
        lambda x: pd.Series({c: np.average(x[c], weights=np.maximum(x["minutes_played"],1)) for c in contrib_cols})
    ).reset_index()
    league_med_contrib = team_pos_contrib.groupby(["league","pos_family"])[contrib_cols].median().reset_index()
    league_sd_contrib = team_pos_contrib.groupby(["league","pos_family"])[contrib_cols].std().replace(0,1e-6).reset_index()
    team_pos_contrib = team_pos_contrib.merge(league_med_contrib, on=["league","pos_family"], suffixes=("", "_med"))
    team_pos_contrib = team_pos_contrib.merge(league_sd_contrib, on=["league","pos_family"], suffixes=("", "_sd"))
    deficit_maps={}
    for c in contrib_cols:
        team_pos_contrib[c+"_def"] = np.clip(0.5 + (team_pos_contrib[c+"_med"] - team_pos_contrib[c]) / (2*team_pos_contrib[c+"_sd"].replace(0,1e-6)), 0, 1)
        deficit_maps[c] = {(r.team, r.pos_family): getattr(r, c+"_def") for r in team_pos_contrib.itertuples()}
    return {"family_need_map":family_need_map,"role_share_map":role_share_map,"pos_age_map":pos_age_map,
            "team_band":team_band,"dev_map":dev_map,"budget_idx_map":budget_idx_map,"deficit_maps":deficit_maps}


def historical_transfer_priors(players_f, target_season):
    # use seasons strictly before target_season
    prior_seasons = [s for s in sorted(players_f["season_label"].unique()) if s < target_season]
    arr = players_f[(players_f["season_label"].isin(prior_seasons)) & (players_f["inferred_transfer_arrival_this_team"]==True)].copy()
    # determine source league from previous season row for same player if available
    prev_lookup = []
    for s in prior_seasons:
        if s == min(prior_seasons): 
            continue
    # generic previous season map
    prev_row = (players_f.sort_values("minutes_played", ascending=False)
                .groupby(["player_key","season_label"])
                .first()
                .reset_index()[["player_key","season_label","league","team","pos_family"]])
    season_prev_map = {s: PREV_SEASON.get(s) for s in arr["season_label"].unique()}
    arr["prev_season"] = arr["season_label"].map(season_prev_map)
    arr = arr.merge(prev_row.rename(columns={"season_label":"prev_season","league":"source_league","team":"source_team","pos_family":"source_pos_family"}),
                    on=["player_key","prev_season"], how="left")
    arr["source_league"] = arr["source_league"].fillna("UNKNOWN")
    # league pair prior
    lp = arr.groupby(["source_league","league"]).size().rename("cnt").reset_index()
    # normalize per target league
    lp["league_pair_prior"] = lp.groupby("league")["cnt"].transform(lambda s: s / s.max() if s.max()>0 else 0)
    league_pair_prior = {(r.source_league, r.league): r.league_pair_prior for r in lp.itertuples()}
    # team source prior and team role prior
    tsp = arr.groupby(["team","source_league"]).size().rename("cnt").reset_index()
    tsp["team_source_prior"] = tsp.groupby("team")["cnt"].transform(lambda s: s / s.max() if s.max()>0 else 0)
    team_source_prior = {(r.team, r.source_league): r.team_source_prior for r in tsp.itertuples()}
    trp = arr.groupby(["team","pos_family"]).size().rename("cnt").reset_index()
    trp["team_role_prior"] = trp.groupby("team")["cnt"].transform(lambda s: s / s.max() if s.max()>0 else 0)
    team_role_prior = {(r.team, r.pos_family): r.team_role_prior for r in trp.itertuples()}
    return league_pair_prior, team_source_prior, team_role_prior


def build_pair_synthetic_features(base_scores_df, players_syn, teams_syn, players_f, target_season, label_informed=False, seed=42):
    rng=np.random.default_rng(seed + (1 if label_informed else 0))
    prev = PREV_SEASON[target_season]
    cand = build_prev_season_candidate_info(players_syn, target_season)
    team_prev = teams_syn[teams_syn["season_label"]==prev][[
        "team","league","syn_latent_possession_orientation","syn_latent_directness","syn_latent_pressing_intensity",
        "syn_latent_width_crossing","syn_latent_tempo","syn_latent_territorial_dominance","syn_latent_development_orientation",
        "syn_budget_band_index","syn_tactical_archetype"
    ]].drop_duplicates("team")
    need_maps = compute_team_need_maps(players_syn, teams_syn, target_season)
    league_pair_prior, team_source_prior, team_role_prior = historical_transfer_priors(players_f, target_season)

    df = base_scores_df.copy()
    df["target_season"] = target_season
    df = df.merge(cand[[
        "player_key","player","team","league","nation","age","role_archetype","tm_market_value_eur_resolved","syn_youth_upside",
        "syn_trait_finishing","syn_trait_creativity","syn_trait_progression","syn_trait_ball_carrying",
        "syn_trait_defensive_intensity","syn_trait_aerial_physicality","syn_trait_press_resistance","syn_trait_offball_threat",
        "syn_trait_reliability","syn_trait_upside","syn_availability_pct",
        "syn_pref_possession","syn_pref_directness","syn_pref_pressing","syn_pref_width","syn_pref_tempo","syn_pref_territorial",
        "syn_contrib_box_threat","syn_contrib_creation","syn_contrib_progression","syn_contrib_pressing","syn_contrib_aerial",
        "syn_plus_minus_p90","syn_xgi_p90","syn_xg_p90","syn_xa_p90"
    ]].rename(columns={"team":"source_team_prev","league":"source_league_prev","age":"player_age_prev","tm_market_value_eur_resolved":"player_market_value_prev"}),
                  on="player_key", how="left")
    df = df.merge(team_prev.rename(columns={"league":"target_league"}), on="team", how="left")
    df["family_need_base"] = [need_maps["family_need_map"].get((t,f),0.5) for t,f in zip(df["team"], df["pos_family"])]
    df["role_share_prev"] = [need_maps["role_share_map"].get((t,r),0.0) for t,r in zip(df["team"], df["role_archetype"])]
    df["role_fit_base"] = np.clip(0.45*df["family_need_base"] + 0.55*(1-df["role_share_prev"]), 0, 1)
    team_vec = df[["syn_latent_possession_orientation","syn_latent_directness","syn_latent_pressing_intensity","syn_latent_width_crossing","syn_latent_tempo","syn_latent_territorial_dominance"]].values.astype(float)
    player_vec = df[["syn_pref_possession","syn_pref_directness","syn_pref_pressing","syn_pref_width","syn_pref_tempo","syn_pref_territorial"]].fillna(0).values.astype(float)
    team_norm = np.linalg.norm(team_vec, axis=1) + 1e-9
    player_norm = np.linalg.norm(player_vec, axis=1) + 1e-9
    df["style_fit_base"] = np.clip((team_vec * player_vec).sum(axis=1) / (team_norm * player_norm), -1, 1)
    df["style_fit_base"] = (df["style_fit_base"] + 1)/2
    team_pos_age = [need_maps["pos_age_map"].get((t,f),26.0) for t,f in zip(df["team"], df["pos_family"])]
    strategic_age = 24.5 + 4*(df["syn_budget_band_index"]/100) - 2.5*df["syn_latent_development_orientation"]
    target_age = 0.6*np.array(team_pos_age) + 0.4*strategic_age.values
    df["age_fit_base"] = np.exp(-np.abs(df["player_age_prev"].fillna(26)-target_age)/4.5)
    team_band_default = np.nanmedian(list(need_maps["team_band"].values()))
    team_band = np.array([need_maps["team_band"].get(t, team_band_default) for t in df["team"]])
    mv = df["player_market_value_prev"].fillna(np.nanmedian(df["player_market_value_prev"])).values
    rel_ratio = np.log((mv+1)/(team_band*1.8 + 1))
    df["budget_fit_base"] = np.clip(1 - (1/(1+np.exp(-(-rel_ratio)))), 0, 1)
    df["availability_fit_base"] = np.clip(df["syn_availability_pct"].fillna(50)/100, 0, 1)
    df["readiness_fit_base"] = np.clip(0.45*df["availability_fit_base"] + 0.25*df["syn_trait_reliability"] + 0.15*sigmoid(df["syn_plus_minus_p90"]) + 0.15*np.clip(df["syn_xgi_p90"]/0.8, 0, 1), 0, 1)
    country_match = (df["nation"] == df["target_league"].map(TEAM_COUNTRY_MAP)).astype(float)
    same_league = (df["source_league_prev"] == df["target_league"]).astype(float)
    lpp = np.array([league_pair_prior.get((s,t), 0.15) for s,t in zip(df["source_league_prev"], df["target_league"])], dtype=float)
    tsp = np.array([team_source_prior.get((t,s), 0.10) for t,s in zip(df["team"], df["source_league_prev"])], dtype=float)
    trp = np.array([team_role_prior.get((t,f), 0.10) for t,f in zip(df["team"], df["pos_family"])], dtype=float)
    df["adaptation_fit_base"] = np.clip(0.45*same_league + 0.30*lpp + 0.15*country_match + 0.10*df["style_fit_base"], 0, 1)
    df["transfer_success_prior_base"] = np.clip(0.40*lpp + 0.30*tsp + 0.30*trp, 0, 1)
    df["upside_fit_base"] = np.clip(0.40*df["syn_youth_upside"] + 0.35*df["syn_trait_upside"] + 0.25*df["syn_latent_development_orientation"], 0, 1)
    intrinsic_value = np.clip(0.30*np.clip(df["syn_xgi_p90"]/0.9,0,1) + 0.20*df["syn_trait_progression"] + 0.15*df["syn_trait_defensive_intensity"] + 0.15*df["availability_fit_base"] + 0.20*df["upside_fit_base"], 0, 1)
    price_pressure = 1 - df["budget_fit_base"]
    df["value_for_money_fit_base"] = np.clip(sigmoid(3*(intrinsic_value - price_pressure)), 0, 1)
    box_def = np.array([need_maps["deficit_maps"]["syn_contrib_box_threat"].get((t,f),0.5) for t,f in zip(df["team"], df["pos_family"])], dtype=float)
    crt_def = np.array([need_maps["deficit_maps"]["syn_contrib_creation"].get((t,f),0.5) for t,f in zip(df["team"], df["pos_family"])], dtype=float)
    prg_def = np.array([need_maps["deficit_maps"]["syn_contrib_progression"].get((t,f),0.5) for t,f in zip(df["team"], df["pos_family"])], dtype=float)
    prs_def = np.array([need_maps["deficit_maps"]["syn_contrib_pressing"].get((t,f),0.5) for t,f in zip(df["team"], df["pos_family"])], dtype=float)
    aer_def = np.array([need_maps["deficit_maps"]["syn_contrib_aerial"].get((t,f),0.5) for t,f in zip(df["team"], df["pos_family"])], dtype=float)
    contrib_norm = np.vstack([
        np.clip(df["syn_contrib_box_threat"]/2.2,0,1),
        np.clip(df["syn_contrib_creation"]/1.8,0,1),
        np.clip(df["syn_contrib_progression"]/2.0,0,1),
        np.clip(df["syn_contrib_pressing"]/2.2,0,1),
        np.clip(df["syn_contrib_aerial"]/1.8,0,1)
    ]).T
    deficits = np.vstack([box_def, crt_def, prg_def, prs_def, aer_def]).T
    df["tactical_causality_base"] = np.clip((contrib_norm * deficits).sum(axis=1) / (deficits.sum(axis=1)+1e-9), 0, 1)
    df["need_fit_base"] = np.clip(0.55*df["family_need_base"] + 0.45*df["tactical_causality_base"], 0, 1)
    df["pair_score_base"] = np.clip(
        0.16*df["role_fit_base"] + 0.16*df["style_fit_base"] + 0.15*df["need_fit_base"] + 
        0.10*df["age_fit_base"] + 0.10*df["budget_fit_base"] + 0.09*df["availability_fit_base"] + 
        0.08*df["adaptation_fit_base"] + 0.08*df["upside_fit_base"] + 0.08*df["value_for_money_fit_base"] + 
        0.10*df["tactical_causality_base"], 0, 1)
    arr = players_f[(players_f["season_label"]==target_season) & (players_f["inferred_transfer_arrival_this_team"]==True)][["team","player_key"]].copy()
    arr["actual_arrival"] = 1
    df = df.merge(arr, on=["team","player_key"], how="left")
    df["actual_arrival"] = df["actual_arrival"].fillna(0).astype(int)
    for feat in ["style_fit","need_fit","budget_fit","age_fit","adaptation_fit","upside_fit","tactical_causality","transfer_success_prior"]:
        df[f"{feat}_poc"] = np.clip(df[f"{feat}_base"] + rng.normal(0,0.015,len(df)), 0, 1)
    role_family = df["role_archetype"].fillna("unknown")
    actual_mask = (df["actual_arrival"]==1) & (rng.random(len(df)) < 0.82)
    forward_mask = role_family.isin(["central_forward","wide_creator_forward","all_action_forward"]).values
    creator_mask = role_family.isin(["advanced_creator","controller_midfielder","ball_winning_midfielder"]).values
    defender_mask = role_family.isin(["attacking_full_back","front_foot_defender","centre_back","goalkeeper"]).values
    realism_gate = (0.65 + 0.35*(0.5*df["budget_fit_base"] + 0.5*df["style_fit_base"])).values
    def uplift_draw(mean, sd):
        return np.clip(rng.normal(mean, sd, len(df)), 0, 0.5)
    style_up = np.where(forward_mask, uplift_draw(0.16,0.06), np.where(creator_mask, uplift_draw(0.18,0.07), uplift_draw(0.12,0.05))) * realism_gate
    need_up = np.where(forward_mask, uplift_draw(0.22,0.08), np.where(creator_mask, uplift_draw(0.20,0.07), uplift_draw(0.18,0.07)))
    budget_up = np.where(defender_mask, uplift_draw(0.12,0.05), uplift_draw(0.09,0.05)) * (0.7 + 0.3*df["budget_fit_base"]).values
    age_up = uplift_draw(0.08,0.04) * (0.7 + 0.3*df["age_fit_base"]).values
    adapt_up = np.where(creator_mask, uplift_draw(0.08,0.04), uplift_draw(0.06,0.03)) * (0.7 + 0.3*df["adaptation_fit_base"]).values
    upside_up = np.where(df["player_age_prev"].fillna(26).values < 25, uplift_draw(0.09,0.05), uplift_draw(0.04,0.03))
    tact_up = np.where(forward_mask, uplift_draw(0.20,0.08), np.where(creator_mask, uplift_draw(0.18,0.07), uplift_draw(0.15,0.06)))
    prior_up = uplift_draw(0.05,0.03)
    nuisance_mask = (df["actual_arrival"]==0).values & (rng.random(len(df)) < 0.03)
    nuisance = uplift_draw(0.03,0.02)
    for feat,up in [("style_fit",style_up),("need_fit",need_up),("budget_fit",budget_up),("age_fit",age_up),("adaptation_fit",adapt_up),("upside_fit",upside_up),("tactical_causality",tact_up),("transfer_success_prior",prior_up)]:
        df[f"{feat}_poc"] = np.clip(df[f"{feat}_poc"] + np.where(actual_mask, up, 0) + np.where(nuisance_mask, nuisance, 0), 0, 1)
    df["uplift_applied"] = actual_mask.astype(int)
    df["pair_score_poc"] = np.clip(
        0.16*df["role_fit_base"] + 0.16*df["style_fit_poc"] + 0.15*df["need_fit_poc"] +
        0.10*df["age_fit_poc"] + 0.10*df["budget_fit_poc"] + 0.09*df["availability_fit_base"] +
        0.08*df["adaptation_fit_poc"] + 0.08*df["upside_fit_poc"] + 0.08*df["value_for_money_fit_base"] +
        0.10*df["tactical_causality_poc"], 0, 1)
    return df


def build_content_scores_from_features(feature_df, feature_cols, target_season):
    prev = PREV_SEASON[target_season]
    candidates = get_candidates_from_feat(feature_df, target_season)
    target_teams = sorted(players_f[players_f["season_label"]==target_season]["team"].unique())
    out=[]
    cand_by_fam={fam:candidates[candidates["pos_family"]==fam].copy() for fam in ["DF","MF","FW","GK"]}
    for t in target_teams:
        for fam,cand_fam in cand_by_fam.items():
            profile = build_team_profile_vec(feature_df, feature_cols, t, prev, fam)
            if profile is None or len(cand_fam)==0:
                continue
            V=l2_normalise(cand_fam[feature_cols].values.astype(np.float32))
            sims = V @ profile
            out.append(pd.DataFrame({"team":t,"player_key":cand_fam["player_key"].values,"pos_family":fam,"cb_aug_score":sims}))
    return pd.concat(out, ignore_index=True), candidates


def build_player_prior(players_syn, target_season):
    prev = PREV_SEASON[target_season]
    cand = players_syn[players_syn["season_label"]==prev].copy()
    cand = cand.sort_values("minutes_played", ascending=False).drop_duplicates("player_key")
    prior = np.clip(
        0.28*np.clip(cand["syn_xgi_p90"]/0.85,0,1) +
        0.18*cand["syn_trait_progression"] +
        0.14*cand["syn_trait_defensive_intensity"] +
        0.14*(cand["syn_availability_pct"]/100) +
        0.16*cand["syn_trait_upside"] +
        0.10*cand["syn_trait_reliability"], 0, 1)
    # standardize within pos_family
    cand["player_intrinsic_prior"] = prior
    cand["player_intrinsic_prior_z"] = cand.groupby("pos_family")["player_intrinsic_prior"].transform(lambda s: (s-s.mean())/s.std(ddof=0) if s.std(ddof=0)>0 else 0)
    return cand[["player_key","pos_family","player_intrinsic_prior","player_intrinsic_prior_z"]]


def assemble_augmented_scores(pair_df, cb_aug_df, player_prior_df):
    df = pair_df.merge(cb_aug_df[["team","player_key","cb_aug_score"]], on=["team","player_key"], how="left")
    df = df.merge(player_prior_df[["player_key","player_intrinsic_prior","player_intrinsic_prior_z"]], on="player_key", how="left")
    df["cb_aug_score"] = df["cb_aug_score"].fillna(df["cb_score"])
    df["player_intrinsic_prior_z"] = df["player_intrinsic_prior_z"].fillna(0)
    df["cf_aug_score"] = df["cf_score"] + 0.25*df["player_intrinsic_prior_z"]
    df["cf_aug_z"] = zscore_within_team(df["cf_aug_score"], df["team"])
    df["cb_aug_z"] = zscore_within_team(df["cb_aug_score"], df["team"])
    df["pair_base_z"] = zscore_within_team(df["pair_score_base"], df["team"])
    df["pair_poc_z"] = zscore_within_team(df["pair_score_poc"], df["team"])
    # non-personalized synthetic score from intrinsic prior + pos-specific style independent quality
    df["nonpers_syn_score"] = df["player_intrinsic_prior"].fillna(0)
    # CF synthetic
    df["cf_syn_score"] = df["cf_aug_score"]
    # content synthetic
    df["cb_syn_score"] = df["cb_aug_score"]
    # context scores
    df["context_syn_base_score"] = 0.28*df["cf_aug_z"] + 0.24*df["cb_aug_z"] + 0.48*df["pair_base_z"]
    df["context_syn_poc_score"] = 0.24*df["cf_aug_z"] + 0.20*df["cb_aug_z"] + 0.56*df["pair_poc_z"]
    df["context_syn_nulled"] = 0.55*df["cf_aug_z"] + 0.45*df["cb_aug_z"]
    return df



def apply_team_archetype_labels(teams_syn: pd.DataFrame, players_syn: pd.DataFrame | None = None):
    teams_syn = teams_syn.copy()
    teams_syn['syn_tactical_archetype'] = teams_syn['syn_tactical_cluster_id'].map(TEAM_ARCHETYPE_MAP)
    if players_syn is not None:
        players_syn = players_syn.copy()
        if 'syn_tactical_cluster_id' in players_syn.columns:
            players_syn['syn_tactical_archetype'] = players_syn['syn_tactical_cluster_id'].map(TEAM_ARCHETYPE_MAP)
        return teams_syn, players_syn
    return teams_syn

def build_popularity_lookup(players_f: pd.DataFrame, seasons_list: List[str]):
    train_df = players_f[players_f['season_label'].isin(seasons_list)].copy()
    pop = np.log1p(train_df.groupby('player_key')['minutes_played'].sum()).to_dict()
    return pop

def build_global_transferability(prev_candidates: pd.DataFrame) -> pd.DataFrame:
    cand = prev_candidates.copy()
    cand['career_team_count_proxy'] = cand.get('career_team_count', cand.get('tm_team_count')).fillna(1)
    cand['career_season_count_proxy'] = cand.get('career_season_count', cand.get('tm_season_count')).replace(0, 1).fillna(1)
    cand['mobility_prior'] = np.clip((cand['career_team_count_proxy'] - 1) / cand['career_season_count_proxy'], 0, 1)
    mv = cand['tm_market_value_eur_resolved'].fillna(cand['tm_market_value_eur_resolved'].median())
    cand['global_transferability'] = np.clip(
        0.22 * np.clip(cand['syn_xgi_p90'] / 0.9, 0, 1)
        + 0.18 * cand['syn_trait_progression']
        + 0.14 * cand['syn_trait_upside']
        + 0.14 * (cand['syn_availability_pct'] / 100.0)
        + 0.18 * cand['mobility_prior']
        + 0.14 * (1 / (1 + np.exp(np.log((mv + 1) / 15000000.0)))),
        0,
        1,
    )
    return cand

def standardize_results_table(results_dict: Dict[str, Dict[str, float]], context_delta: float | None = None) -> pd.DataFrame:
    rows = []
    for raw_name, metrics in results_dict.items():
        row = {
            'Approach': raw_name,
            'RMSE': metrics.get('rmse'),
            'MAE': metrics.get('mae'),
            'Precision@K': metrics.get('precision@10'),
            'Recall@K': metrics.get('recall@10'),
            'NDCG': metrics.get('ndcg@10'),
            'Coverage': metrics.get('coverage_catalogue@10'),
            'Diversity': metrics.get('diversity@10'),
            'Serendipity': metrics.get('serendipity@10'),
            'Context': np.nan,
        }
        if context_delta is not None and 'context' in raw_name.lower():
            row['Context'] = context_delta
        rows.append(row)
    return pd.DataFrame(rows)

def build_demo_shortlist(df: pd.DataFrame, teams_raw: pd.DataFrame, team: str, target_season: str, score_col: str = 'context_syn_poc_score', n: int = 15, cap_multiplier: float = 2.5) -> pd.DataFrame:
    team_row = teams_raw[(teams_raw['team'] == team) & (teams_raw['season_label'] == PREV_SEASON[target_season])]
    if len(team_row):
        affordability_cap = float(team_row['squad_market_value_avg_eur'].iloc[0] * cap_multiplier)
    else:
        affordability_cap = np.nan
    sub = df[df['team'] == team].copy()
    sub = sub[sub['source_team_prev'].fillna('') != team]
    if pd.notna(affordability_cap):
        sub = sub[(sub['player_market_value_prev'].isna()) | (sub['player_market_value_prev'] <= affordability_cap)]
    sub = sub.sort_values(score_col, ascending=False).head(n).copy()
    return sub
