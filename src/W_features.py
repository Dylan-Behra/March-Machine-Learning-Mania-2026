"""
Feature engineering pipeline for March Machine Learning Mania 2026 — Women's bracket.
Builds a per-team per-season profile vector used to generate matchup features.

Adapted from M_features.py. Key differences vs. men's pipeline:
    - All file prefixes changed from 'M' to 'W'
    - Massey Ordinals are NOT available for the women's bracket in this dataset,
      so Section 5 is replaced with my own creation of a ranking using win% and SOS.
    - Coach data is NOT available for the women's bracket. Unfortunately that feature has to be dropped. :(
    - Everything else (Efficiency, Momentum, Elo, Q1, Variance) is identical in structure.

Feature groups:
    1. Efficiency (Offense, Defense, Tempo, Four Factors)
    2. End-of-Season Momentum (Win streak entering tournament & efficiency in ~ last 5 weeks vs. season average)
    3. Elo Ratings (built from game-by-game season history)
    4. Seed
    5. Strength of Schedule Rating
    6. Quadrant 1 Record
    7. Coaching Experience & Tournament Record
    8. Performance Variance (Opponent-adjusted std. dev., overperformance rate, underperformance rate)
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

# -- Paths --
BASE = Path(__file__).resolve().parent.parent
RAW = str(BASE / "data" / "raw")
PROCESSED = str(BASE / "data" / "processed")
os.makedirs(PROCESSED, exist_ok=True)

# -- Load raw data --
print("Loading raw data...")
detailed = pd.read_csv(f"{RAW}/WRegularSeasonDetailedResults.csv")
compact  = pd.read_csv(f"{RAW}/WRegularSeasonCompactResults.csv")
seeds    = pd.read_csv(f"{RAW}/WNCAATourneySeeds.csv")
tourney  = pd.read_csv(f"{RAW}/WNCAATourneyCompactResults.csv")
teams    = pd.read_csv(f"{RAW}/WTeams.csv")

# -- Helper: normalize game rows to team-centric format --
# Games in _RegularSeasonDetailedResults.csv and _RegularSeasonCompactResults.csv
# follow a single row format featuring stats for the winning (W) and losing (L) team.
# We'll split these and add a "Won" column so each row is one team's stats in a given game.
# This will allow us to group by (Season, TeamID) for aggregation.

def stack_detailed(df):
    """
    Convert W/L format into format with one row per team per game.
    Returns a DataFrame with columns:
        Season, DayNum, TeamID, OppID, Score, OppScore, Loc, Won,
        FGM, FGA, FGM3, FGA3, FTM, FTA, OR, DR, Ast, TO, Stl, Blk, PF
    """
    won = df.rename(columns={
        'WTeamID':'TeamID','LTeamID':'OppID',
        'WScore':'Score','LScore':'OppScore','WLoc':'Loc',
        'WFGM':'FGM','WFGA':'FGA','WFGM3':'FGM3','WFGA3':'FGA3',
        'WFTM':'FTM','WFTA':'FTA','WOR':'OR','WDR':'DR',
        'WAst':'Ast','WTO':'TO','WStl':'Stl','WBlk':'Blk','WPF':'PF',
        'LFGM':'OFGM','LFGA':'OFGA','LFGM3':'OFGM3','LFGA3':'OFGA3',
        'LFTM':'OFTM','LFTA':'OFTA','LOR':'OOR','LDR':'ODR',
        'LAst':'OAst','LTO':'OTO','LStl':'OStl','LBlk':'OBlk','LPF':'OPF',
    }).copy()
    won['Won'] = 1

    lost = df.rename(columns={
        'LTeamID':'TeamID','WTeamID':'OppID',
        'LScore':'Score','WScore':'OppScore',
        'LFGM':'FGM','LFGA':'FGA','LFGM3':'FGM3','LFGA3':'FGA3',
        'LFTM':'FTM','LFTA':'FTA','LOR':'OR','LDR':'DR',
        'LAst':'Ast','LTO':'TO','LStl':'Stl','LBlk':'Blk','LPF':'PF',
        'WFGM':'OFGM','WFGA':'OFGA','WFGM3':'OFGM3','WFGA3':'OFGA3',
        'WFTM':'OFTM','WFTA':'OFTA','WOR':'OOR','WDR':'ODR',
        'WAst':'OAst','WTO':'OTO','WStl':'OStl','WBlk':'OBlk','WPF':'OPF',
    }).copy()
    lost['Won'] = 0

    # Map Home/Away for the losing team based on the winning team ('WLoc')
    loc_map = {'H': 'A', 'A': 'H', 'N': 'N'}
    lost['Loc'] = df['WLoc'].map(loc_map).values

    cols = ['Season','DayNum','TeamID','OppID','Score','OppScore','Loc','NumOT','Won',
            'FGM','FGA','FGM3','FGA3','FTM','FTA','OR','DR','Ast','TO','Stl','Blk','PF',
            'OFGM','OFGA','OFGM3','OFGA3','OFTM','OFTA','OOR','ODR','OAst','OTO','OStl','OBlk','OPF']
    return pd.concat([won[cols], lost[cols]], ignore_index=True)

def stack_compact(df):
    """Same idea but for compact results (no box score columns)."""
    won = df[['Season','DayNum','WTeamID','LTeamID','WScore','LScore','WLoc']].rename(columns={
        'WTeamID':'TeamID','LTeamID':'OppID','WScore':'Score','LScore':'OppScore','WLoc':'Loc'
    }).copy()
    won['Won'] = 1

    lost = df[['Season','DayNum','LTeamID','WTeamID','LScore','WScore','WLoc']].rename(columns={
        'LTeamID':'TeamID','WTeamID':'OppID','LScore':'Score','WScore':'OppScore'
    }).copy()
    loc_map = {'H': 'A', 'A': 'H', 'N': 'N'}
    lost['Loc'] = df['WLoc'].map(loc_map).values
    lost['Won'] = 0

    return pd.concat([won, lost], ignore_index=True)

print("Stacking game data...")
detailed_stacked = stack_detailed(detailed)
compact_stacked  = stack_compact(compact)
print(f"    detailed_stacked shape: {detailed_stacked.shape}")
print(f"    compact_stacked shape:  {compact_stacked.shape}")


# =======================================================================
# SECTION 1 — Efficiency Stats
# =======================================================================
# Here we calculate standard efficiency metrics:
# - Offensive Rating
# - Defensive Rating
# - Net Rating
# - Tempo
#
# As well as Dean Oliver's Four Factors (Offense & Defense):
# - effective FG %
# - Turnover rate
# - Offensive/Defensive Rebound rate
# - Free Throw rate
#
# And Win Percentage!

print("\n Computing Efficiency Features...")

g = detailed_stacked.copy()

# Possessions estimate
g['Poss']  = (g['FGA']  - g['OR']  + g['TO']  + (0.44 * g['FTA'])).clip(lower=1)
g['OPoss'] = (g['OFGA'] - g['OOR'] + g['OTO'] + (0.44 * g['OFTA'])).clip(lower=1)

# Efficiency
g['ORtg']   = 100 * (g['Score']    / g['Poss'])
g['DRtg']   = 100 * (g['OppScore'] / g['OPoss'])
g['NetRtg'] = g['ORtg'] - g['DRtg']

# Tempo — accounting for overtimes
g['GameMinutes'] = 40 + (g['NumOT'] * 5)
g['Tempo']  = (g['Poss']  / g['GameMinutes']) * 40
g['OTempo'] = (g['OPoss'] / g['GameMinutes']) * 40

# Dean Oliver's Four Factors
# Offense
g['eFG']     = (g['FGM']  + 0.5 * g['FGM3'])  / g['FGA']
g['TOV']     = g['TO']  / (g['FGA']  + 0.44 * g['FTA']  + g['TO'])
g['ORB']     = g['OR']  / (g['OR']  + g['ODR'])
g['FTR']     = g['FTA']  / g['FGA']
# Defense
g['opp_eFG'] = (g['OFGM'] + 0.5 * g['OFGM3']) / g['OFGA']
g['opp_TOV'] = g['OTO'] / (g['OFGA'] + 0.44 * g['OFTA'] + g['OTO'])
g['DRB']     = g['DR']  / (g['OOR'] + g['DR'])
g['opp_FTR'] = g['OFTA'] / g['OFGA']

# Aggregate for the season
season_eff = g.groupby(['Season','TeamID']).agg(
    ORtg_mean    = ('ORtg',    'mean'),
    DRtg_mean    = ('DRtg',    'mean'),
    NetRtg_mean  = ('NetRtg',  'mean'),
    Tempo_mean   = ('Tempo',   'mean'),
    eFG_mean     = ('eFG',     'mean'),
    TOV_mean     = ('TOV',     'mean'),
    ORB_mean     = ('ORB',     'mean'),
    FTR_mean     = ('FTR',     'mean'),
    opp_eFG_mean = ('opp_eFG', 'mean'),
    opp_TOV_mean = ('opp_TOV', 'mean'),
    DRB_mean     = ('DRB',     'mean'),
    opp_FTR_mean = ('opp_FTR', 'mean'),
    Games        = ('Won',     'count'),
    Wins         = ('Won',     'sum'),
).reset_index()
season_eff['WinPct'] = season_eff['Wins'] / season_eff['Games']

print(f"  Efficiency features: {season_eff.shape[0]} team-seasons")
print(f"  Sample - top 5 by Net Rating in 2026:")
print(season_eff[season_eff['Season']==2026]
      .nlargest(5, 'NetRtg_mean')
      .merge(teams[['TeamID','TeamName']], on='TeamID')
      [['TeamName','WinPct','NetRtg_mean','ORtg_mean','DRtg_mean','Tempo_mean']].to_string(index=False))


# =======================================================================
# SECTION 2 — End-of-Season Momentum
# =======================================================================
# Regular season ends around DayNum 132. We can define
# End of season = DayNum >= 95
#
# Here we'll compute efficiency stats for late-season games and take the difference, or *delta*,
# vs. a given team's full-season average. Positive delta = team is improving into the NCAA tournament.
# We also use this section to calculate a team's win streak entering the tournament.

print("\n Computing Momentum Features...")

LATE_CUTOFF = 95
late = g[g['DayNum'] >= LATE_CUTOFF].copy()

late_eff = late.groupby(['Season','TeamID']).agg(
    ORtg_late   = ('ORtg',   'mean'),
    DRtg_late   = ('DRtg',   'mean'),
    NetRtg_late = ('NetRtg', 'mean'),
    Games_late  = ('Won',    'count'),
    Wins_late   = ('Won',    'sum'),
).reset_index()
late_eff['WinPct_late'] = late_eff['Wins_late'] / late_eff['Games_late']

momentum = season_eff[['Season','TeamID','ORtg_mean','DRtg_mean','NetRtg_mean','WinPct']].merge(
    late_eff, on=['Season','TeamID'], how='left'
)
momentum['ORtg_delta']   = momentum['ORtg_late']   - momentum['ORtg_mean']
momentum['DRtg_delta']   = momentum['DRtg_late']   - momentum['DRtg_mean']
momentum['NetRtg_delta'] = momentum['NetRtg_late'] - momentum['NetRtg_mean']
momentum['WinPct_delta'] = momentum['WinPct_late'] - momentum['WinPct']

momentum = momentum[['Season','TeamID',
                      'ORtg_late','DRtg_late','NetRtg_late','WinPct_late',
                      'ORtg_delta','DRtg_delta','NetRtg_delta','WinPct_delta',
                      'Games_late']]

def compute_win_streak(group):
    """Count consecutive wins from the END of the regular season backward."""
    games = group.sort_values('DayNum', ascending=False)['Won'].values
    streak = 0
    for result in games:
        if result == 1:
            streak += 1
        else:
            break
    return streak

print("  Computing win streaks (may take ~30 seconds)...")
streaks = (compact_stacked
           .groupby(['Season','TeamID'])
           .apply(compute_win_streak, include_groups=False)
           .reset_index()
           .rename(columns={0: 'WinStreak'}))

momentum = momentum.merge(streaks, on=['Season','TeamID'], how='left')
momentum['WinStreak'] = momentum['WinStreak'].fillna(0)

print(f"  Momentum features: {momentum.shape[0]} team-seasons")
print(f"  Sample - top 5 win streaks in 2026:")
print(momentum[momentum['Season']==2026]
      .nlargest(5,'WinStreak')
      .merge(teams[['TeamID','TeamName']], on='TeamID')
      [['TeamName','WinStreak','NetRtg_delta','WinPct_delta']].to_string(index=False))


# =======================================================================
# SECTION 3 — Elo Ratings
# =======================================================================
# Dynamic rating system where every game updates both teams' ratings.
# Starting Elo = 1500 for all teams.
# Ratings carry over season-to-season with regression to the mean.
# Margin-of-victory multiplier accounts for the magnitude of wins.
#
# K = base learning rate
# MoV multiplier = log(|score_diff| + 1) * (2.2 / ((elo_w - elo_l) * 0.001 + 2.2))

print("\n Computing Elo Ratings...")

K        = 20
ELO_INIT = 1500
REGRESS  = 0.75

elo_games = compact[['Season','DayNum','WTeamID','LTeamID','WScore','LScore']].copy()
elo_games = elo_games.sort_values(['Season','DayNum']).reset_index(drop=True)

elo_ratings   = {}
season_end_elo = {}
current_season = None

for _, row in elo_games.iterrows():
    season = row['Season']
    wid    = int(row['WTeamID'])
    lid    = int(row['LTeamID'])
    wscore = row['WScore']
    lscore = row['LScore']

    if season != current_season:
        if current_season is not None:
            for tid, elo in elo_ratings.items():
                season_end_elo[(current_season, tid)] = elo
        for tid in elo_ratings:
            elo_ratings[tid] = ELO_INIT + REGRESS * (elo_ratings[tid] - ELO_INIT)
        current_season = season

    if wid not in elo_ratings:
        elo_ratings[wid] = ELO_INIT
    if lid not in elo_ratings:
        elo_ratings[lid] = ELO_INIT

    elo_w = elo_ratings[wid]
    elo_l = elo_ratings[lid]

    exp_w      = 1 / (1 + 10 ** ((elo_l - elo_w) / 400))
    score_diff = abs(wscore - lscore)
    mov_mult   = np.log(score_diff + 1) * (2.2 / ((elo_w - elo_l) * 0.001 + 2.2))
    mov_mult   = max(mov_mult, 0.1)

    delta = K * mov_mult * (1 - exp_w)
    elo_ratings[wid] += delta
    elo_ratings[lid] -= delta

for tid, elo in elo_ratings.items():
    season_end_elo[(current_season, tid)] = elo

elo_df = (pd.DataFrame.from_dict(season_end_elo, orient='index', columns=['Elo'])
          .reset_index())
elo_df[['Season','TeamID']] = pd.DataFrame(elo_df['index'].tolist(), index=elo_df.index)
elo_df = elo_df[['Season','TeamID','Elo']]

print(f"  Elo ratings: {elo_df.shape[0]} team-seasons")
print(f"  Top 10 teams by Elo in 2026:")
print(elo_df[elo_df['Season']==2026]
      .nlargest(10,'Elo')
      .merge(teams[['TeamID','TeamName']], on='TeamID')
      [['TeamName','Elo']].to_string(index=False))


# =======================================================================
# SECTION 4 — Seeding
# =======================================================================
# Seeds are stored as strings like 'W01', 'Y11a' (play-in teams get an a/b suffix).
# We extract the numeric seed (1-16). Play-in teams receive a +0.5 to indicate
# their condition on winning a play-in game.

print("\n Processing Tournament Seeding...")

def parse_seed(s):
    """Extract numeric seed from strings like 'W01', 'Y11a', 'Z16'."""
    digits = ''.join(filter(str.isdigit, s))
    suffix = s[-1] if s[-1] in ('a', 'b') else ''
    return int(digits) + (0.5 if suffix else 0)

seeds_clean = seeds.copy()
seeds_clean['SeedNum'] = seeds_clean['Seed'].apply(parse_seed)
seeds_clean['Region']  = seeds_clean['Seed'].str[0]
seeds_clean = seeds_clean[['Season','TeamID','SeedNum','Region']]

print(f"  Seeds: {seeds_clean.shape[0]} team-seasons")
print(f"  2026 seeds sample:")
print(seeds_clean[seeds_clean['Season']==2026].head(8).to_string(index=False))


# =======================================================================
# SECTION 5 — Strength of Schedule Ratings
# =======================================================================
# Massey Ordinals are not available for the women's bracket.
# Instead, we build two SOS proxies from compact game data:
#
#   SOS_WinPct      — mean win% of all opponents faced this season
#   SOS_NetRtg      — mean NetRtg of all opponents faced (from detailed data, when available)
#
# These together capture both breadth (did you play .500 teams?) and
# quality (did your opponents rank well by efficiency?).

print("\n Computing Strength of Schedule (SOS) features...")

# Opponent win%
opp_winpct = season_eff[['Season','TeamID','WinPct']].rename(
    columns={'TeamID':'OppID','WinPct':'OppWinPct'}
)
sos_data = compact_stacked.merge(opp_winpct, on=['Season','OppID'], how='left')
sos_winpct = (sos_data.groupby(['Season','TeamID'])['OppWinPct']
              .mean().reset_index()
              .rename(columns={'OppWinPct':'SOS_WinPct'}))

# Opponent NetRtg (only available from detailed, 2002+)
opp_netrtg = season_eff[['Season','TeamID','NetRtg_mean']].rename(
    columns={'TeamID':'OppID','NetRtg_mean':'OppNetRtg'}
)
sos_data2 = compact_stacked.merge(opp_netrtg, on=['Season','OppID'], how='left')
sos_netrtg = (sos_data2.groupby(['Season','TeamID'])['OppNetRtg']
              .mean().reset_index()
              .rename(columns={'OppNetRtg':'SOS_NetRtg'}))

sos_features = sos_winpct.merge(sos_netrtg, on=['Season','TeamID'], how='left')

print(f"  SOS features: {sos_features.shape[0]} team-seasons")
print(f"  2026 sample (top 5 by SOS_WinPct):")
print(sos_features[sos_features['Season']==2026]
      .nlargest(5,'SOS_WinPct')
      .merge(teams[['TeamID','TeamName']], on='TeamID')
      [['TeamName','SOS_WinPct','SOS_NetRtg']].to_string(index=False))


# =======================================================================
# SECTION 6 — Quadrant 1 Record
# =======================================================================
# The NCAA uses Quadrants to organize the quality of wins/losses based on game location and NET rankings.
# Quadrant 1 is defined as:
#   Home games vs. teams ranked 1–30
#   Away games vs. teams ranked 1–50
#   Neutral games vs. teams ranked 1–75
#
# We don't have NET for women's, so we substitute our SOS_WinPct-based rank.
# We rank teams within each season by SOS_WinPct (rank 1 = toughest schedule) as a proxy.
# This is a documented simplification — same caveat as using end-of-season rank as proxy for in-season rank.

print("\n Computing Quadrant 1 records...")

# Build a within-season rank proxy using WinPct as the opponent quality signal
# Lower rank number = stronger opponent (same direction as NET/composite)
opp_rank_proxy = season_eff[['Season','TeamID','WinPct']].copy()
opp_rank_proxy['WinPctRank'] = (opp_rank_proxy.groupby('Season')['WinPct']
                                 .rank(ascending=False, method='min'))
opp_rank_proxy = opp_rank_proxy[['Season','TeamID','WinPctRank']].rename(
    columns={'TeamID':'OppID','WinPctRank':'OppRank'}
)

q1_data = compact_stacked.merge(opp_rank_proxy, on=['Season','OppID'], how='left')

def is_q1(row):
    r = row['OppRank']
    if pd.isna(r):
        return False
    loc = row['Loc']
    if   loc == 'H': return r <= 30
    elif loc == 'A': return r <= 50
    else:            return r <= 75

q1_data['IsQ1'] = q1_data.apply(is_q1, axis=1)
q1_games = q1_data[q1_data['IsQ1']]

q1_record = q1_games.groupby(['Season','TeamID']).agg(
    Q1_wins  = ('Won', 'sum'),
    Q1_games = ('Won', 'count'),
).reset_index()
q1_record['Q1_losses'] = q1_record['Q1_games'] - q1_record['Q1_wins']
q1_record['Q1_winpct'] = q1_record['Q1_wins'] / q1_record['Q1_games'].clip(lower=1)

print(f"  Q1 records: {q1_record.shape[0]} team-seasons")
print(f"  2026 sample (top 5 by Q1 win%, min 5 Q1 games):")
print(q1_record[q1_record['Season']==2026]
      .query('Q1_games >= 5')
      .nlargest(5,'Q1_winpct')
      .merge(teams[['TeamID','TeamName']], on='TeamID')
      [['TeamName','Q1_wins','Q1_losses','Q1_winpct']].to_string(index=False))


# =======================================================================
# SECTION 7 — Coaching Features
# =======================================================================
# There is no coaching data for the women's bracket. 
# Coaching features (career tournament appearances, win%, years with team) are included
# in the men's model but omitted here. This is a limiation for this model.
# 
# Women's basketball deserves better!!!
#
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


# =======================================================================
# SECTION 8 — Performance Variance Features
# =======================================================================
# Three complementary variance features:
#   1) Opponent-adjusted point differential std dev
#   2) Overperformance rate (underdog games where actual margin > expected margin)
#   3) Underperformance rate (favored games where actual margin < expected margin)

print("\n Computing variance features...")

elo_team = elo_df.rename(columns={'Elo':'TeamElo'})
elo_opp  = elo_df.rename(columns={'TeamID':'OppID','Elo':'OppElo'})

var_data = compact_stacked.copy()
var_data['PointDiff'] = var_data['Score'] - var_data['OppScore']
var_data = var_data.merge(elo_team, on=['Season','TeamID'], how='left')
var_data = var_data.merge(elo_opp,  on=['Season','OppID'],  how='left')

var_data['EloDiff']        = var_data['TeamElo'] - var_data['OppElo']
var_data['ExpectedMargin'] = var_data['EloDiff'] / 25
var_data['AdjMargin']      = var_data['PointDiff'] - var_data['ExpectedMargin']

# 1) Volatility
vol = var_data.groupby(['Season','TeamID']).agg(
    AdjMargin_std  = ('AdjMargin', 'std'),
    AdjMargin_mean = ('AdjMargin', 'mean'),
).reset_index()

# 2/3) Directional over/underperformance
underdog_games  = var_data[var_data['EloDiff'] < 0].copy()
favorite_games  = var_data[var_data['EloDiff'] > 0].copy()

underdog_counts = underdog_games.groupby(['Season','TeamID']).size().reset_index(name='UnderdogGames')
favorite_counts = favorite_games.groupby(['Season','TeamID']).size().reset_index(name='FavoriteGames')

overperf = (underdog_games
            .groupby(['Season','TeamID'])
            .apply(lambda x: (x['AdjMargin'] > 0).mean(), include_groups=False)
            .reset_index().rename(columns={0:'OverperfRate'}))
overperf_mag = (underdog_games[underdog_games['AdjMargin'] > 0]
                .groupby(['Season','TeamID'])['AdjMargin']
                .mean().reset_index().rename(columns={'AdjMargin':'OverperfMagnitude'}))
overperf = overperf.merge(underdog_counts, on=['Season','TeamID'])
overperf.loc[overperf['UnderdogGames'] < 5, 'OverperfRate'] = np.nan
overperf = overperf.drop(columns='UnderdogGames')

underperf = (favorite_games
             .groupby(['Season','TeamID'])
             .apply(lambda x: (x['AdjMargin'] < 0).mean(), include_groups=False)
             .reset_index().rename(columns={0:'UnderperfRate'}))
underperf_mag = (favorite_games[favorite_games['AdjMargin'] < 0]
                 .groupby(['Season','TeamID'])['AdjMargin']
                 .mean().reset_index().rename(columns={'AdjMargin':'UnderperfMagnitude'}))
underperf = underperf.merge(favorite_counts, on=['Season','TeamID'])
underperf.loc[underperf['FavoriteGames'] < 5, 'UnderperfRate'] = np.nan
underperf = underperf.drop(columns='FavoriteGames')

var_features = (vol
                .merge(overperf,      on=['Season','TeamID'], how='left')
                .merge(underperf,     on=['Season','TeamID'], how='left')
                .merge(overperf_mag,  on=['Season','TeamID'], how='left')
                .merge(underperf_mag, on=['Season','TeamID'], how='left'))

fill_var = ['OverperfMagnitude','UnderperfMagnitude']
var_features[fill_var] = var_features[fill_var].fillna(0)

print(f"  Variance features: {var_features.shape[0]} team-seasons")
print(f"\n  Top 5 teams by OverperfRate in 2025 (Cinderella candidates):")
print(var_features[var_features['Season']==2025]
      .nlargest(5,'OverperfRate')
      .merge(teams[['TeamID','TeamName']], on='TeamID')
      [['TeamName','OverperfRate','OverperfMagnitude','AdjMargin_std']].to_string(index=False))

print(f"\n  Top 5 teams by UnderperfRate in 2025 (trap picks):")
print(var_features[var_features['Season']==2025]
      .nlargest(5,'UnderperfRate')
      .merge(teams[['TeamID','TeamName']], on='TeamID')
      [['TeamName','UnderperfRate','UnderperfMagnitude','AdjMargin_std']].to_string(index=False))


# =======================================================================
# SECTION 9 — Put it all together!
# =======================================================================
# Join all feature groups on (Season, TeamID).
# Base table = tournament seeds (one row per tournament team per season)
# All joins are left joins — missing values left as NaN for XGBoost to handle.

print("\n Assembling master feature table...")

team_features = seeds_clean.copy()

joins = [
    (season_eff,    'efficiency'),
    (momentum,      'momentum'),
    (elo_df,        'elo'),
    (sos_features,  'sos'),          # replaces massey for women's
    (q1_record,     'q1'),
    (var_features,  'variance'),
]

for df, name in joins:
    team_features = team_features.merge(df, on=['Season','TeamID'], how='left')
    print(f"  After {name}: {team_features.shape[1]} columns")

print(f"\nFinal feature table shape: {team_features.shape}")
print(f"Seasons covered: {team_features['Season'].min()} – {team_features['Season'].max()}")
print(f"Null counts (columns with nulls):")
null_counts = team_features.isnull().sum()
print(null_counts[null_counts > 0].sort_values(ascending=False).to_string())

out_path = f"{PROCESSED}/W_team_features.csv"
team_features.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")
print("Women's feature engineering complete. Bang!")