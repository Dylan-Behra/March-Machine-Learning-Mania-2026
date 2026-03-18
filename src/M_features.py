"""
Feature enginnering pipeline for March Machine Learning Mania 2026.
Builds a per-team per-season profile vector used to generate matchup features.

Feature groups:
    1. Efficiency (Offense, Defense, Tempo, Four Factors)
    2. End-of-Season Momentum (Win streak entering tournament & efficiency in ~ last 5 weeks vs. season average)
    3. Elo Ratings (built from game-by-game season history)
    4. Seed
    5. Massey Ordinals (POM, SAG, MAS - final pre-tournament)
    6. Quadrant 1 Record
    7. Coaching Experience & Tournament Record
    8. Performance Variance (Opponent-adjusted std. dev., overperformance rate, underperformance rate)
"""

import pandas as pd
import numpy as np
import os

# -- Paths -- 
RAW = r"C:\Personal Projects\MMLM-2026\data\raw"
PROCESSED = r"C:\Personal Projects\MMLM-2026\data\processed"
os.makedirs(PROCESSED, exist_ok=True)

# -- Load raw data --
print("Loading raw data...")
detailed = pd.read_csv(f"{RAW}/MRegularSeasonDetailedResults.csv")
compact  = pd.read_csv(f"{RAW}/MRegularSeasonCompactResults.csv")
seeds    = pd.read_csv(f"{RAW}/MNCAATourneySeeds.csv")
tourney  = pd.read_csv(f"{RAW}/MNCAATourneyCompactResults.csv")
massey   = pd.read_csv(f"{RAW}/MMasseyOrdinals.csv")
coaches  = pd.read_csv(f"{RAW}/MTeamCoaches.csv")
teams    = pd.read_csv(f"{RAW}/MTeams.csv")

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

    cols = ['Season','DayNum','TeamID','OppID','Score','OppScore','Loc', 'NumOT', 'Won', 
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
detailed_stacked   = stack_detailed(detailed)   # detailed: 2003–2026
compact_stacked = stack_compact(compact)     # compact:  1985–2026
print(f"    detailed_stacked shape: {detailed_stacked.shape}")
print(f"    compact_stacked shape: {compact_stacked.shape}")


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
g['Poss'] = (g['FGA'] - g['OR'] + g['TO'] + (0.44 * g['FTA'])).clip(lower=1)
g['OPoss'] = (g['OFGA'] - g['OOR'] + g['OTO'] + (0.44 * g['OFTA'])).clip(lower=1)

# Efficiency
g['ORtg'] = 100 * (g['Score'] / g['Poss'])
g['DRtg'] = 100 * (g['OppScore'] / g['OPoss'])
g['NetRtg'] = g['ORtg'] - g['DRtg']

# Tempo - accounting for Overtimes
g['GameMinutes'] = 40 + (g['NumOT'] * 5) # Standard OT is 5 minutes of playing time
g['Tempo'] = (g['Poss'] / g['GameMinutes']) * 40
g['OTempo'] = (g['OPoss'] / g['GameMinutes']) * 40

# Dean Oliver's Four Factors
# Offense
g['eFG'] = (g['FGM'] + 0.5 * g['FGM3']) / g['FGA']
g['TOV'] = g['TO'] / (g['FGA'] + 0.44 * g['FTA'] + g['TO'])
g['ORB'] = g['OR'] / (g['OR'] + g['ODR'])
g['FTR'] = g['FTA'] / g['FGA']
# Defense
g['opp_eFG'] = (g['OFGM'] + 0.5 * g['OFGM3']) / g['OFGA']
g['opp_TOV'] = g['OTO'] / (g['OFGA'] + 0.44 * g['OFTA'] + g['OTO'])
g['DRB'] = g['DR'] / (g['OOR'] + g['DR'])
g['opp_FTR'] = g['OFTA'] / g['OFGA']

# Aggregate for the season
season_eff = g.groupby(['Season', 'TeamID']).agg(
    ORtg_mean = ('ORtg', 'mean'),
    DRtg_mean = ('DRtg', 'mean'),
    NetRtg_mean = ('NetRtg', 'mean'),
    Tempo_mean = ('Tempo', 'mean'),
    eFG_mean = ('eFG', 'mean'),
    TOV_mean = ('TOV', 'mean'),
    ORB_mean = ('ORB', 'mean'),
    FTR_mean = ('FTR', 'mean'),
    opp_eFG_mean = ('opp_eFG', 'mean'),
    opp_TOV_mean = ('opp_TOV', 'mean'),
    DRB_mean = ('DRB', 'mean'),
    opp_FTR_mean = ('opp_FTR', 'mean'),
    Games = ('Won', 'count'),
    Wins = ('Won', 'sum'),
).reset_index()
season_eff['WinPct'] = season_eff['Wins'] / season_eff['Games']

# Print-out
print(f" Efficiency features: {season_eff.shape[0]} team-seasons")
print(f" Sample - top 5 by Net Rating in 2026:")
print(season_eff[season_eff['Season']==2026]
      .nlargest(5, 'NetRtg_mean')
      .merge(teams[['TeamID', 'TeamName']], on='TeamID')
      [['TeamName', 'WinPct', 'NetRtg_mean', 'ORtg_mean', 'DRtg_mean', 'Tempo_mean']].to_string(index=False))


# =======================================================================
# SECTION 2 — End-of-Season Momentum
# =======================================================================
# Regular season ends around DayNum 132. We can define  
# End of season = DayNum >= 95
#
# Here we'll compute efficiency stats for late-season games and take the difference, or *delta*,
# vs. a given team's full-season average. Positive delta = team is improving into the NCAA tournament
# We also use this section to calculate a team's win streak entering the tournament.

print("\n Computing Momentum Features...")

# Establish and apply cutoff
LATE_CUTOFF = 95
late = g[g['DayNum'] >= LATE_CUTOFF].copy()

# Calculate efficiencies for end-of-season segment
late_eff = late.groupby(['Season','TeamID']).agg(
    ORtg_late    = ('ORtg',   'mean'),
    DRtg_late    = ('DRtg',   'mean'),
    NetRtg_late  = ('NetRtg', 'mean'),
    Games_late = ('Won',  'count'),
    Wins_late  = ('Won',  'sum'),
).reset_index()
late_eff['WinPct_late'] = late_eff['Wins_late'] / late_eff['Games_late']

# Merge with season averages and compute deltas
momentum = season_eff[['Season','TeamID','ORtg_mean','DRtg_mean','NetRtg_mean', 'WinPct']].merge(
    late_eff, on=['Season','TeamID'], how='left'
)
momentum['ORtg_delta']   = momentum['ORtg_late']    - momentum['ORtg_mean']
momentum['DRtg_delta']   = momentum['DRtg_late']    - momentum['DRtg_mean']
momentum['NetRtg_delta'] = momentum['NetRtg_late']  - momentum['NetRtg_mean']
momentum['WinPct_delta'] = momentum['WinPct_late']  - momentum['WinPct']

momentum = momentum[['Season','TeamID',
                     'ORtg_late','DRtg_late','NetRtg_late','WinPct_late', 
                     'ORtg_delta','DRtg_delta','NetRtg_delta', 'WinPct_delta',
                     'Games_late']]

# Win streak leading into the tourney
# Find consecutive wins counting backwards from the last regular season game (including conference tournaments)
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

print("  Computing win streaks (Should take ~30 seconds)...")
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
      .nlargest(5, 'WinStreak')
      .merge(teams[['TeamID', 'TeamName']], on='TeamID')
      [['TeamName', 'WinStreak', 'NetRtg_delta', 'WinPct_delta']].to_string(index=False))


# =======================================================================
# SECTION 3 — Elo Ratings
# =======================================================================

# Here I'm looking to create a dynamic rating system where every game updates both teams' ratings.
# Starting Elo = 1500 for all teams.
# Ratings will carry over season-to-season, with a built-in regression to the mean.
# I'll also use a margin-of-victory multiplier so the magnitude of wins are taken into account.
#
# K = base learning rate (how much rating shifts / game)
# MoV multiplier = log(|score_diff| + 1) * (2.2 / ((elo_w - elo_l)) * 0.001 + 2.2))
#
# The log(|score_diff) + 1) portion makes it so a 30-point win counts for more than a 5-point win, but not proportionally more. The +1 prevents 1-point wins from resulting in a 0 multiplier
#
# (2.2 / ((elo_w - elo_l)) * 0.001 + 2.2) allows us to adjust rating change based on level of competition. 
# Higher rated teams are expected to win and when they do will result in a multiplier less than 1. Lower rated teams are not expected to win and when they do will result in multiplier greater than 1.

print("\n Computing Elo Ratings...")

K         = 20      # Default rating shift / game
ELO_INIT  = 1500    # Initial rating for all teams
REGRESS   = 0.75    # Season-to-season regression to the mean (keep 75% of prior rating difference)

# Work from compact (goes back to 1985, gives more history)
elo_games = compact[['Season','DayNum','WTeamID','LTeamID','WScore','LScore']].copy()
elo_games = elo_games.sort_values(['Season','DayNum']).reset_index(drop=True)

elo_ratings = {}   # {TeamID: current_elo}
season_end_elo = {}  # {(Season, TeamID): elo at end of RS}
current_season = None

for _, row in elo_games.iterrows():
    season  = row['Season']
    wid     = int(row['WTeamID'])
    lid     = int(row['LTeamID'])
    wscore  = row['WScore']
    lscore  = row['LScore']

    # Season transition — regress ratings toward mean
    if season != current_season:
        if current_season is not None:
            # Save end-of-season ratings before regression
            for tid, elo in elo_ratings.items():
                season_end_elo[(current_season, tid)] = elo
        # Regress toward mean for new season
        for tid in elo_ratings:
            elo_ratings[tid] = ELO_INIT + REGRESS * (elo_ratings[tid] - ELO_INIT)
        current_season = season

    # Initialize unseen teams
    if wid not in elo_ratings:
        elo_ratings[wid] = ELO_INIT
    if lid not in elo_ratings:
        elo_ratings[lid] = ELO_INIT

    elo_w = elo_ratings[wid]
    elo_l = elo_ratings[lid]

    # Expected Win Prob. for winner
    exp_w = 1 / (1 + 10 ** ((elo_l - elo_w) / 400)) # 400 is a scaling constant - 400-point Elo gap corresponds to ~ 10:1 odds.

    # Margin of Victory multiplier
    score_diff = abs(wscore - lscore)
    mov_mult   = np.log(score_diff + 1) * (2.2 / ((elo_w - elo_l) * 0.001 + 2.2))
    mov_mult   = max(mov_mult, 0.1)  # floor to prevent weird edge cases

    # Update ratings
    # (1 - exp_w) is the expectation a team loses - barely moves K when winner was favored, shifts K greatly when an upset occurs
    delta = K * mov_mult * (1 - exp_w)
    elo_ratings[wid] += delta
    elo_ratings[lid] -= delta

# Save end-of-season elo ratings
for tid, elo in elo_ratings.items():
    season_end_elo[(current_season, tid)] = elo

# Convert to DataFrame
elo_df = (pd.DataFrame.from_dict(season_end_elo, orient='index', columns=['Elo'])
          .reset_index())
elo_df[['Season', 'TeamID']] = pd.DataFrame(elo_df['index'].tolist(), index=elo_df.index)
elo_df = elo_df[['Season', 'TeamID', 'Elo']]

print(f"  Elo ratings: {elo_df.shape[0]} team-seasons")
print(f"  Top 10 teams by Elo in 2026:")

print(elo_df[elo_df['Season']==2026]
      .nlargest(10,'Elo')
      .merge(teams[['TeamID', 'TeamName']], on = 'TeamID')
      [['TeamName','Elo']].to_string(index=False))


# =======================================================================
# SECTION 4 — Seeding
# =======================================================================
# Seeds are stored as strings like 'W01', 'Y11a' (play-in teams get an a/b suffix).
# We extract the numeric seed (1-16). Play-in teams receive a 0.5 to indicate their condition on winning a play-in game.

print("\n Processing Tournament Seeding...")

def parse_seed(s):
    """Extract numeric seed from strings like 'W01', 'Y11a', and 'Z16'."""
    digits = ''.join(filter(str.isdigit, s))
    suffix = s[-1] if s[-1] in ('a', 'b') else ''
    return int(digits) + (0.5 if suffix else 0)

seeds_clean = seeds.copy()
seeds_clean['SeedNum'] = seeds_clean['Seed'].apply(parse_seed)
seeds_clean['Region']  = seeds_clean['Seed'].str[0]  # W, X, Y, Z
seeds_clean = seeds_clean[['Season','TeamID','SeedNum','Region']]

print(f"  Seeds: {seeds_clean.shape[0]} team-seasons")
print(f"  2026 seeds sample:")
print(seeds_clean[seeds_clean['Season']==2026].head(8).to_string(index=False))


# =======================================================================
# SECTION 5 — Massey Ordinals
# =======================================================================
# We use the final pre-tournament ranking (RankingDayNum = 133) from the following systems:
#   POM = KenPom - gold standard for college basketball efficiency ranking (2003-2026) 
#   MAS = Massey - provided data for the Kaggle competition (2003-2026)
#   RPI = NCAA's official ranking system before 2019 (2003-2026)
#   NET = NCAA's current official ranking, replaced RPI in 2019 (2019-2026)
# 
# *Note NET is included despite partial coverage as XGBoost should handle the NaNs well.
# 
# Throughout all Massey Ordinal systems, a lower rank = better team, with 1 being the best-ranked team in the country. 
# I will compute a simple average composite rank from among the systems.

print("\n Processing Massey ordinals...")

SYSTEMS = ['POM', 'MAS', 'RPI', 'NET']

# Gather end-of-season rankings for each system
massey_final = (massey[massey['SystemName'].isin(SYSTEMS)]
                .groupby(['Season','TeamID','SystemName'])
                .apply(lambda x: x.loc[x['RankingDayNum'].idxmax()],
                       include_groups=False)
                .reset_index()
                [['Season','TeamID','SystemName','OrdinalRank']])

# Pivot table with Season and Team ID as indices, System Rankings as columns
massey_wide = massey_final.pivot_table(
    index=['Season','TeamID'],
    columns='SystemName',
    values='OrdinalRank'
).reset_index()
massey_wide.columns.name = None
# Rename system ranking columns to Rank_Pom, Rank_MAS, and so forth
massey_wide.columns = ['Season','TeamID'] + [f'Rank_{s}' for s in massey_wide.columns[2:]]
rank_cols = [c for c in massey_wide.columns if c.startswith('Rank_')]

# Composite rank - average of POM, MAS, and RPI rankings
# We exclude NET here because it would result in NaNs for all teams before 2019
composite_systems = ['Rank_POM', 'Rank_MAS', 'Rank_RPI']
massey_wide['Rank_composite'] = massey_wide[composite_systems].mean(axis=1)

# Sanity checks
print(f"  Massey features: {massey_wide.shape[0]} team-seasons")
all_systems = massey_wide[rank_cols].notna().all(axis=1).sum()
core_three = massey_wide[composite_systems].notna().all(axis=1).sum()
print(f"  Teams with all 4 systems: {all_systems} of {len(massey_wide)}")
print(f"  Teams with core 3 (POM/MAS/RPI): {core_three} of {len(massey_wide)}")

# Sample for 2026
print(f"  2026 sample (top 5 by composite rank):")
print(massey_wide[massey_wide['Season']==2026]
      .nsmallest(5,'Rank_composite')
      .merge(teams[['TeamID','TeamName']], on='TeamID')
      [['TeamName','Rank_MAS','Rank_POM','Rank_RPI','Rank_NET','Rank_composite']]
      .to_string(index=False))


# =======================================================================
# SECTION 6 — Quadrant 1 Record
# =======================================================================
# The NCAA uses Quadrants to organize the quality of wins/losses based on game location and NET rankings.  
# Quadrant 1 is defined as:
#   Home games vs. teams ranked 1–30
#   Away games vs. teams ranked 1–50
#   Neutral games vs. teams ranked 1–75
#
# Here I'll use the composite rank previously built to run the same logic.
# I'm using the end-of-season composite rank as a proxy for in-season rank —
# this is a known simplication documented for future improvement.
# The NCAA also says they consider Quadrant 3/4 losses "incredibly important when it comes time for NCAA tournament selection and seeding". 
# Admittedly I'm limiting my scope to Q1 wins and could add Q3/Q4 losses as a future improvement.

print("\n Computing Quadrant 1 records...")

# Get our opponent composite rank from the Massey Ordinals
opp_rank = massey_wide[['Season','TeamID','Rank_composite']].rename(
    columns={'TeamID':'OppID','Rank_composite':'OppRank'}
)

# Merge opponent ranks into compact long format
q1_data = compact_stacked.merge(opp_rank, on=['Season','OppID'], how='left')

# Classify games as Q1 or not based on opponent rank and location
def is_q1(row):
    r = row['OppRank']
    if pd.isna(r):
        return False
    loc = row['Loc']
    if loc == 'H':      # Home
        return r <= 30
    elif loc == 'A':    # Away
        return r <= 50
    else:               # Neutral site
        return r <= 75

# Apply is_q1 and filter for Q1 games
q1_data['IsQ1'] = q1_data.apply(is_q1, axis=1)
q1_games = q1_data[q1_data['IsQ1']]

# Calculate Q1 record, win %
q1_record = q1_games.groupby(['Season','TeamID']).agg(
    Q1_wins   = ('Won', 'sum'),
    Q1_games  = ('Won', 'count'),
).reset_index()
q1_record['Q1_losses'] = q1_record['Q1_games'] - q1_record['Q1_wins']
q1_record['Q1_winpct'] = q1_record['Q1_wins'] / q1_record['Q1_games'].clip(lower=1)

# Print-out
print(f"  Q1 records: {q1_record.shape[0]} team-seasons")
print(f"  2026 sample (top 5 by Q1 win%):")
print(q1_record[q1_record['Season']==2026]
      .query('Q1_games >= 5')  # minimum 5 Q1 games for a meaningful rate
      .nlargest(5,'Q1_winpct')
      .merge(teams[['TeamID','TeamName']], on='TeamID')
      [['TeamName','Q1_wins','Q1_losses','Q1_winpct']].to_string(index=False))


# =======================================================================
# SECTION 7 — Coaching Features
# =======================================================================
# For each team-season, we find the head coach at tournament time (DayNum ~132) and compute:
#   - Career tournament appearances
#   - Total prior tournament games coached
#   - Career tournament win% entering this season
#   - Years coaching this specific team
#
# *Note on data leakage: we use cumulative stats BEFORE the current season.

print("\n Computing coaching features...")

# Get coach active at tournament time (LastDayNum ~ 132)
coaches_active = coaches[coaches['LastDayNum'] >= 100].copy()

# Tournament results in long format so each team has one row per game
tourney_long = stack_compact(tourney)

# Join coach names on to each tourney game
tourney_with_coach = tourney_long.merge(
    coaches_active[['Season','TeamID','CoachName']],
    on=['Season','TeamID'], how='left'
)

# Group all tournament games together into one row per coach-season
coach_tourney = (tourney_with_coach
                 .groupby(['CoachName','Season'])
                 .agg(T_wins=('Won','sum'), T_games=('Won','count'))
                 .reset_index()
                 .sort_values(['CoachName','Season']))

# Career tournament record per coach 
# Subtracting current year record from cumulative record to avoid data leakage
coach_tourney['Career_T_wins']  = coach_tourney.groupby('CoachName')['T_wins'].cumsum()  - coach_tourney['T_wins']
coach_tourney['Career_T_games'] = coach_tourney.groupby('CoachName')['T_games'].cumsum() - coach_tourney['T_games']
coach_tourney['Career_T_winpct'] = (
    coach_tourney['Career_T_wins'] / coach_tourney['Career_T_games'].clip(lower=1)
)
coach_tourney['Career_T_appearances'] = (
    coach_tourney.groupby('CoachName').cumcount()  # 0 before first appearance
)

# Years with team
coaches_sorted = coaches_active.sort_values(['TeamID','CoachName', 'Season'])
coaches_sorted['YearsWithTeam'] = (coaches_sorted.groupby(['TeamID','CoachName']).cumcount() + 1)

# For each team-season, get the coach's career tournament stats entering that season
coach_career = coach_tourney[['CoachName','Season','Career_T_wins',
                               'Career_T_games','Career_T_winpct',
                               'Career_T_appearances']].copy()

# Add current season numbers to get post-season totals
coach_career['Career_T_wins']        = coach_career['Career_T_wins']  + coach_tourney['T_wins']
coach_career['Career_T_games']       = coach_career['Career_T_games'] + coach_tourney['T_games']
coach_career['Career_T_appearances'] = coach_career['Career_T_appearances'] + 1
coach_career['Career_T_winpct']      = (coach_career['Career_T_wins'] / 
                                        coach_career['Career_T_games'].clip(lower=1))

# Shift forward by 1 — tourney stats from season N become the features for season N+1
coach_career['Season'] = coach_career['Season'] + 1

# Final coaching feature table
coach_features = (coaches_active[['Season','TeamID','CoachName']]
                  .merge(coach_career,
                         on=['CoachName','Season'], how='left')
                  .merge(coaches_sorted[['Season','TeamID','CoachName','YearsWithTeam']],
                         on=['Season','TeamID','CoachName'], how='left'))
coach_features = coach_features[['Season','TeamID', 'CoachName','Career_T_wins','Career_T_games',
                                  'Career_T_winpct','Career_T_appearances','YearsWithTeam']]

# Fill NaNs for coaches with no prior tourney experience
fill_cols = ['Career_T_wins','Career_T_games','Career_T_winpct',
                'Career_T_appearances']
coach_features[fill_cols] = coach_features[fill_cols].fillna(0)

# Print-out 
# We set Season == 2025 because we don't have data for the 2026 tourney yet.
# Season == 2026 would return weird results due to NaNs placeholding all 2026 coach stats
print(f"  2026 Coaches sample:")
coach_sample = (coach_features[coach_features['Season']==2026]
                .merge(teams[['TeamID', 'TeamName']], on='TeamID'))

print("\n  Top 5 by career tournament appearances:")
print(coach_sample.nlargest(5, 'Career_T_appearances')
      [['TeamName', 'CoachName', 'Career_T_appearances', 'Career_T_wins', 'Career_T_winpct', 'YearsWithTeam']].to_string(index=False))

print("\n  Top 5 by career tournament wins:")
print(coach_sample.nlargest(5,'Career_T_wins')
      [['TeamName','CoachName','Career_T_appearances','Career_T_wins','Career_T_winpct','YearsWithTeam']]
      .to_string(index=False))

print("\n  Top 5 by career tournament win %  (min 5 appearances):")
print(coach_sample[coach_sample['Career_T_appearances'] >= 5]
      .nlargest(5,'Career_T_winpct')
      [['TeamName','CoachName','Career_T_appearances','Career_T_wins','Career_T_winpct','YearsWithTeam']]
      .to_string(index=False))


# =======================================================================
# SECTION 8 — Performance Variance Features
# =======================================================================
# Three complementary variance features:
#   1) Opponent-adjusted point differential std dev
#      — Overall volatility in point differential controlling for opponent strength
#
#   2) Overperformance rate
#      — When a team is the underdog by Elo, how often do they beat expectations?
#      — Proxy for upset potential / punching above weight class
#
#   3) Underperformance rate
#      — When a team is favored by Elo, how often do they underperform?
#      — Proxy for upset risk / vulnerability as a favorite
#
# For 2 and 3, "beats/fails" expectations = actual margin exceeds/falls short of expected margin based on Elo.
# 
# *Note: end-of-season Elo once again used as a proxy for in-season elo.
# True in-season elo would be more precise — potential future improvement.
# Another improvement would be weighing over/underperformance rate based on pre-game win probability to better reflect magnitude of upset.

print("\n Computing variance features...")

# Merge Elo into compact_stacked for team and opponent
elo_team = elo_df.rename(columns={'Elo': 'TeamElo'})
elo_opp  = elo_df.rename(columns={'TeamID': 'OppID',  'Elo': 'OppElo'})

var_data = compact_stacked.copy()
var_data['PointDiff'] = var_data['Score'] - var_data['OppScore']
var_data = var_data.merge(elo_team, on=['Season','TeamID'], how='left')
var_data = var_data.merge(elo_opp,  on=['Season','OppID'],  how='left')

# Expected margin based on Elo
# Simple linear scaling — each 25 Elo points ≈ 1 point
var_data['EloDiff']         = var_data['TeamElo'] - var_data['OppElo']
var_data['ExpectedMargin']  = var_data['EloDiff'] / 25
var_data['AdjMargin']       = var_data['PointDiff'] - var_data['ExpectedMargin']

# 1) Overall opponent-adjusted volatility
vol = var_data.groupby(['Season','TeamID']).agg(
    AdjMargin_std  = ('AdjMargin', 'std'),
    AdjMargin_mean = ('AdjMargin', 'mean'),  # positive = generally outperforms Elo
).reset_index()

# 2/3) Directional Over/Underperformance 
# Split into games where team was underdog vs. favorite by Elo
underdog_games  = var_data[var_data['EloDiff'] < 0].copy()
favorite_games  = var_data[var_data['EloDiff'] > 0].copy()

# Counts for underdog/favorite games
underdog_counts = (underdog_games.groupby(['Season', 'TeamID'])
                   .size().reset_index(name='UnderdogGames'))
favorite_counts = (favorite_games.groupby(['Season', 'TeamID'])
                   .size().reset_index(name='FavoriteGames'))

# Overperformance: 
# Underdog games where actual margin > expected margin
# (i.e., AdjMargin > 0 — underdog beat expectations)
overperf = (underdog_games
            .groupby(['Season','TeamID'])
            .apply(lambda x: (x['AdjMargin'] > 0).mean(), include_groups=False)
            .reset_index()
            .rename(columns={0: 'OverperfRate'}))

# Average magnitude of overperformance
overperf_mag = (underdog_games[underdog_games['AdjMargin'] > 0]
                .groupby(['Season','TeamID'])['AdjMargin']
                .mean().reset_index()
                .rename(columns={'AdjMargin': 'OverperfMagnitude'}))

# Minimum number of underdog games to generate reliable overperformance rates
overperf = overperf.merge(underdog_counts, on=['Season', 'TeamID'])
overperf.loc[overperf['UnderdogGames'] < 5, 'OverperfRate'] = np.nan
overperf = overperf.drop(columns='UnderdogGames')

# Underperformance: 
# Favorite games where actual margin < expected margin
# (i.e., AdjMargin < 0 — favorite missed expectations)
underperf = (favorite_games
             .groupby(['Season','TeamID'])
             .apply(lambda x: (x['AdjMargin'] < 0).mean(),include_groups=False)
             .reset_index()
             .rename(columns={0: 'UnderperfRate'}))

# Average magnitude of underperformance
underperf_mag = (favorite_games[favorite_games['AdjMargin'] < 0]
                 .groupby(['Season','TeamID'])['AdjMargin']
                 .mean().reset_index()
                 .rename(columns={'AdjMargin': 'UnderperfMagnitude'}))

# Minimum number of favored games to generate reliable overperformance rates
underperf = underperf.merge(favorite_counts, on=['Season','TeamID'])
underperf.loc[underperf['FavoriteGames'] < 5, 'UnderperfRate'] = np.nan
underperf = underperf.drop(columns='FavoriteGames')

# Assemble variance feature table
var_features = (vol
                 .merge(overperf,      on=['Season','TeamID'], how='left')
                 .merge(underperf,     on=['Season','TeamID'], how='left')
                 .merge(overperf_mag,  on=['Season','TeamID'], how='left')
                 .merge(underperf_mag, on=['Season','TeamID'], how='left'))

# Fill NaNs for teams with no underdog/favorite games
fill_var = ['OverperfMagnitude','UnderperfMagnitude']
var_features[fill_var] = var_features[fill_var].fillna(0)

print(f"  Variance features: {var_features.shape[0]} team-seasons")
print(f"\n  Top 5 teams by OverperfRate in 2025 (Cinderella candidates):")
print(var_features[var_features['Season']==2025]
      .nlargest(5,'OverperfRate')
      .merge(teams[['TeamID','TeamName']], on='TeamID')
      [['TeamName','OverperfRate','OverperfMagnitude','AdjMargin_std']]
      .to_string(index=False))

print(f"\n  Top 5 teams by UnderperfRate in 2025 (trap picks):")
print(var_features[var_features['Season']==2025]
      .nlargest(5,'UnderperfRate')
      .merge(teams[['TeamID','TeamName']], on='TeamID')
      [['TeamName','UnderperfRate','UnderperfMagnitude','AdjMargin_std']]
      .to_string(index=False))


# =======================================================================
# SECTION 9 — Put it all together!
# =======================================================================
# Join all feature groups on (Season, TeamID).
# Base table = tournament seeds (one row per tournament team per season)
# All joins are left joins - missing values left as NaN for XGBoost to handle

print("\n Assembling master feature table...")

# Base table
team_features = seeds_clean.copy()

joins = [
    (season_eff,    'efficiency'),
    (momentum,      'momentum'),
    (elo_df,        'elo'),
    (massey_wide,   'massey'),
    (q1_record,     'q1'),
    (coach_features,'coaching'),
    (var_features,  'variance'),
]

# Loop to merge each of our feature tables
for df, name in joins:
    team_features = team_features.merge(df, on=["Season", 'TeamID'], how='left')
    print(f"  After {name}: {team_features.shape[1]} columns")

# Sanity checks
# *Note: NaNs are expected here as our sample includes seasons from 1985 to 2002 for which we don't have detailed data
print(f"\nFinal feature table shape: {team_features.shape}")
print(f"Seasons covered: {team_features['Season'].min()} – {team_features['Season'].max()}")
print(f"Null counts (columns with nulls):")
null_counts = team_features.isnull().sum()
print(null_counts[null_counts > 0].sort_values(ascending=False).to_string())

# Save
out_path = f"{PROCESSED}/M_team_features.csv"
team_features.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")
print("Feature engineering complete. Bang!")