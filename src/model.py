"""
model.py — March Machine Learning Mania 2026
XGBoost model: matchup construction, walk-forward cross-validation, and submission generation.
 
Pipeline overview:
    1. Load team feature tables for Men's (M) and Women's (W) brackets
    2. Build matchup dataset from historical tournament games
       - Each row is a game expressed as Team A vs. Team B feature *differences*
       - Team ordering is randomized to avoid position bias (lower ID is not always "Team A")
    3. Walk-forward cross-validation (train on seasons < T, evaluate on season T)
       - Reports per-season log-loss and Brier score, plus overall
    4. Train final models on all historical data
    5. Generate submission file using SampleSubmission.csv as the ID template
 
Output: submission.csv  (ID, Pred columns)
 
Notes on feature construction:
    - All matchup features are *differences*: feat_A - feat_B
    - Seed difference is included as a raw feature (higher seed = worse team)
    - For metrics where lower = better (DRtg, TOV, rankings), the difference
      still points in the same direction once you take A - B
    - NaN handling: XGBoost's native missing-value support handles remaining NaNs;
      no imputation needed for tree models
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss, brier_score_loss
import warnings
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------
BASE      = Path(__file__).resolve().parent.parent
RAW       = str(BASE / "data" / "raw")
PROCESSED = str(BASE / "data" / "processed")
OUT       = str(BASE / "submissions")

os.makedirs(OUT, exist_ok=True)

# Walk-forward CV: first holdout season (all seasons before this are initial train)
CV_START_SEASON = 2010
 
# XGBoost hyperparameters — tuned conservatively to avoid overfitting on small tournament samples
XGB_PARAMS = {
    'objective':        'binary:logistic',
    'eval_metric':      'logloss',
    'n_estimators':     400,
    'learning_rate':    0.03,
    'max_depth':        4,
    'subsample':        0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,       # prevents splits on very few tournament games
    'gamma':            1.0,     # minimum loss reduction for a split
    'reg_alpha':        0.1,     # L1 regularization
    'reg_lambda':       2.0,     # L2 regularization
    'random_state':     42,
    'n_jobs':           -1,
    'tree_method':      'hist',  # fast histogram method
}
 
# Features that exist only in the men's pipeline (Massey rankings + coaching)
M_ONLY_FEATURES = [
    'Rank_MAS', 'Rank_NET', 'Rank_POM', 'Rank_RPI', 'Rank_composite',
    'Career_T_wins', 'Career_T_games', 'Career_T_winpct',
    'Career_T_appearances', 'YearsWithTeam',
]
 
# Features that exist only in the women's pipeline (SOS proxies)
W_ONLY_FEATURES = ['SOS_WinPct','SOS_NetRtg']
 
# -----------------------------------------------------------------------
# SECTION 1 — Feature column list
# -----------------------------------------------------------------------
# Base features shared by both brackets
BASE_FEATURES = [
    # Identity / seeding
    'SeedNum',
    # Efficiency
    'ORtg_mean', 'DRtg_mean', 'NetRtg_mean', 'Tempo_mean',
    'eFG_mean', 'TOV_mean', 'ORB_mean', 'FTR_mean',
    'opp_eFG_mean', 'opp_TOV_mean', 'DRB_mean', 'opp_FTR_mean',
    'WinPct',
    # Momentum
    'ORtg_late', 'DRtg_late', 'NetRtg_late', 'WinPct_late',
    'ORtg_delta', 'DRtg_delta', 'NetRtg_delta', 'WinPct_delta',
    'Games_late', 'WinStreak',
    # Elo
    'Elo',
    # Q1 record
    'Q1_wins', 'Q1_games', 'Q1_losses', 'Q1_winpct',
    # Coaching (men's only — WTeamCoaches.csv not available in this dataset)
    # Moved to M_ONLY_FEATURES
    # Variance
    'AdjMargin_std', 'AdjMargin_mean',
    'OverperfRate', 'UnderperfRate',
    'OverperfMagnitude', 'UnderperfMagnitude',
]
 
 
def get_feature_cols(gender: str) -> list[str]:
    """Return the feature list for M or W, including gender-specific columns."""
    cols = BASE_FEATURES.copy()
    if gender == 'M':
        cols += M_ONLY_FEATURES
    else:
        cols += W_ONLY_FEATURES
    return cols
 
 
# -----------------------------------------------------------------------
# SECTION 2 — Matchup dataset construction
# -----------------------------------------------------------------------
 
def build_matchup_dataset(tourney_df: pd.DataFrame,
                          team_features: pd.DataFrame,
                          feature_cols: list[str],
                          seed: int = 42) -> pd.DataFrame:
    """
    Build a matchup dataset from historical tournament games.
 
    Each game becomes one row of feature differences (Team A - Team B).
    Team A / Team B assignment is randomized to prevent the model from
    learning a spurious 'lower TeamID wins more' signal.
 
    Parameters
    ----------
    tourney_df    : compact tournament results (Season, WTeamID, LTeamID)
    team_features : per-team per-season feature table
    feature_cols  : list of numeric features to difference
    seed          : random seed for team-order randomization
 
    Returns
    -------
    DataFrame with columns:
        Season, TeamA, TeamB, Label (1 if A won, 0 if B won),
        + one diff column per feature (feat_A - feat_B)
    """
    rng = np.random.default_rng(seed)
    rows = []
 
    for _, game in tourney_df.iterrows():
        season = game['Season']
        w_id   = int(game['WTeamID'])
        l_id   = int(game['LTeamID'])
 
        # Randomly assign A/B to winner/loser
        if rng.random() < 0.5:
            team_a, team_b, label = w_id, l_id, 1
        else:
            team_a, team_b, label = l_id, w_id, 0
 
        row = {'Season': season, 'TeamA': team_a, 'TeamB': team_b, 'Label': label}
        rows.append(row)
 
    matchups = pd.DataFrame(rows)
 
    # Pull team features for A and B
    feats = team_features[['Season','TeamID'] + feature_cols].copy()
 
    matchups = matchups.merge(
        feats.rename(columns={'TeamID':'TeamA', **{c: f'{c}_A' for c in feature_cols}}),
        on=['Season','TeamA'], how='left'
    )
    matchups = matchups.merge(
        feats.rename(columns={'TeamID':'TeamB', **{c: f'{c}_B' for c in feature_cols}}),
        on=['Season','TeamB'], how='left'
    )
 
    # Compute differences (A - B)
    for col in feature_cols:
        matchups[f'diff_{col}'] = matchups[f'{col}_A'] - matchups[f'{col}_B']
 
    # Drop raw A/B columns — model sees only diffs
    drop_cols = [f'{c}_A' for c in feature_cols] + [f'{c}_B' for c in feature_cols]
    matchups = matchups.drop(columns=drop_cols)
 
    return matchups
 
 
def build_prediction_row(season: int,
                         team_a: int,
                         team_b: int,
                         team_features: pd.DataFrame,
                         feature_cols: list[str]) -> pd.DataFrame:
    """
    Build a single prediction row for a given matchup.
    Always computes features as TeamA - TeamB (lower ID = A in submission format).
    """
    feats = team_features[['Season','TeamID'] + feature_cols].copy()
 
    row_a = feats[(feats['Season']==season) & (feats['TeamID']==team_a)].squeeze()
    row_b = feats[(feats['Season']==season) & (feats['TeamID']==team_b)].squeeze()
 
    diffs = {}
    for col in feature_cols:
        val_a = row_a[col] if col in row_a.index else np.nan
        val_b = row_b[col] if col in row_b.index else np.nan
        diffs[f'diff_{col}'] = val_a - val_b
 
    return pd.DataFrame([diffs])
 
 
# -----------------------------------------------------------------------
# SECTION 3 — Walk-forward cross-validation
# -----------------------------------------------------------------------
 
def walk_forward_cv(matchup_df: pd.DataFrame,
                    diff_cols: list[str],
                    cv_start: int = CV_START_SEASON) -> dict:
    """
    Walk-forward (expanding window) cross-validation.
 
    For each season T >= cv_start:
        Train on all seasons < T
        Predict season T
        Record log-loss and Brier score
 
    Returns a dict with per-season results and aggregate metrics.
    """
    seasons = sorted(matchup_df[matchup_df['Season'] >= cv_start]['Season'].unique())
    results = []
 
    print(f"  Walk-forward CV: evaluating seasons {seasons[0]}–{seasons[-1]}")
 
    for season in seasons:
        train = matchup_df[matchup_df['Season'] < season]
        test  = matchup_df[matchup_df['Season'] == season]
 
        if len(train) == 0 or len(test) == 0:
            continue
 
        X_train = train[diff_cols].values
        y_train = train['Label'].values
        X_test  = test[diff_cols].values
        y_test  = test['Label'].values
 
        model = xgb.XGBClassifier(**XGB_PARAMS, verbosity=0)
        model.fit(X_train, y_train)
 
        probs = model.predict_proba(X_test)[:, 1]
        # Clip predictions away from 0/1 to avoid infinite log-loss
        probs = np.clip(probs, 1e-6, 1 - 1e-6)
 
        ll    = log_loss(y_test, probs)
        brier = brier_score_loss(y_test, probs)
 
        results.append({
            'Season': season,
            'N_games': len(test),
            'LogLoss': round(ll,    4),
            'Brier':   round(brier, 4),
        })
 
    cv_df = pd.DataFrame(results)
 
    # Aggregate — weight by number of games
    total_games = cv_df['N_games'].sum()
    wtd_ll    = (cv_df['LogLoss'] * cv_df['N_games']).sum() / total_games
    wtd_brier = (cv_df['Brier']   * cv_df['N_games']).sum() / total_games
 
    return {
        'per_season': cv_df,
        'weighted_logloss': round(wtd_ll,    4),
        'weighted_brier':   round(wtd_brier, 4),
    }
 
 
# -----------------------------------------------------------------------
# SECTION 4 — Training and submission helpers
# -----------------------------------------------------------------------
 
def train_final_model(matchup_df: pd.DataFrame,
                      diff_cols: list[str]) -> xgb.XGBClassifier:
    """Train on all available historical data."""
    X = matchup_df[diff_cols]
    y = matchup_df['Label'].values
    model = xgb.XGBClassifier(**XGB_PARAMS, verbosity=0)
    model.fit(X, y)
    return model
 
 
def print_feature_importance(model: xgb.XGBClassifier,
                              diff_cols: list[str],
                              top_n: int = 15,
                              gender: str = '') -> None:
    """Print top N features by XGBoost gain importance."""
    importance = model.get_booster().get_score(importance_type='gain')
    imp_df = (pd.DataFrame.from_dict(importance, orient='index', columns=['Gain'])
              .reset_index().rename(columns={'index':'Feature'})
              .sort_values('Gain', ascending=False)
              .head(top_n))
    # Strip diff_ prefix for readability
    imp_df['Feature'] = imp_df['Feature'].str.replace('diff_', '', regex=False)
    imp_df['Gain'] = imp_df['Gain'].round(1)
    label = f" [{gender}]" if gender else ""
    print(f"\n  Top {top_n} features by gain{label}:")
    print(imp_df.to_string(index=False))
 
 
def generate_predictions(model: xgb.XGBClassifier,
                         sample_sub: pd.DataFrame,
                         team_features: pd.DataFrame,
                         feature_cols: list[str],
                         gender_prefix: str) -> pd.DataFrame:
    """
    Generate win probability predictions for all matchups in the sample submission
    that belong to the given gender prefix ('M' or 'W').
 
    Submission ID format: Season_TeamA_TeamB  (TeamA < TeamB by convention)
    P(TeamA wins) is predicted.
    """
    if gender_prefix == 'M':
        mask = sample_sub['ID'].apply(
            lambda x: int(x.split('_')[1]) < 3000)
    else:
        mask = sample_sub['ID'].apply(
            lambda x: int(x.split('_')[1]) >= 3000)
        
    sub_ids = sample_sub[mask]['ID'].values
 
    diff_cols = [f'diff_{c}' for c in feature_cols]
    preds = []
 
    for game_id in sub_ids:
        parts  = game_id.split('_')
        season = int(parts[0])
        team_a = int(parts[1])
        team_b = int(parts[2])
 
        # Skip if either team not in feature table (non-tournament teams)
        a_exists = len(team_features[(team_features['Season']==season) & 
                                     (team_features['TeamID']==team_a)]) > 0
        b_exists = len(team_features[(team_features['Season']==season) & 
                                     (team_features['TeamID']==team_b)]) > 0
    
        if not a_exists or not b_exists:
            preds.append({'ID': game_id, 'Pred': 0.5})
            continue

        row = build_prediction_row(season, team_a, team_b, team_features, feature_cols)
        prob = model.predict_proba(row[diff_cols].values)[0, 1]
        prob = float(np.clip(prob, 0.025, 0.975))  # keep well away from boundaries
        preds.append({'ID': game_id, 'Pred': prob})
 
    return pd.DataFrame(preds)
 
 
# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------
 
def run_bracket(gender: str,
                tourney_file: str,
                features_file: str) -> tuple[xgb.XGBClassifier, pd.DataFrame, list[str]]:
    """
    Full pipeline for one bracket (M or W).
 
    Returns (model, team_features_df, diff_cols) for use in submission generation.
    """
    print(f"\n{'='*60}")
    print(f"  {gender.upper()} BRACKET")
    print(f"{'='*60}")
 
    # Load data
    tourney_df    = pd.read_csv(f"{RAW}/{tourney_file}")
    team_features = pd.read_csv(f"{PROCESSED}/{features_file}")
    feature_cols  = get_feature_cols(gender)
    diff_cols     = [f'diff_{c}' for c in feature_cols]
 
    print(f"  Loaded {len(team_features)} team-seasons, "
          f"{len(tourney_df)} tournament games")
 
    # Build matchup dataset
    print("\n  Building matchup dataset...")
    matchup_df = build_matchup_dataset(tourney_df, team_features, feature_cols)
    matchup_df = matchup_df[matchup_df['Season'] >= 2003]  # detailed data starts 2003
    print(f"  Matchup dataset: {len(matchup_df)} games "
          f"({matchup_df['Season'].min()}–{matchup_df['Season'].max()})")
 
    # Null check on diff features
    null_pct = matchup_df[diff_cols].isnull().mean().mean() * 100
    print(f"  Avg null % across diff features: {null_pct:.1f}%  (XGBoost handles natively)")
 
    # Walk-forward CV
    print("\n  Running walk-forward cross-validation...")
    cv = walk_forward_cv(matchup_df, diff_cols)
    print(f"\n  Per-season results:")
    print(cv['per_season'].to_string(index=False))
    print(f"\n  Weighted log-loss: {cv['weighted_logloss']}")
    print(f"  Weighted Brier:    {cv['weighted_brier']}")
 
    # Train final model
    print("\n  Training final model on all data...")
    model = train_final_model(matchup_df, diff_cols)
    print_feature_importance(model, diff_cols, top_n=15, gender=gender)
 
    return model, team_features, feature_cols
 
 
if __name__ == '__main__':
 
    # Run both brackets
    m_model, m_features, m_feat_cols = run_bracket(
        gender        = 'M',
        tourney_file  = 'MNCAATourneyCompactResults.csv',
        features_file = 'M_team_features.csv',
    )
 
    w_model, w_features, w_feat_cols = run_bracket(
        gender        = 'W',
        tourney_file  = 'WNCAATourneyCompactResults.csv',
        features_file = 'W_team_features.csv',
    )
 
    # ------------------------------------------------------------------
    # Generate submission
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  GENERATING SUBMISSION")
    print(f"{'='*60}")
 
    sample_sub = pd.read_csv(f"{RAW}/SampleSubmissionStage2.csv")
    print(f"  Sample submission rows: {len(sample_sub)}")
    print(f"  ID format sample: {sample_sub['ID'].iloc[0]}")
 
    print("\n  Predicting Men's matchups...")
    m_preds = generate_predictions(
        model         = m_model,
        sample_sub    = sample_sub,
        team_features = m_features,
        feature_cols  = m_feat_cols,
        gender_prefix = 'M',
    )
    print(f"  Generated {len(m_preds)} M predictions")
 
    print("\n  Predicting Women's matchups...")
    w_preds = generate_predictions(
        model         = w_model,
        sample_sub    = sample_sub,
        team_features = w_features,
        feature_cols  = w_feat_cols,
        gender_prefix = 'W',
    )
    print(f"  Generated {len(w_preds)} W predictions")
 
    # Combine and align to sample submission order
    all_preds = pd.concat([m_preds, w_preds], ignore_index=True)
    submission = sample_sub[['ID']].merge(all_preds, on='ID', how='left')
 
    # Sanity checks
    n_missing = submission['Pred'].isnull().sum()
    if n_missing > 0:
        print(f"\n  WARNING: {n_missing} rows missing predictions — filling with 0.5")
        submission['Pred'] = submission['Pred'].fillna(0.5)
 
    print(f"\n  Submission shape: {submission.shape}")
    print(f"  Pred range: [{submission['Pred'].min():.4f}, {submission['Pred'].max():.4f}]")
    print(f"  Pred mean:  {submission['Pred'].mean():.4f}  (expect ~0.50)")
    print(f"\n  Sample predictions:")
    print(submission.head(10).to_string(index=False))
 
    out_path = f"{OUT}/submission.csv"
    submission.to_csv(out_path, index=False)
    print(f"\n  Saved to {out_path}")
    print("\nDone. Go fill out that bracket!")
