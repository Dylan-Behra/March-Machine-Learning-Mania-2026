# March Machine Learning Mania 2026

My submission to the [March Machine Learning Mania 2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026) Kaggle competition — predicting the outcomes of every possible NCAA Men's and Women's basketball tournament matchup.

When I first signed up for this competition, I told a friend I was planning to build an XGBoost model, what most competitors use and what previous winners have relied on. He asked: *"Well if that's the case, how do people differentiate themselves?"*

Great question, Thomas. My hope? The feature engineering.

---

## Approach

The model predicts win probability for any two-team matchup using **feature differences** — each matchup is represented as Team A's features minus Team B's features — then trains an XGBoost classifier on historical tournament results.

Separate pipelines and models are trained for the **Men's** and **Women's** brackets due to differences in data availability, historical trends, and results.

### Feature Engineering

Both brackets share a common feature set:

| Feature Group | Details |
|---|---|
| **Efficiency Stats** | Offensive Rating, Defensive Rating, Net Rating, Tempo; Dean Oliver's Four Factors (eFG%, TOV%, ORB%, FTR) |
| **End-of-Season Momentum** | Late-season efficiency delta vs. season average; win streak entering tournament |
| **Elo Rating** | Dynamic rating system (K=20, init=1500, 75% season-to-season regression) built from full game history |
| **NCAA Seed** | Numeric seed (1–16) with play-in handling |
| **Quadrant 1 Record** | Wins/losses vs. top-tier opponents based on opponent strength and game location |
| **Variance / Consistency** | Opponent-adjusted margin standard deviation; overperformance and underperformance rates |

Men's bracket adds:

| Feature Group | Details |
|---|---|
| **Massey Ordinals** | KenPom (POM), Massey (MAS), RPI, NET rankings + composite rank |
| **Coaching Features** | Career tournament appearances, wins, win rate, years with current team |

Women's bracket substitutes:

| Feature Group | Details |
|---|---|
| **Strength of Schedule** | Custom SOS ranking proxy using opponent win% and net rating (Massey Ordinals unavailable for women's) |

> **Women's basketball deserves better data!**

### Model

- **Algorithm**: XGBoost (`binary:logistic`)
- **Hyperparameters**: 400 estimators, LR=0.03, max depth=4, subsample=0.8, min child weight=5, L1/L2 regularization
- **Validation**: Walk-forward cross-validation from 2010 onward (expanding window, one season held out at a time), reporting log-loss and Brier score per season
- **Predictions clipped** to [0.025, 0.975] to avoid boundary overconfidence

---

## Project Structure

```
├── src/
│   ├── unzip_data.py        # Extract Kaggle competition zip
│   ├── M_features.py        # Men's feature engineering pipeline
│   ├── W_features.py        # Women's feature engineering pipeline
│   └── model.py             # XGBoost modeling, CV, and submission generation
├── notebooks/
│   ├── EDA.ipynb            # Exploratory data analysis
│   └── bracket_explorer.ipynb  # Interactive bracket exploration
├── data/
│   ├── raw/                 # Kaggle competition data (gitignored)
│   └── processed/           # Generated feature tables (gitignored)
├── submissions/
│   └── submission.csv       # Stage 2 predictions
└── requirements.txt
```

## How to Run

```bash
pip install -r requirements.txt

# 1. Extract competition data (if starting from zip)
python src/unzip_data.py

# 2. Build feature tables
python src/M_features.py
python src/W_features.py

# 3. Train models and generate submission
python src/model.py
```

> Raw and processed data are gitignored per Kaggle's IP guidelines. Download competition data from Kaggle and place the zip in `data/raw/`.

---

## Limitations

- **No injury data**: Player-level box scores are not provided by the competition, making it impossible to capture in-game player availability. A proxy approach would require rotation tracking that the data doesn't support.
- **Women's data gaps**: No Massey Ordinals or coaching records are available for the women's bracket.