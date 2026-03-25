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

## 2026 Predictions

Odds generated via 10,000 Monte Carlo simulations of the full bracket using the trained XGBoost models. See [bracket_explorer.ipynb](notebooks/bracket_explorer.ipynb) for the full breakdown by region.

### Men's Championship Contenders

| Seed | Team | Region | Final Four | Finals | Champion |
|---|---|---|---|---|---|
| 1 | Duke | W | 61.2% | 48.5% | **27.7%** |
| 1 | Michigan | Y | 46.6% | 32.2% | 18.8% |
| 1 | Arizona | Z | 39.3% | 21.0% | 11.6% |
| 1 | Florida | X | 44.0% | 14.9% | 8.8% |
| 2 | Houston | X | 26.4% | 10.7% | 6.0% |
| 3 | Gonzaga | Z | 17.9% | 9.3% | 5.5% |
| 2 | Iowa St | Y | 18.8% | 8.9% | 3.8% |
| 2 | Purdue | Z | 23.1% | 10.8% | 3.3% |
| 3 | Virginia | Y | 16.5% | 6.3% | 2.4% |
| 2 | Connecticut | W | 11.2% | 5.7% | 2.1% |

### Men's Cinderella Watch *(seed ≥ 9, P(Sweet 16) ≥ 20%)*

| Seed | Team | Region | R1 | R2 | Sweet 16 | Elite 8 |
|---|---|---|---|---|---|---|
| 11 | South Florida | W | 100% | 65.8% | 26.6% | 8.6% |
| 9 | St Louis | Y | 100% | 60.7% | 22.3% | 9.9% |
| 11 | VCU | X | 100% | 41.7% | 22.8% | 3.7% |

### Women's Championship Contenders

| Seed | Team | Region | Final Four | Finals | Champion |
|---|---|---|---|---|---|
| 1 | Connecticut | W | 80.2% | 66.0% | **47.5%** |
| 1 | UCLA | Z | 73.0% | 54.7% | 23.3% |
| 1 | Texas | Y | 81.8% | 32.2% | 15.2% |
| 1 | South Carolina | X | 83.8% | 29.0% | 10.7% |
| 2 | LSU | Z | 21.6% | 9.7% | 2.4% |
| 2 | Michigan | Y | 8.0% | 1.5% | 0.3% |
| 3 | Louisville | Y | 7.2% | 0.9% | 0.1% |
| 2 | Iowa | X | 6.7% | 1.4% | 0.1% |

### Women's Cinderella Watch *(seed ≥ 9, P(Sweet 16) ≥ 20%)*

None met this threshold — the Women's bracket is heavily top-loaded, with the four 1-seeds averaging a 79.7% probability of reaching the Final Four.

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

## Limitations / Potential Future Improvements

- **No injury data**: Player-level box scores are not provided, requiring injury data to be outsourced. Despite this, health is essential in playoff season and certainly contributes to the noise of March Madness. I hope I might be able to find a workaround for this in following years. 
- **Women's data gaps**: No Massey Ordinals or coaching records are available for the women's bracket.
- **Upset awareness**: The model treats each game independently and doesn't encode known base rates of tournament variance — in the vast majority of Men's tournaments, at least one 9+ seed reaches the Sweet 16. A future version could study profiles of historical Cinderella teams and use those patterns to redistribute probability mass toward upsets in a statistically grounded way.