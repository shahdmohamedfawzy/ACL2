# 🏨 International Hotel Booking Analytics

Can you tell where a hotel is located just from how guests rated it? This project puts that to the test — predicting a hotel's world region from review scores and traveler demographics, no location data allowed.

Built on 50K merged hotel/user/review records from Kaggle, with three models compared head-to-head and a live inference function at the end.

## What's inside

- End-to-end cleaning pipeline for a 3-table merge (hotels + users + reviews)
- Two quick data-viz questions (best city per traveler type, best value-for-money by age group)
- Five modeling experiments — Logistic Regression, Random Forest, and a small neural net — run under different feature sets to see what's actually predictive vs. what's just leaking the answer
- SHAP + LIME explainability on the winning model
- A ready-to-call function that takes a new review and returns a predicted region with a confidence score

## The catch: data leakage

The first pass at this hit **98% test accuracy** with a neural net — which sounds great until you notice it's cheating. The hotel-level "baseline" score columns (`cleanliness_base`, `staff_base`, etc.) are basically a fingerprint for each individual hotel, so the model was learning to recognize *hotels*, not regions.

Strip those out and accuracy drops to a much more honest **~37–40%** across the remaining models — still well above the ~9% you'd get guessing randomly across 11 regions, but a realistic signal rather than an illusion.

| Model | Features | Test Accuracy | Verdict |
|---|---|---|---|
| FFNN | scores + demographics + hotel baselines | 98.0% | 🚩 leaks hotel identity |
| Logistic Regression | scores + demographics | 34.5% | weak, poor per-class balance |
| FFNN | scores + demographics | 29.8% | collapses to one dominant class |
| FFNN | scores + demographics + deltas | 40.7% | better, still uneven across classes |
| **Random Forest** | scores + demographics | **37.0%** | ✅ most balanced, chosen model |

The Random Forest is the model we'd actually ship — it doesn't touch the leaking hotel-baseline features and holds up reasonably evenly across all 11 regions. `score_value_for_money`, `score_facilities`, and `score_location` come out as the top predictors, confirmed independently by both feature importances and SHAP.

## Quick answers along the way

- **Best city by traveler type:** Dubai wins for Business and Family travelers; Amsterdam wins for Solo and Couples.
- **Best value-for-money by age group:** China and the Netherlands show up near the top for almost every age bracket.

## Try it yourself

```python
infer_country_group({
    "score_overall": 8.7,
    "score_cleanliness": 9.0,
    "score_comfort": 9.2,
    "score_facilities": 8.8,
    "score_location": 8.6,
    "score_staff": 8.5,
    "score_value_for_money": 8.2,
    "user_gender": "Female",
    "age": 39,
    "traveller_type": "Family",
})
# → Predicted region: North America (Mexico), confidence 58.6%
```

The function handles all the preprocessing internally — encoding, one-hot columns, delta calculation — so you can feed it raw review inputs directly.

## Project layout

```
.
├── team03-ms1-acl.ipynb          # full pipeline: cleaning → EDA → modeling → explainability → inference
├── team03-ms1-acl_report.pdf     # write-up with detailed methodology and charts
└── README.md
```

Everything currently lives in one notebook. Data comes from the Kaggle dataset `international-hotel-booking-analytics` (`hotels.csv`, `users.csv`, `reviews.csv`), originally read from `/kaggle/input/...`.

## Running it locally

```bash
git clone https://github.com/<org>/<repo>.git
cd <repo>

python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# grab hotels.csv, users.csv, reviews.csv from Kaggle, drop them in a data/ folder,
# then swap the /kaggle/input/... paths in the notebook for your local data/ path

jupyter notebook team03-ms1-acl.ipynb
```

Built with Python 3.11. Main dependencies: `pandas`, `numpy`, `scikit-learn`, `tensorflow`, `matplotlib`, `seaborn`, `missingno`, `shap`, `lime`.

## Where this could go next

- Pull signal out of `review_text` instead of relying on structured scores alone
- Try XGBoost/LightGBM as a stronger drop-in for the Random Forest
- Group regions hierarchically (continent → sub-region) so classes are less imbalanced
- Rebalance training with SMOTE or class-weighted loss for the weaker classes
