# jira_pipeline_models_v3.py

import os
os.environ["RICH_DISABLE"] = "1"
os.environ["PYTORCH_LIGHTNING_DISABLE_RICH"] = "1"

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from ngboost import NGBoost
from ngboost.distns import Normal
from ngboost.scores import LogScore

# =========================
# Config
# =========================
ISSUES_PATH = "issues.csv"
COMMENTS_PATH = "comments.csv"
CHANGELOG_PATH = "changelog.csv"
LINKS_PATH = "issuelinks.csv"

CHUNKSIZE = 250_000

TARGET_RAW = "resolution_time_hours"
TARGET_LOG = "target_log"  # log1p(TARGET_RAW)
TRIM_UPPER_Q = 0.99

CAT_COLS = ["priority.name", "status.name", "issuetype.name", "project.key", "assignee", "reporter"]

# Save options
SAVE_PARQUET = False  # اجعليها True بعد تثبيت pyarrow
FEATURE_TABLE_OUT_PARQUET = "final_feature_table.parquet"
FEATURE_TABLE_OUT_CSV = "final_feature_table.csv"

REPORT_OUT = "decision_support_report_top30.csv"

# =========================
# Helpers
# =========================
def safe_to_datetime(s):
    return pd.to_datetime(s, errors="coerce")

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def eval_reg(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
    }

def to_log1p(x):
    return np.log1p(x)

def from_log1p(x):
    return np.expm1(x)

def encode_categoricals_for_tree(X_train: pd.DataFrame, X_test: pd.DataFrame, cat_cols):
    """
    Convert pandas categorical/string columns into integer codes consistently.
    Works for XGBoost/NGBoost which need numeric arrays.
    """
    Xtr = X_train.copy()
    Xte = X_test.copy()
    for c in cat_cols:
        if c in Xtr.columns:
            # make sure it's categorical
            Xtr[c] = Xtr[c].astype("category")
            # align categories by union
            cats = pd.Index(Xtr[c].cat.categories).union(pd.Index(Xte[c].astype("category").cat.categories))
            Xtr[c] = Xtr[c].cat.set_categories(cats)
            Xte[c] = Xte[c].astype("category").cat.set_categories(cats)
            # codes: unknown -> -1
            Xtr[c] = Xtr[c].cat.codes.astype(np.int32)
            Xte[c] = Xte[c].cat.codes.astype(np.int32)
    return Xtr, Xte

# =========================
# 1) Load issues & define target
# =========================
print("Loading issues...")

issues_usecols = [
    "id", "key", "created", "updated", "resolutiondate",
    "assignee", "reporter",
    "priority.name", "status.name", "issuetype.name", "project.key"
]

issues = pd.read_csv(
    ISSUES_PATH,
    usecols=lambda c: c in issues_usecols,
    low_memory=True
)

issues["created"] = safe_to_datetime(issues["created"])
issues["updated"] = safe_to_datetime(issues["updated"])
issues["resolutiondate"] = safe_to_datetime(issues["resolutiondate"])

issues[TARGET_RAW] = (issues["resolutiondate"] - issues["created"]).dt.total_seconds() / 3600.0

issues = issues.dropna(subset=["key", "created", "resolutiondate", TARGET_RAW]).copy()
issues = issues[issues[TARGET_RAW] >= 0].copy()

print("Issues rows (resolved):", len(issues))
print("Date range:", issues["created"].min(), "->", issues["created"].max())

# =========================
# 1.1) Trim + log target
# =========================
upper = issues[TARGET_RAW].quantile(TRIM_UPPER_Q)
issues = issues[issues[TARGET_RAW] <= upper].copy()
issues[TARGET_LOG] = to_log1p(issues[TARGET_RAW])

print(f"After trimming at q={TRIM_UPPER_Q}: {len(issues)} rows")
print("Target (hours) summary after trim:")
print(issues[TARGET_RAW].describe())

issue_created = issues.set_index("key")["created"]

# =========================
# 2) Links features
# =========================
print("\nLoading issuelinks...")

links = pd.read_csv(LINKS_PATH, low_memory=True)

if "key" in links.columns:
    link_key_col = "key"
else:
    candidates = [c for c in links.columns if "key" in c.lower()]
    link_key_col = candidates[0] if candidates else links.columns[0]

link_counts = links.groupby(link_key_col).size().rename("link_count").reset_index()
link_counts = link_counts.rename(columns={link_key_col: "key"})

# =========================
# 3) Comments features (chunked)
# =========================
print("\nExtracting comments features (chunked)...")

comment_count = {}
first_comment = {}
last_comment = {}
unique_authors = {}

comments_usecols = ["key", "comment.author", "comment.created"]

for i, chunk in enumerate(pd.read_csv(
    COMMENTS_PATH,
    usecols=comments_usecols,
    chunksize=CHUNKSIZE,
    low_memory=True
)):
    chunk["comment.created"] = safe_to_datetime(chunk["comment.created"])
    chunk = chunk.dropna(subset=["key", "comment.created"])

    vc = chunk.groupby("key").size()
    for k, v in vc.items():
        comment_count[k] = comment_count.get(k, 0) + int(v)

    gmin = chunk.groupby("key")["comment.created"].min()
    gmax = chunk.groupby("key")["comment.created"].max()

    for k, t in gmin.items():
        prev = first_comment.get(k)
        first_comment[k] = t if prev is None or t < prev else prev

    for k, t in gmax.items():
        prev = last_comment.get(k)
        last_comment[k] = t if prev is None or t > prev else prev

    # unique authors (exact)
    ga = chunk.groupby("key")["comment.author"].apply(lambda s: set(s.dropna().astype(str)))
    for k, s in ga.items():
        if k not in unique_authors:
            unique_authors[k] = s
        else:
            unique_authors[k] |= s

    if (i + 1) % 5 == 0:
        print(f"  processed chunks: {i+1}")

comments_feat = pd.DataFrame({"key": list(comment_count.keys())})
comments_feat["comment_count"] = comments_feat["key"].map(comment_count).fillna(0).astype(int)
comments_feat["first_comment_ts"] = comments_feat["key"].map(first_comment)
comments_feat["last_comment_ts"] = comments_feat["key"].map(last_comment)
comments_feat["unique_comment_authors"] = comments_feat["key"].map(lambda k: len(unique_authors.get(k, set()))).fillna(0).astype(int)

comments_feat = comments_feat.merge(
    issue_created.rename("issue_created"),
    left_on="key", right_index=True,
    how="left"
)
comments_feat["first_comment_delay_hours"] = (
    (comments_feat["first_comment_ts"] - comments_feat["issue_created"]).dt.total_seconds() / 3600.0
)
comments_feat["comment_span_hours"] = (
    (comments_feat["last_comment_ts"] - comments_feat["first_comment_ts"]).dt.total_seconds() / 3600.0
)

comments_feat = comments_feat.drop(columns=["issue_created", "first_comment_ts", "last_comment_ts"])
print("Comments features rows:", len(comments_feat))

# =========================
# 4) Changelog features (chunked)
# =========================
print("\nExtracting changelog features (chunked)...")

status_trans = {}
assignee_changes = {}
priority_changes = {}
changelog_first = {}
changelog_last = {}
reopen_proxy = {}

changelog_usecols = ["key", "created", "field", "toString"]

for i, chunk in enumerate(pd.read_csv(
    CHANGELOG_PATH,
    usecols=changelog_usecols,
    chunksize=CHUNKSIZE,
    low_memory=True
)):
    chunk["created"] = safe_to_datetime(chunk["created"])
    chunk = chunk.dropna(subset=["key", "created", "field"])

    gmin = chunk.groupby("key")["created"].min()
    gmax = chunk.groupby("key")["created"].max()

    for k, t in gmin.items():
        prev = changelog_first.get(k)
        changelog_first[k] = t if prev is None or t < prev else prev

    for k, t in gmax.items():
        prev = changelog_last.get(k)
        changelog_last[k] = t if prev is None or t > prev else prev

    field_lower = chunk["field"].astype(str).str.lower()

    sub = chunk[field_lower.eq("status")]
    if len(sub):
        vc = sub.groupby("key").size()
        for k, v in vc.items():
            status_trans[k] = status_trans.get(k, 0) + int(v)

        sub_r = sub[sub["toString"].astype(str).str.contains("reopen", case=False, na=False)]
        if len(sub_r):
            vc2 = sub_r.groupby("key").size()
            for k, v in vc2.items():
                reopen_proxy[k] = reopen_proxy.get(k, 0) + int(v)

    sub = chunk[field_lower.eq("assignee")]
    if len(sub):
        vc = sub.groupby("key").size()
        for k, v in vc.items():
            assignee_changes[k] = assignee_changes.get(k, 0) + int(v)

    sub = chunk[field_lower.eq("priority")]
    if len(sub):
        vc = sub.groupby("key").size()
        for k, v in vc.items():
            priority_changes[k] = priority_changes.get(k, 0) + int(v)

    if (i + 1) % 5 == 0:
        print(f"  processed chunks: {i+1}")

all_keys = set(changelog_first.keys()) | set(status_trans.keys()) | set(assignee_changes.keys()) | set(priority_changes.keys())

changelog_feat = pd.DataFrame({"key": list(all_keys)})
changelog_feat["status_transition_count"] = changelog_feat["key"].map(status_trans).fillna(0).astype(int)
changelog_feat["assignee_change_count"] = changelog_feat["key"].map(assignee_changes).fillna(0).astype(int)
changelog_feat["priority_change_count"] = changelog_feat["key"].map(priority_changes).fillna(0).astype(int)
changelog_feat["reopen_proxy_count"] = changelog_feat["key"].map(reopen_proxy).fillna(0).astype(int)
changelog_feat["changelog_first_ts"] = changelog_feat["key"].map(changelog_first)
changelog_feat["changelog_last_ts"] = changelog_feat["key"].map(changelog_last)
changelog_feat["changelog_span_hours"] = (
    (changelog_feat["changelog_last_ts"] - changelog_feat["changelog_first_ts"]).dt.total_seconds() / 3600.0
)
changelog_feat = changelog_feat.drop(columns=["changelog_first_ts", "changelog_last_ts"])
print("Changelog features rows:", len(changelog_feat))

# =========================
# 5) Merge into final ML table
# =========================
df = issues.merge(link_counts, on="key", how="left")
df = df.merge(comments_feat, on="key", how="left")
df = df.merge(changelog_feat, on="key", how="left")

num_cols = [
    "link_count",
    "comment_count", "unique_comment_authors", "first_comment_delay_hours", "comment_span_hours",
    "status_transition_count", "assignee_change_count", "priority_change_count", "reopen_proxy_count",
    "changelog_span_hours",
]
for c in num_cols:
    if c in df.columns:
        df[c] = df[c].fillna(0)

for c in CAT_COLS:
    if c in df.columns:
        df[c] = df[c].fillna("UNKNOWN").astype(str)

df = df.dropna(subset=["created", TARGET_RAW, TARGET_LOG]).copy()
df = df.sort_values("created").reset_index(drop=True)

print("\nFinal dataset shape:", df.shape)
# =========================
# Productivity Feature Engineering (NEW)
# =========================
print("\nAdding productivity features...")

# تأكد من الترتيب الزمني
df = df.sort_values("created").reset_index(drop=True)

# -------------------------
# Developer historical stats
# -------------------------
df["dev_project_issue_count"] = (
    df.groupby(["assignee", "project.key"]).cumcount()
)

df["dev_res_cum_sum"] = (
    df.groupby("assignee")[TARGET_RAW].cumsum()
    - df[TARGET_RAW]
)

df["dev_issue_count"] = df.groupby("assignee").cumcount()

df["dev_avg_resolution_time"] = (
    df["dev_res_cum_sum"] / df["dev_issue_count"].replace(0, np.nan)
).fillna(df[TARGET_RAW].mean())

# -------------------------
# Developer workload (approx)
# -------------------------
df["dev_open_issue_count"] = (
    df.groupby("assignee").cumcount()
    - df.groupby("assignee")["resolutiondate"].cumcount()
).clip(lower=0)

# -------------------------
# Project backlog
# -------------------------
df["project_created_count"] = df.groupby("project.key").cumcount()

df["project_resolved_count"] = (
    df.sort_values("resolutiondate")
      .groupby("project.key")
      .cumcount()
      .reindex(df.index, fill_value=0)
)

df["project_backlog_size"] = (
    df["project_created_count"] - df["project_resolved_count"]
).clip(lower=0)

# -------------------------
# Project velocity (14 days)
# -------------------------
df["resolutiondate_tmp"] = df["resolutiondate"]

project_velocity = []
for proj, g in df.groupby("project.key"):
    g = g.sort_values("resolutiondate_tmp")
    times = g["resolutiondate_tmp"]
    counts = []
    for t in times:
        window_start = t - pd.Timedelta(days=14)
        counts.append(((times >= window_start) & (times <= t)).sum())
    project_velocity.extend(counts)

df["project_velocity_rolling"] = project_velocity

df = df.drop(columns=["dev_res_cum_sum", "project_created_count", "project_resolved_count", "resolutiondate_tmp"])

print("Productivity features added.")

print("Target summary (hours) after merge:")
print(df[TARGET_RAW].describe())

# =========================
# Save feature table (CSV default)
# =========================
# CSV works بدون أي مكتبات إضافية، لكنه كبير وبطيء نسبيًا
df.to_csv(FEATURE_TABLE_OUT_CSV, index=False)
print(f"\nSaved: {FEATURE_TABLE_OUT_CSV}")

# Parquet optional (requires pyarrow or fastparquet)
if SAVE_PARQUET:
    df.to_parquet(FEATURE_TABLE_OUT_PARQUET, index=False)
    print(f"Saved: {FEATURE_TABLE_OUT_PARQUET}")

# =========================
# 6) Time-aware split
# =========================
cut = int(len(df) * 0.8)
train_df = df.iloc[:cut].copy()
test_df  = df.iloc[cut:].copy()

y_train_log = train_df[TARGET_LOG].values
y_test_log  = test_df[TARGET_LOG].values

# Raw targets (hours) for evaluation and for XGBoost objective
y_train_raw = from_log1p(y_train_log)
y_test_raw = from_log1p(y_test_log)

drop_cols = ["id", "key", "created", "updated", "resolutiondate", TARGET_RAW, TARGET_LOG]
feature_cols = [c for c in df.columns if c not in drop_cols]

X_train = train_df[feature_cols].copy()
X_test  = test_df[feature_cols].copy()

# =========================
# 7) Models
# =========================
print("\n=== Model Training & Evaluation (CatBoost + XGBoost + NGBoost) ===")

# ---- 7.1 CatBoost (train on log-target; eval on hours)
cat_features_idx = [X_train.columns.get_loc(c) for c in CAT_COLS if c in X_train.columns]

cb = CatBoostRegressor(
    depth=8,
    learning_rate=0.05,
    iterations=1600,
    loss_function="RMSE",
    random_seed=42,
    verbose=0
)
cb.fit(X_train, y_train_log, cat_features=cat_features_idx)
pred_cb = from_log1p(cb.predict(X_test))

print("CatBoost:", eval_reg(y_test_raw, pred_cb))

# ---- 7.2 XGBoost squaredlogerror (train on raw hours)
# Need numeric encoding for categoricals
X_train_xgb, X_test_xgb = encode_categoricals_for_tree(X_train, X_test, CAT_COLS)

xgb = XGBRegressor(
    n_estimators=1400,
    learning_rate=0.03,
    max_depth=8,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    objective="reg:squaredlogerror",
    random_state=42,
    tree_method="hist",
)
xgb.fit(X_train_xgb, y_train_raw)
pred_xgb = xgb.predict(X_test_xgb)

print("XGBoost(SquaredLogError):", eval_reg(y_test_raw, pred_xgb))

# ---- 7.3 NGBoost (train on log-target; gives distribution -> intervals)
# NGBoost requires numeric features -> reuse encoded X_train_xgb
ngb = NGBoost(
    Dist=Normal,
    Score=LogScore,
    n_estimators=800,
    learning_rate=0.03,
    random_state=42
)
ngb.fit(X_train_xgb.values, y_train_log)
pred_ngb_log = ngb.predict(X_test_xgb.values)
pred_ngb = from_log1p(pred_ngb_log)

print("NGBoost:", eval_reg(y_test_raw, pred_ngb))

# Prediction intervals (10%-90%) from NGBoost distribution on log scale
dist = ngb.pred_dist(X_test_xgb.values)
q10_log = dist.ppf(0.1)
q90_log = dist.ppf(0.9)
pred_q10 = from_log1p(q10_log)
pred_q90 = from_log1p(q90_log)

# =========================
# 8) Decision Support report (use NGBoost intervals + bottleneck rules)
# =========================
report_cols = ["key"]
for c in ["priority.name", "status.name", "issuetype.name", "project.key"]:
    if c in test_df.columns:
        report_cols.append(c)

report = test_df[report_cols].copy()

# Use NGBoost mean prediction for ranking + intervals for uncertainty
report["pred_resolution_hours"] = pred_ngb
report["pred_q10"] = pred_q10
report["pred_q90"] = pred_q90
report["uncertainty_width"] = report["pred_q90"] - report["pred_q10"]

for c in ["assignee_change_count", "status_transition_count", "comment_count",
          "first_comment_delay_hours", "link_count"]:
    if c in test_df.columns:
        report[c] = test_df[c].values

def recommend(row):
    recs = []
    if row.get("uncertainty_width", 0) >= 72:  # 3 days width
        recs.append("High uncertainty: add buffer + monitor frequently")
    if row.get("assignee_change_count", 0) >= 3:
        recs.append("Reduce handoffs: assign stable owner / clarify ownership")
    if row.get("status_transition_count", 0) >= 6:
        recs.append("Workflow instability: investigate rework + tighten DoD")
    if row.get("first_comment_delay_hours", 0) >= 24:
        recs.append("Slow response: enforce first-response/review SLA")
    if row.get("comment_count", 0) >= 15:
        recs.append("High discussion: schedule sync / clarify requirements")
    if row.get("link_count", 0) >= 5:
        recs.append("Dependency-heavy: review blockers + decompose task")
    if not recs:
        recs.append("Normal risk: proceed")
    return " | ".join(recs)

report["recommendations"] = report.apply(recommend, axis=1)

report_sorted = report.sort_values("pred_resolution_hours", ascending=False).head(30)
report_sorted.to_csv(REPORT_OUT, index=False)
print(f"\nSaved: {REPORT_OUT}")

print("\nPipeline completed successfully.")
