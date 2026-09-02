"""
RazorGuard - Day 1: Synthetic dataset generator with leakage-safe historical features.

DESIGN
------
1. Generate static "world" entities first: customers (with signup_date, region_code)
   and products (with category, product_rating, base price). None of this depends
   on any order outcome.

2. Generate order *skeletons*: order_id, customer_id, product_id, timestamp, and
   order-level attributes (order_value, discount_pct, is_cod, delivery_distance_km,
   browser_family, device_age_days, session_hour). Still no target, no history
   features yet.

3. Sort ALL orders globally by timestamp (ties broken by order_id) and process them
   ONE AT A TIME in that order. For each order:
      a. Read the customer's history dict (built only from orders already processed,
         i.e. strictly earlier in time) -> compute customer-side historical features.
      b. Read the category's running counters (same guarantee) -> compute
         category_return_rate.
      c. Combine historical + order-level features into a latent logit -> propensity
         -> Bernoulli draw -> `returned`.
      d. ONLY NOW append this order's (timestamp, returned) into the customer history
         and increment the category counters.

   Because step (d) always happens after steps (a)-(c) for the same order, and orders
   are processed in non-decreasing timestamp order, no feature can ever be computed
   from an order with timestamp >= the current order's timestamp. This is the
   leakage-safety invariant, and it holds by construction rather than by convention.

4. Assign train / validation / test splits by calendar month (Jan-May / Jun / Jul-Aug).

Run:  python src/data_generator.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path

# ----------------------------------------------------------------------------
# Config (fixed, documented, seedable)
# ----------------------------------------------------------------------------

SEED = 42

N_CUSTOMERS = 1500
N_PRODUCTS_PER_CATEGORY = 15
N_ORDERS = 10_000

CATEGORIES = [
    "fashion_apparel", "footwear", "electronics_accessories", "home_kitchen",
    "beauty_personal_care", "mobile_electronics", "books_stationery", "sports_fitness",
]

# Category base prices (INR) - rough real-world plausibility, not tied to return risk
CATEGORY_BASE_PRICE = {
    "fashion_apparel": 1200, "footwear": 2200, "electronics_accessories": 900,
    "home_kitchen": 1800, "beauty_personal_care": 700, "mobile_electronics": 15000,
    "books_stationery": 450, "sports_fitness": 1600,
}

START_DATE = pd.Timestamp("2026-01-01")
END_DATE = pd.Timestamp("2026-09-01")  # exclusive upper bound (so Aug 31 is the last valid day)

REGION_CODES = ["north", "south", "east", "west", "central"]
BROWSER_FAMILIES = ["chrome", "safari", "firefox", "edge", "in_app_webview"]

# Cold-start smoothing prior for category_return_rate (Beta prior)
CATEGORY_PRIOR_ALPHA = 3.0
CATEGORY_PRIOR_BETA = 15.0  # prior mean = 3/18 = 0.1667, close to target base rate

# Cold-start default for a customer's return_rate_90d with no prior orders
CUSTOMER_PRIOR_RETURN_RATE = 0.18

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


# ----------------------------------------------------------------------------
# Step 1: static world entities
# ----------------------------------------------------------------------------

def generate_customers(rng: np.random.Generator) -> pd.DataFrame:
    customer_id = np.arange(1, N_CUSTOMERS + 1)
    # signup_date: some customers pre-date the observation window (long-tenured),
    # some sign up during it (new customers). Always <= END_DATE.
    signup_offset_days = rng.integers(low=-500, high=(END_DATE - START_DATE).days, size=N_CUSTOMERS)
    signup_date = START_DATE + pd.to_timedelta(signup_offset_days, unit="D")
    region_code = rng.choice(REGION_CODES, size=N_CUSTOMERS)
    return pd.DataFrame({
        "customer_id": customer_id,
        "signup_date": signup_date,
        "region_code": region_code,
    })


def generate_products(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    pid = 1
    for cat in CATEGORIES:
        base_price = CATEGORY_BASE_PRICE[cat]
        for _ in range(N_PRODUCTS_PER_CATEGORY):
            rating = float(np.clip(rng.normal(4.0, 0.5), 1.0, 5.0))
            price_mult = float(np.clip(rng.lognormal(mean=0.0, sigma=0.35), 0.3, 3.0))
            rows.append({
                "product_id": pid,
                "category": cat,
                "product_rating": round(rating, 2),
                "base_price": round(base_price * price_mult, 2),
            })
            pid += 1
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Step 2: order skeletons (no target, no history features yet)
# ----------------------------------------------------------------------------

def generate_order_skeleton(customers: pd.DataFrame, products: pd.DataFrame,
                             rng: np.random.Generator) -> pd.DataFrame:
    order_id = np.arange(1, N_ORDERS + 1)

    customer_idx = rng.integers(0, N_CUSTOMERS, size=N_ORDERS)
    chosen_customers = customers.iloc[customer_idx].reset_index(drop=True)

    product_idx = rng.integers(0, len(products), size=N_ORDERS)
    chosen_products = products.iloc[product_idx].reset_index(drop=True)

    # Order timestamp: uniform between max(signup_date, START_DATE) and END_DATE,
    # at second-level granularity (avoids exact-timestamp ties across ~21M seconds).
    window_start = chosen_customers["signup_date"].clip(lower=START_DATE)
    window_seconds = (END_DATE - window_start).dt.total_seconds().astype(np.int64)
    offset_seconds = (rng.random(N_ORDERS) * window_seconds).astype(np.int64)
    timestamp = window_start + pd.to_timedelta(offset_seconds, unit="s")

    is_cod = rng.random(N_ORDERS) < 0.35
    discount_pct = np.clip(rng.beta(2, 6, size=N_ORDERS) * 0.5, 0, 0.45)
    quantity = rng.integers(1, 4, size=N_ORDERS)
    order_value = np.round(
        chosen_products["base_price"].to_numpy() * quantity * (1 - discount_pct) *
        np.clip(rng.lognormal(0, 0.15, size=N_ORDERS), 0.5, 2.0), 2
    )
    delivery_distance_km = np.round(np.clip(rng.gamma(4.0, 60.0, size=N_ORDERS), 2, 3000), 1)
    device_age_days = rng.integers(0, 1500, size=N_ORDERS)
    session_hour = rng.integers(0, 24, size=N_ORDERS)
    browser_family = rng.choice(BROWSER_FAMILIES, size=N_ORDERS)

    df = pd.DataFrame({
        "order_id": order_id,
        "customer_id": chosen_customers["customer_id"].to_numpy(),
        "region_code": chosen_customers["region_code"].to_numpy(),
        "product_id": chosen_products["product_id"].to_numpy(),
        "category": chosen_products["category"].to_numpy(),
        "product_rating": chosen_products["product_rating"].to_numpy(),
        "timestamp": timestamp,
        "order_value": order_value,
        "discount_pct": np.round(discount_pct, 4),
        "is_cod": is_cod,
        "quantity": quantity,
        "delivery_distance_km": delivery_distance_km,
        "device_age_days": device_age_days,
        "session_hour": session_hour,
        "browser_family": browser_family,
    })

    df = df.merge(customers[["customer_id", "signup_date"]], on="customer_id", how="left")
    df["customer_age_days"] = (df["timestamp"] - df["signup_date"]).dt.days
    assert (df["customer_age_days"] >= 0).all(), "customer_age_days must never be negative"

    # Sort globally by timestamp (tie-break by order_id) - this fixes the processing
    # order used in Step 3 and is what the leakage guarantee relies on.
    df = df.sort_values(["timestamp", "order_id"]).reset_index(drop=True)
    return df


# ----------------------------------------------------------------------------
# Step 3: sequential, leakage-safe feature + label construction
# ----------------------------------------------------------------------------

@dataclass
class CustomerHistory:
    order_timestamps: list
    returned_flags: list


def build_dataset(order_skeleton: pd.DataFrame, rng: np.random.Generator,
                   beta0: float = -2.85) -> pd.DataFrame:
    customer_hist: dict[int, CustomerHistory] = {}
    category_counts: dict[str, list[int]] = {cat: [0, 0] for cat in CATEGORIES}  # [orders, returns]

    n = len(order_skeleton)
    out_previous_orders = np.zeros(n, dtype=np.int64)
    out_previous_returns = np.zeros(n, dtype=np.int64)
    out_return_rate_90d = np.zeros(n, dtype=np.float64)
    out_orders_last_30d = np.zeros(n, dtype=np.int64)
    out_category_return_rate = np.zeros(n, dtype=np.float64)
    out_returned = np.zeros(n, dtype=np.int64)
    out_return_propensity = np.zeros(n, dtype=np.float64)  # latent p, kept for diagnostics only

    ts = order_skeleton["timestamp"].to_numpy()
    cust_ids = order_skeleton["customer_id"].to_numpy()
    categories = order_skeleton["category"].to_numpy()
    order_values = order_skeleton["order_value"].to_numpy()
    discount_pcts = order_skeleton["discount_pct"].to_numpy()
    is_cods = order_skeleton["is_cod"].to_numpy()
    delivery_km = order_skeleton["delivery_distance_km"].to_numpy()

    log_order_value = np.log1p(order_values)
    log_order_value_z = (log_order_value - log_order_value.mean()) / log_order_value.std()
    high_value_flag = (order_values > np.percentile(order_values, 75)).astype(float)

    noise = rng.normal(0, 0.65, size=n)  # irreducible randomness in the latent process

    for i in range(n):
        cid = cust_ids[i]
        cat = categories[i]
        current_ts = ts[i]

        # --- (a) customer historical features, from strictly-prior orders only ---
        hist = customer_hist.get(cid)
        if hist is None or len(hist.order_timestamps) == 0:
            previous_orders = 0
            previous_returns = 0
            return_rate_90d = CUSTOMER_PRIOR_RETURN_RATE
            orders_last_30d = 0
        else:
            hist_ts = np.array(hist.order_timestamps)
            hist_ret = np.array(hist.returned_flags)
            previous_orders = len(hist_ts)
            previous_returns = int(hist_ret.sum())

            window_90 = hist_ts >= (current_ts - np.timedelta64(90, "D"))
            n_90 = int(window_90.sum())
            return_rate_90d = float(hist_ret[window_90].mean()) if n_90 > 0 else CUSTOMER_PRIOR_RETURN_RATE

            window_30 = hist_ts >= (current_ts - np.timedelta64(30, "D"))
            orders_last_30d = int(window_30.sum())

        # --- (b) category running rate, from strictly-prior orders only ---
        cat_orders, cat_returns = category_counts[cat]
        category_return_rate = (cat_returns + CATEGORY_PRIOR_ALPHA) / (
            cat_orders + CATEGORY_PRIOR_ALPHA + CATEGORY_PRIOR_BETA
        )

        # --- (c) latent propensity -> label ---
        z = (
            beta0
            + 1.55 * return_rate_90d
            + 1.35 * category_return_rate
            + 0.55 * is_cods[i]
            + 0.42 * log_order_value_z[i]
            + 0.95 * discount_pcts[i]
            + 0.20 * (delivery_km[i] / 500.0)
            + 0.70 * (is_cods[i] * high_value_flag[i])          # interaction 1
            + 1.10 * (category_return_rate * discount_pcts[i])   # interaction 2
            + noise[i]
        )
        p = sigmoid(z)
        returned = int(rng.random() < p)

        out_previous_orders[i] = previous_orders
        out_previous_returns[i] = previous_returns
        out_return_rate_90d[i] = return_rate_90d
        out_orders_last_30d[i] = orders_last_30d
        out_category_return_rate[i] = category_return_rate
        out_return_propensity[i] = p
        out_returned[i] = returned

        # --- (d) append AFTER use: future orders may see this, current one never did ---
        if hist is None:
            hist = CustomerHistory(order_timestamps=[], returned_flags=[])
            customer_hist[cid] = hist
        hist.order_timestamps.append(current_ts)
        hist.returned_flags.append(returned)
        category_counts[cat][0] += 1
        category_counts[cat][1] += returned

    result = order_skeleton.copy()
    result["previous_orders"] = out_previous_orders
    result["previous_returns"] = out_previous_returns
    result["return_rate_90d"] = out_return_rate_90d
    result["orders_last_30d"] = out_orders_last_30d
    result["category_return_rate"] = out_category_return_rate
    result["_latent_propensity"] = out_return_propensity  # diagnostic only, NOT a model feature
    result["returned"] = out_returned
    return result


# ----------------------------------------------------------------------------
# Step 4: temporal split
# ----------------------------------------------------------------------------

def assign_splits(df: pd.DataFrame) -> pd.DataFrame:
    month = df["timestamp"].dt.month
    split = np.where(month <= 5, "train", np.where(month == 6, "validation", "test"))
    df = df.copy()
    df["split"] = split
    return df


# ----------------------------------------------------------------------------
# Step 5: basic data-quality validation
# ----------------------------------------------------------------------------

def validate_dataset(df: pd.DataFrame) -> None:
    assert len(df) == N_ORDERS, f"expected {N_ORDERS} rows, got {len(df)}"
    assert df["order_id"].is_unique, "order_id must be unique"
    assert df["returned"].isin([0, 1]).all(), "returned must be binary"
    assert df["customer_age_days"].ge(0).all(), "customer_age_days must be non-negative"
    assert df["timestamp"].between(START_DATE, END_DATE).all(), "timestamp outside configured window"

    base_rate = df["returned"].mean()
    assert 0.08 <= base_rate <= 0.30, f"return rate {base_rate:.3f} outside plausible sanity band"

    split_counts = df["split"].value_counts()
    assert set(split_counts.index) == {"train", "validation", "test"}
    assert (df.loc[df["split"] == "train", "timestamp"].dt.month <= 5).all()
    assert (df.loc[df["split"] == "validation", "timestamp"].dt.month == 6).all()
    assert (df.loc[df["split"] == "test", "timestamp"].dt.month >= 7).all()

    print("Validation passed.")
    print(f"  Rows: {len(df)}")
    print(f"  Observed return rate: {base_rate:.4f}")
    print(f"  Split sizes: {split_counts.to_dict()}")
    print(f"  Split return rates: {df.groupby('split')['returned'].mean().round(4).to_dict()}")


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------

def generate_full_dataset(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    customers = generate_customers(rng)
    products = generate_products(rng)
    skeleton = generate_order_skeleton(customers, products, rng)
    dataset = build_dataset(skeleton, rng)
    dataset = assign_splits(dataset)
    return dataset


def main():
    df = generate_full_dataset(SEED)
    validate_dataset(df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_facing_cols = [
        "order_id", "customer_id", "timestamp", "split",
        # strong
        "return_rate_90d", "category_return_rate", "is_cod",
        # medium
        "order_value", "discount_pct", "delivery_distance_km", "previous_returns",
        # weak
        "customer_age_days", "previous_orders", "product_rating", "orders_last_30d",
        # non-predictive
        "region_code", "browser_family", "device_age_days", "session_hour",
        # target
        "returned",
    ]
    df[model_facing_cols].to_csv(OUTPUT_DIR / "razorguard_dataset.csv", index=False)
    # Full raw log (includes category, product_id etc.) kept separately for the
    # independent leakage-test recomputation.
    df.to_csv(OUTPUT_DIR / "razorguard_raw_log.csv", index=False)
    print(f"Saved dataset to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
