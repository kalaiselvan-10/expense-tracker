"""Generate a 12-month labeled transaction dataset (synthetic) for modeling.

Each row keeps its true category in `true_category`, which is the answer key
used to measure categorizer accuracy. Some descriptions carry bank-style noise
(e.g. 'UPI/482913/Swiggy Order') so merchant normalization has real work to do.
"""

import random
from pathlib import Path

import pandas as pd

random.seed(42)

MERCHANTS = {
    "Food": ["Swiggy Order", "Zomato Order", "Dominos Pizza", "Cafe Coffee Day",
             "Biryani House", "Local Restaurant", "Burger King", "Chai Point"],
    "Transport": ["Uber Trip", "Ola Ride", "Petrol Pump", "Metro Card Recharge",
                  "Auto Rickshaw", "Bus Pass"],
    "Groceries": ["Big Bazaar Groceries", "Dmart Supermarket", "Reliance Fresh",
                  "Vegetable Market", "Nilgiris Store"],
    "Shopping": ["Amazon Purchase", "Flipkart Order", "Myntra Clothing",
                 "Lifestyle Store", "Croma Electronics"],
    "Health": ["Pharmacy", "Apollo Clinic", "Medplus", "City Hospital", "Dental Care"],
    "Entertainment": ["Movie Tickets", "Pvr Cinemas", "Game Store", "Concert Pass"],
}
RANGES = {  # category: (min amount, max amount, transactions per month)
    "Food": (150, 700, 10), "Transport": (80, 1500, 8), "Groceries": (400, 3000, 4),
    "Shopping": (500, 3500, 3), "Health": (200, 1200, 2), "Entertainment": (200, 900, 2),
}
ACCOUNTS = ["HDFC Savings", "HDFC Card"]
rows = []


def noisy(description: str) -> str:
    roll = random.random()
    if roll < 0.20:
        return f"UPI/{random.randint(100000, 999999)}/{description}"
    if roll < 0.30:
        return f"{description.upper()} #{random.randint(1000, 9999)}"
    return description


for month in pd.period_range("2025-10", "2026-09", freq="M"):
    def add(day, description, amount, category):
        rows.append({
            "date": f"{month.year}-{month.month:02d}-{day:02d}",
            "description": noisy(description),
            "amount": amount,
            "account": random.choice(ACCOUNTS),
            "true_category": category,
        })

    add(1, "Salary Credit", 50000, "Income")
    add(2, "House Rent", -12000, "Rent")
    add(5, "Netflix Subscription", -649, "Subscriptions")
    add(7, "Electricity Bill", -random.randint(1500, 2200), "Bills")
    add(10, "Mobile Recharge", -299, "Bills")
    for category, (low, high, count) in RANGES.items():
        for _ in range(count):
            add(random.randint(1, 28), random.choice(MERCHANTS[category]),
                -random.randint(low, high), category)

df = pd.DataFrame(rows).sort_values("date", kind="stable").reset_index(drop=True)
out = Path(__file__).resolve().parents[1] / "data" / "labeled_transactions.csv"
df.to_csv(out, index=False)
print(f"Saved {len(df)} rows to {out}")
print(df["true_category"].value_counts().to_string())
