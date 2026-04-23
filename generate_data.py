"""
Synthetic Sales Data Generator
Generates 3 years of realistic sales data with seasonal patterns,
regional variation, and diverse product mix.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_sales_data(output_path: str = None, num_rows: int = 12000) -> pd.DataFrame:
    """Generate realistic synthetic sales data."""
    np.random.seed(42)

    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "sales_data.csv")

    # Date range: 3 years
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2025, 12, 31)
    total_days = (end_date - start_date).days

    # Products and categories
    products = {
        "Electronics": [
            ("Laptop Pro X1", 899, 620),
            ("Wireless Earbuds", 79, 32),
            ("Smart Watch Ultra", 349, 195),
            ("4K Monitor", 449, 290),
            ("Mechanical Keyboard", 129, 58),
            ("Gaming Mouse", 69, 28),
            ("USB-C Hub", 49, 18),
            ("Portable SSD 1TB", 109, 55),
        ],
        "Clothing": [
            ("Premium Jacket", 189, 72),
            ("Running Shoes", 129, 48),
            ("Casual T-Shirt", 29, 9),
            ("Denim Jeans", 69, 24),
            ("Winter Hoodie", 59, 21),
            ("Sports Shorts", 35, 12),
        ],
        "Home & Office": [
            ("Standing Desk", 549, 320),
            ("Ergonomic Chair", 399, 210),
            ("LED Desk Lamp", 45, 15),
            ("Bookshelf Organizer", 89, 38),
            ("Coffee Maker Pro", 149, 65),
            ("Air Purifier", 199, 95),
        ],
        "Accessories": [
            ("Leather Wallet", 49, 16),
            ("Backpack Elite", 89, 35),
            ("Sunglasses", 59, 18),
            ("Phone Case Premium", 25, 7),
            ("Travel Mug", 19, 6),
        ],
        "Software & Services": [
            ("Cloud Storage Plan", 99, 15),
            ("VPN Annual Sub", 79, 10),
            ("Design Suite License", 299, 45),
            ("Project Mgmt Tool", 149, 22),
        ],
    }

    regions = ["North America", "Europe", "Asia Pacific", "Latin America", "Middle East"]
    region_weights = [0.35, 0.28, 0.22, 0.10, 0.05]

    customer_segments = ["Enterprise", "SMB", "Consumer", "Government"]
    segment_weights = [0.25, 0.30, 0.35, 0.10]

    records = []
    order_id = 10000

    for _ in range(num_rows):
        # Random date with slight bias toward recent dates
        day_offset = int(np.random.beta(2, 1.5) * total_days)
        date = start_date + timedelta(days=day_offset)

        # Seasonal multiplier
        month = date.month
        if month in [11, 12]:       # Holiday season boost
            seasonal = np.random.uniform(1.3, 1.8)
        elif month in [1, 2]:       # Post-holiday dip
            seasonal = np.random.uniform(0.7, 0.9)
        elif month in [6, 7, 8]:    # Summer
            seasonal = np.random.uniform(1.0, 1.3)
        else:
            seasonal = np.random.uniform(0.9, 1.1)

        # Pick category and product
        category = np.random.choice(list(products.keys()),
                                     p=[0.30, 0.20, 0.20, 0.15, 0.15])
        product_name, base_price, base_cost = products[category][
            np.random.randint(0, len(products[category]))
        ]

        # Region selection
        region = np.random.choice(regions, p=region_weights)

        # Regional price adjustment
        region_mult = {
            "North America": 1.0,
            "Europe": 1.08,
            "Asia Pacific": 0.88,
            "Latin America": 0.82,
            "Middle East": 1.05,
        }[region]

        # Quantity (higher for cheaper items)
        if base_price < 50:
            quantity = int(np.random.exponential(4) + 1)
        elif base_price < 200:
            quantity = int(np.random.exponential(2) + 1)
        else:
            quantity = int(np.random.exponential(1) + 1)

        quantity = max(1, min(quantity, 50))

        # Calculate financials
        unit_price = round(base_price * region_mult * seasonal * np.random.uniform(0.92, 1.08), 2)
        unit_cost = round(base_cost * np.random.uniform(0.95, 1.05), 2)
        revenue = round(unit_price * quantity, 2)
        cost = round(unit_cost * quantity, 2)
        profit = round(revenue - cost, 2)

        # Customer
        customer_id = f"CUST-{np.random.randint(1000, 5000):04d}"
        customer_segment = np.random.choice(customer_segments, p=segment_weights)

        # Discount (occasional)
        discount = 0.0
        if np.random.random() < 0.25:
            discount = np.random.choice([5, 10, 15, 20, 25])

        order_id += 1

        records.append({
            "Date": date.strftime("%Y-%m-%d"),
            "OrderID": f"ORD-{order_id}",
            "Product": product_name,
            "Category": category,
            "Region": region,
            "Quantity": quantity,
            "UnitPrice": unit_price,
            "Revenue": revenue,
            "Cost": cost,
            "Profit": profit,
            "Discount": discount,
            "CustomerID": customer_id,
            "CustomerSegment": customer_segment,
        })

    df = pd.DataFrame(records)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[OK] Generated {len(df)} sales records -> {output_path}")

    return df


if __name__ == "__main__":
    generate_sales_data()
