
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"


def check(condition, message):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {message}")
    return bool(condition)


def main():
    print("=" * 55)
    print("PART A - CLEANED DATA VALIDATION")
    print("=" * 55)

    results = []

    required_files = [
        "customers.csv",
        "products.csv",
        "orders.csv",
        "tickets.csv",
    ]

    for name in required_files:
        results.append(
            check((CLEAN / name).exists(), f"{name} exists")
        )

    if not all((CLEAN / name).exists() for name in required_files):
        print("\nCreate missing files before continuing.")
        return

    customers = pd.read_csv(CLEAN / "customers.csv")
    products = pd.read_csv(CLEAN / "products.csv")
    orders = pd.read_csv(CLEAN / "orders.csv")
    tickets = pd.read_csv(CLEAN / "tickets.csv")

    raw_orders = pd.read_csv(RAW / "orders.csv")

    print("\n--- Record counts ---")
    print(f"Raw customers: {len(pd.read_csv(RAW / 'customers.csv'))}")
    print(f"Clean customers: {len(customers)}")
    print(f"Clean products: {len(products)}")
    print(f"Raw orders: {len(raw_orders)}")
    print(f"Clean orders: {len(orders)}")
    print(f"Clean tickets: {len(tickets)}")

    print("\n--- Customer checks ---")
    results.append(check(
        customers["customer_id"].is_unique,
        "Customer IDs are unique"
    ))
    results.append(check(
        customers["age"].dropna().between(18, 100).all(),
        "All non-missing ages are between 18 and 100"
    ))
    results.append(check(
        customers["city"].notna().all(),
        "Customer cities are present"
    ))

    print("\n--- Product checks ---")
    results.append(check(
        products["product_id"].is_unique,
        "Product IDs are unique"
    ))
    results.append(check(
        (products["price_inr"] > 0).all(),
        "Product prices are positive"
    ))

    print("\n--- Order checks ---")
    results.append(check(
        orders["order_id"].is_unique,
        "Order IDs are unique"
    ))
    results.append(check(
        orders["status"].isin(["completed", "cancelled"]).all(),
        "Only completed and cancelled orders remain"
    ))
    results.append(check(
        (orders["quantity"] > 0).all(),
        "Order quantities are positive"
    ))
    results.append(check(
        (orders["order_amount"] > 0).all(),
        "Order amounts are positive"
    ))
    results.append(check(
        orders["delivery_minutes"].dropna().ge(0).all(),
        "No negative delivery durations remain"
    ))
    results.append(check(
        orders.loc[
            orders["status"] == "cancelled",
            "delivery_minutes"
        ].isna().all(),
        "Cancelled orders have no delivery duration"
    ))

    print("\n--- Ticket checks ---")
    results.append(check(
        tickets["ticket_id"].is_unique,
        "Ticket IDs are unique"
    ))
    results.append(check(
        len(tickets) == len(pd.read_csv(RAW / "tickets.csv")),
        "Ticket count is preserved"
    ))

    print("\n--- Summary ---")
    passed = sum(results)
    failed = len(results) - passed
    print(f"Checks passed: {passed}")
    print(f"Checks failed: {failed}")

    if failed == 0:
        print("VALIDATION COMPLETED: ALL CHECKS PASSED")
    else:
        print("VALIDATION NEEDS ATTENTION")


if __name__ == "__main__":
    main()
