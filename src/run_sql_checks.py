
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "instamart.db"
SQL_PATH = ROOT / "sql" / "part_a_checks.sql"

def main():
    sql_text = SQL_PATH.read_text(encoding="utf-8")

    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row
        statements = [
            statement.strip()
            for statement in sql_text.split(";")
            if statement.strip()
        ]

        for number, statement in enumerate(statements, start=1):
            print(f"\n--- SQL CHECK {number} ---")
            cursor = connection.execute(statement)
            rows = cursor.fetchall()

            if not rows:
                print("(No rows returned)")
                continue

            for row in rows:
                print(" | ".join(str(value) for value in row))

    print("\nSQL checks completed successfully.")

if __name__ == "__main__":
    main()
