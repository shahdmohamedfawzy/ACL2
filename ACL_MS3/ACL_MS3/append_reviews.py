import csv

# File to append into
CSV_FILE = "reviews.csv"

# New review rows to add (starting from 50107)
new_rows = [
   [50501, 8, 8, "2024-05-28", 9.0, 8.9, 8.8, 8.9, 9.1, 9.2, 8.8, "Copacabana Lux Rio gym excellent. Equipment modern well maintained always."],
[50502, 8, 8, "2024-06-30", 9.1, 9.0, 8.9, 9.0, 9.2, 9.3, 8.9, "Rio gym Copacabana Lux spacious. Personal trainers available helpful staff."],
[50503, 8, 8, "2024-07-28", 9.0, 8.9, 8.8, 8.9, 9.1, 9.2, 8.8, "Copacabana Lux fitness center impressive. Free weights cardio machines excellent."],
    ]

# Append to CSV
with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
    writer.writerows(new_rows)

print(f"[DONE] Added {len(new_rows)} new rows to {CSV_FILE}")
