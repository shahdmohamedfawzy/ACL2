# -*- coding: utf-8 -*-
import pandas as pd
from neo4j import GraphDatabase
import os, sys

CONFIG_FILE = "config.txt"
BATCH_SIZE = 1000  # number of rows per UNWIND batch


# ---------------------------------------------------------
# LOAD CONFIG
# ---------------------------------------------------------
def load_config():
    if not os.path.exists(CONFIG_FILE):
        print("[ERROR] config.txt not found!")
        sys.exit(1)

    config = {}
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if "=" not in line:
                continue
            key, value = line.strip().split("=", 1)
            config[key] = value.strip()

    return config["URI"], config["USERNAME"], config["PASSWORD"]


# ---------------------------------------------------------
# CONNECT TO AURA
# ---------------------------------------------------------
def get_driver():
    uri, username, password = load_config()
    return GraphDatabase.driver(uri, auth=(username, password))


# ---------------------------------------------------------
# CLEAR DB
# ---------------------------------------------------------
def clear_database(session):
    session.run("MATCH (n) DETACH DELETE n")
    print("[INFO] Database cleared.")


# ---------------------------------------------------------
# CREATE CONSTRAINTS
# ---------------------------------------------------------
def create_constraints(session):
    constraints = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Traveller) REQUIRE t.user_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (h:Hotel) REQUIRE h.hotel_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (c:City) REQUIRE c.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Country) REQUIRE c.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (r:Review) REQUIRE r.review_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (cg:Continent) REQUIRE cg.name IS UNIQUE"
    ]
    for c in constraints:
        session.run(c)
        print("[Constraint Created] " + c)


# ---------------------------------------------------------
# BATCH UNWIND LOADER
# ---------------------------------------------------------
def batch_import(session, query, data, batch_size=BATCH_SIZE):
    for i in range(0, len(data), batch_size):
        chunk = data[i:i+batch_size]
        session.run(query, batch=chunk)
    print(f"[BATCH DONE] {len(data)} rows")


# ---------------------------------------------------------
# Traveller: user_id (unique identifier), age, type, gender
# ---------------------------------------------------------
def load_traveller_nodes(session, df):
    data = df.to_dict("records")

    q = """
    UNWIND $batch AS row
    MERGE (t:Traveller {user_id: row.user_id})
    SET t.age = row.age_group,
        t.gender = row.user_gender,
        t.type = row.traveller_type
    """
    batch_import(session, q, data)
    print("[INFO] Traveller nodes created.")

# ---------------------------------------------------------
# Hotel: hotel_id (unique identifier), name, star_rating, cleanliness_base, 
#        comfort_base, facilities_base, average_reviews_score, amenity
# ---------------------------------------------------------
def load_hotel_nodes(session, df):
    data = df.to_dict("records")

    q = """
    UNWIND $batch AS row
    MERGE (h:Hotel {hotel_id: row.hotel_id})
    SET h.name = row.hotel_name,
        h.star_rating = toFloat(row.star_rating),
        h.cleanliness_base = toFloat(row.cleanliness_base),
        h.comfort_base = toFloat(row.comfort_base),
        h.facilities_base = toFloat(row.facilities_base),
        h.value_for_money_base = toFloat(row.value_for_money_base), 
        h.amenity = row.amenity
    """
    batch_import(session, q, data)
    print("[INFO] Hotel nodes created.")

# ---------------------------------------------------------
# City: name (unique identifier)
# ---------------------------------------------------------
def load_city_nodes(session, df):
    data = df.to_dict("records")

    q = """
    UNWIND $batch AS row
    MERGE (city:City {name: row.city})
    """
    batch_import(session, q, data)
    print("[INFO] City nodes created.")

# ---------------------------------------------------------
# Country: name (unique identifier)
# ---------------------------------------------------------
def load_country_nodes(session, df_list):
    combined = pd.concat(df_list, ignore_index=True)

    # FIX: remove NaN or empty country values
    combined = combined[combined["country"].notna()]
    combined = combined[combined["country"] != ""]

    unique_countries = pd.DataFrame({"country": combined["country"].unique()})

    data = unique_countries.to_dict("records")

    q = """
    UNWIND $batch AS row
    MERGE (c:Country {name: row.country})
    """

    batch_import(session, q, data)
    print("[INFO] Country nodes created.")

# ---------------------------------------------------------
# Continent: name (unique identifier) - CONTINENTS!
# ---------------------------------------------------------
def load_continent_nodes(session, df):
    # Filter out rows where country_group is NaN or empty
    df_filtered = df[df["country_group"].notna()]
    df_filtered = df_filtered[df_filtered["country_group"] != ""]
    
    unique_continents = pd.DataFrame({"continent": df_filtered["country_group"].unique()})
    data = unique_continents.to_dict("records")

    q = """
    UNWIND $batch AS row
    MERGE (cont:Continent {name: row.continent})
    """
    
    batch_import(session, q, data)
    print("[INFO] Continent nodes created.")

# ---------------------------------------------------------
# Review: review_id (unique identifier), text, date, score_overall, 
#         score_cleanliness, score_comfort, score_facilities, 
#         score_location, score_staff, score_value_for_money
# ---------------------------------------------------------
def load_review_nodes(session, df):
    data = df.to_dict("records")

    q = """
    UNWIND $batch AS row
    MERGE (r:Review {review_id: row.review_id})
    SET r.text = row.review_text,
        r.date = row.review_date,
        r.score_overall = toFloat(row.score_overall),
        r.score_cleanliness = toFloat(row.score_cleanliness),
        r.score_comfort = toFloat(row.score_comfort),
        r.score_facilities = toFloat(row.score_facilities),
        r.score_location = toFloat(row.score_location),
        r.score_staff = toFloat(row.score_staff),
        r.score_value_for_money = toFloat(row.score_value_for_money)
    """
    batch_import(session, q, data)
    print("[INFO] Review nodes created.")

# ---------------------------------------------------------
# RELATIONSHIPS
# ---------------------------------------------------------

def load_rel_wrote(session, df):
    data = df.to_dict("records")

    q = """
    UNWIND $batch AS row
    MATCH (t:Traveller {user_id: row.user_id})
    MATCH (r:Review {review_id: row.review_id})
    MERGE (t)-[:WROTE]->(r)
    """
    batch_import(session, q, data)

def load_rel_from_country(session, df):
    data = df.to_dict("records")

    q = """
    UNWIND $batch AS row
    MATCH (t:Traveller {user_id: row.user_id})
    MATCH (c:Country {name: row.country})
    MERGE (t)-[:FROM_COUNTRY]->(c)
    """
    batch_import(session, q, data)

def load_rel_stayed_at(session, df):
    data = df.to_dict("records")

    q = """
    UNWIND $batch AS row
    MATCH (t:Traveller {user_id: row.user_id})
    MATCH (h:Hotel {hotel_id: row.hotel_id})
    MERGE (t)-[:STAYED_AT]->(h)
    """
    batch_import(session, q, data)

def load_rel_reviewed(session, df):
    data = df.to_dict("records")

    q = """
    UNWIND $batch AS row
    MATCH (r:Review {review_id: row.review_id})
    MATCH (h:Hotel {hotel_id: row.hotel_id})
    MERGE (r)-[:REVIEWED]->(h)
    """
    batch_import(session, q, data)

def load_rel_hotel_located_in(session, df):
    data = df.to_dict("records")

    q = """
    UNWIND $batch AS row
    MATCH (h:Hotel {hotel_id: row.hotel_id})
    MATCH (city:City {name: row.city})
    MERGE (h)-[:LOCATED_IN]->(city)
    """
    batch_import(session, q, data)

def load_rel_city_country(session, df):
    data = df.to_dict("records")

    q = """
    UNWIND $batch AS row
    MATCH (city:City {name: row.city})
    MATCH (c:Country {name: row.country})
    MERGE (city)-[:LOCATED_IN]->(c)
    """
    batch_import(session, q, data)

# ---------------------------------------------------------
# (Country) -[:PART_OF]-> (Continent)
# ---------------------------------------------------------
def load_rel_country_continent(session, df):
    # Filter out rows where country_group is NaN or empty
    df_filtered = df[df["country_group"].notna()]
    df_filtered = df_filtered[df_filtered["country_group"] != ""]
    
    # Get unique country-continent pairs
    country_continent = df_filtered[["country", "country_group"]].drop_duplicates()
    data = country_continent.to_dict("records")

    q = """
    UNWIND $batch AS row
    MATCH (c:Country {name: row.country})
    MATCH (cont:Continent {name: row.country_group})
    MERGE (c)-[:PART_OF]->(cont)
    """
    batch_import(session, q, data)
    print("[INFO] Country-Continent relationships created.")

def load_rel_visa(session, df):
    data = df.to_dict("records")

    q = """
    UNWIND $batch AS row
    MATCH (c1:Country {name: row.from})
    MATCH (c2:Country {name: row.to})
    WITH c1, c2, row
    WHERE row.requires_visa = 'Yes'
    MERGE (c1)-[v:NEEDS_VISA]->(c2)
    SET v.visa_type = row.visa_type
    """
    batch_import(session, q, data)
    print("[INFO] Visa relationships loaded (only for requires_visa = Yes).")

# ---------------------------------------------------------
# Compute average_reviews_score for hotels
# ---------------------------------------------------------
def compute_hotel_average_scores(session):
    session.run("""
        MATCH (h:Hotel)<-[:REVIEWED]-(r:Review)
        WITH h, avg(r.score_overall) AS score
        SET h.average_reviews_score = score
    """)
    print("[INFO] Hotel average review scores computed.")

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    print("Loading CSVs...")
    users = pd.read_csv("users.csv")
    hotels = pd.read_csv("hotels.csv")
    reviews = pd.read_csv("reviews.csv")

# ===== FILTER REVIEWS 50001 → 50505 ONLY =====
    reviews = reviews[(reviews["review_id"] >= 50001) & (reviews["review_id"] <= 50505)]

    print(f"[INFO] Loaded {len(reviews)} filtered reviews (50001–50505).")

    visa = pd.read_csv("visa.csv")

    driver = get_driver()
    print("[INFO] Connected to AuraDB.")

    with driver.session() as session:
        # ------------------------------
        # RESET + CONSTRAINTS
        # ------------------------------
        clear_database(session)
        create_constraints(session)

        # ==============================
        # STEP 1 — CREATE ALL NODES
        # ==============================
        load_traveller_nodes(session, users)
        load_hotel_nodes(session, hotels)
        load_city_nodes(session, hotels)
        load_country_nodes(session, [users, hotels, visa])
        load_continent_nodes(session, hotels)  # CONTINENTS!
        load_review_nodes(session, reviews)

        print("[INFO] All nodes created.")

        # ==============================
        # STEP 2 — CREATE RELATIONSHIPS
        # ==============================
        load_rel_wrote(session, reviews)
        load_rel_from_country(session, users)
        load_rel_stayed_at(session, reviews)
        load_rel_reviewed(session, reviews)
        load_rel_hotel_located_in(session, hotels)
        load_rel_city_country(session, hotels)
        load_rel_country_continent(session, hotels)  # Country -> Continent
        load_rel_visa(session, visa)

        print("[INFO] All relationships created.")

        # ==============================
        # STEP 3 — COMPUTE HOTEL AVERAGES
        # ==============================
        compute_hotel_average_scores(session)

    driver.close()
    print("\n===============================")
    print("   KNOWLEDGE GRAPH COMPLETE!   ")
    print("===============================")


if __name__ == "__main__":
    main()