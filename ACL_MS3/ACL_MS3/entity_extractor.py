# entity_extractor.py

import spacy
import re
from typing import Dict, Any, Optional

# Load spaCy model
nlp = spacy.load("en_core_web_sm")


# ---------------------------------------------------------
# AGE GROUP MAPPING (from KG schema)
# ---------------------------------------------------------
AGE_GROUPS = [
    (18, 24, "18-24"),
    (25, 34, "25-34"),
    (35, 44, "35-44"),
    (45, 54, "45-54"),
    (55, 200, "55+"),
]


def map_age_to_group(age: int) -> Optional[str]:
    """Map exact age to age group from KG."""
    for low, high, group_name in AGE_GROUPS:
        if low <= age <= high:
            return group_name
    return None


# ---------------------------------------------------------
# CONTINENT LIST (for matching country_group)
# ---------------------------------------------------------
CONTINENTS = [
    "Africa",
    "Asia", 
    "Europe",
    "North America",
    "South America",
    "Oceania"
]


# ============================================================
# MAIN ENTITY EXTRACTION FUNCTION
# ============================================================
def extract_entities(user_query: str) -> Dict[str, Any]:
    """
    Extract relevant entities from user input using spaCy NER
    plus some pattern-based logic.
    """
    doc = nlp(user_query)
    text_lower = user_query.lower()

    entities: Dict[str, Any] = {}

    # ---------------------------------------------------------
    # BASIC NER EXTRACTION
    # ---------------------------------------------------------
    gpe_entities = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
    org_entities = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "FAC", "WORK_OF_ART"]]
    date_entities = [ent.text for ent in doc.ents if ent.label_ == "DATE"]

    # ---------------------------------------------------------
    # VISA QUERY DETECTION (from/to countries)
    # ---------------------------------------------------------
    is_visa_query = any(k in text_lower for k in ["visa", "entry", "permit", "visas"])

    if is_visa_query:
        # "from X ..."
        if "from" in text_lower and len(gpe_entities) >= 1:
            entities["from_country"] = gpe_entities[0]

        # "... to Y" or "... visit Y"
        if ("to" in text_lower or "visit" in text_lower) and len(gpe_entities) >= 2:
            entities["to_country"] = gpe_entities[1]

        # Visa queries don't care about hotel/city/metrics
        return entities

    # ---------------------------------------------------------
    # CONTINENT EXTRACTION (NEW - PRIORITY!)
    # Check if any continent is mentioned in the query
    # ---------------------------------------------------------
    for continent in CONTINENTS:
        if continent.lower() in text_lower:
            entities["continent"] = continent
            break
    
    # ---------------------------------------------------------
    # CITY EXTRACTION (for hotel-related queries)
    # Only extract city if no continent was found
    # ---------------------------------------------------------
    if "continent" not in entities and gpe_entities:
        entities["city"] = gpe_entities[0]

    # ---------------------------------------------------------
    # HOTEL NAME EXTRACTION
    # ---------------------------------------------------------
    if org_entities:
        entities["hotel_name"] = org_entities[0]

    # ---------------------------------------------------------
    # YEAR EXTRACTION
    # ---------------------------------------------------------
    for ent in date_entities:
        for token in ent.split():
            if token.isdigit() and len(token) == 4:
                entities["year"] = int(token)
                break
        if "year" in entities:
            break

    # ---------------------------------------------------------
    # STAR RATING EXTRACTION
    #   Examples:
    #   - "star rating above 3"
    #   - "5 star hotel"
    # ---------------------------------------------------------
    star_match = re.search(r"(\d)\s*star", text_lower)
    if star_match:
        entities["star_rating"] = int(star_match.group(1))

    above_match = re.search(r"above\s+(\d)", text_lower)
    if above_match:
        entities["star_rating"] = int(above_match.group(1))

    # ---------------------------------------------------------
    # LIMIT (e.g. "top 3 hotels")
    # ---------------------------------------------------------
    top_match = re.search(r"top\s+(\d+)", text_lower)
    if top_match:
        entities["limit"] = int(top_match.group(1))

    # ---------------------------------------------------------
    # SPECIAL PHRASES → AGE GROUP / TRAVELLER TYPE / AMENITY
    # ---------------------------------------------------------
    # "senior trip" should map to age_group "18-24" (your schema)
    if "senior trip" in text_lower or ("senior" in text_lower and "trip" in text_lower):
        entities["age_group"] = "18-24"

    # "honeymoon" should map to traveller_type "Couple"
    if "honeymoon" in text_lower:
        entities["traveller_type"] = "Couple"

    # Pet-friendly hotels
    if "pet" in text_lower or "pet-friendly" in text_lower or "pet friendly" in text_lower:
        entities["amenity"] = "pet friendly"
    
    # Gym/exercise/fitness amenities
    gym_keywords = [
        "gym", "exercise", "fitness", "workout", "active",
        "exercise space", "fitness center", "fitness centre",
        "stay active", "dedicated exercise"
    ]
    if any(keyword in text_lower for keyword in gym_keywords):
        entities["amenity"] = "gym"

    # ---------------------------------------------------------
    # METRIC EXTRACTION (cleanliness, comfort, value for money, etc.)
    # ---------------------------------------------------------
    metric_map = {
        "cleanliness": "cleanliness_base",
        "comfort": "comfort_base",
        "facilities": "facilities_base",
        "value for money": "value_for_money_base",
        "value": "value_for_money_base",
        "review score": "average_reviews_score",
        "average review score": "average_reviews_score",
    }

    for keyword, metric_name in metric_map.items():
        if keyword in text_lower:
            entities["metric"] = metric_name
            break

    # ---------------------------------------------------------
    # AGE EXTRACTION (explicit age → age_group)
    # ---------------------------------------------------------
    age_patterns = [
        r'(\d{2})\s+year',   # "26 years", "26 year old"
        r'age\s+(\d{2})',    # "age 26"
        r'aged\s+(\d{2})',   # "aged 26"
    ]
    for pattern in age_patterns:
        match = re.search(pattern, text_lower)
        if match:
            try:
                age = int(match.group(1))
                if 18 <= age <= 100:
                    age_group = map_age_to_group(age)
                    if age_group:
                        entities["age_group"] = age_group
                break
            except ValueError:
                pass

    # ---------------------------------------------------------
    # GENDER
    # ---------------------------------------------------------
    if any(w in text_lower for w in ["female", "women", "woman"]):
        entities["gender"] = "Female"
    elif any(w in text_lower for w in ["male", "men", "man"]):
        entities["gender"] = "Male"

    # ---------------------------------------------------------
    # TRAVELLER TYPE (generic keywords)
    # ---------------------------------------------------------
    traveller_type_map = {
        "solo": "Solo",
        "family": "Family",
        "couple": "Couple",
        "couples": "Couple",
        "group": "Group",
        "groups": "Group",
        "business": "Business",
    }

    for token in text_lower.split():
        if token in traveller_type_map:
            # Do not overwrite a honeymoon-based Couple if already set
            if "traveller_type" not in entities:
                entities["traveller_type"] = traveller_type_map[token]
            # If it is already "Couple" from honeymoon, leave it
            break

    return entities


# =====================================================================
# TEST BLOCK – READS QUERIES FROM queries.txt
# =====================================================================
if __name__ == "__main__":

    queries_path = "queries.txt"

    # Load queries from file
    try:
        with open(queries_path, "r", encoding="utf-8") as f:
            test_queries = [
                line.strip().strip('"')
                for line in f.readlines()
                if line.strip()
            ]
    except FileNotFoundError:
        print(f"ERROR: Could not find {queries_path}")
        exit(1)

    # Run extractor
    for query in test_queries:
        print("\n" + "=" * 60)
        print("QUERY:", query)
        print("-" * 60)

        extracted = extract_entities(query)

        for key, value in extracted.items():
            print(f"{key:20}: {value}")

        print("=" * 60)