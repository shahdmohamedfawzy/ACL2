# intent_classifier.py

from typing import Literal

IntentType = Literal[
    "HOTEL_SEARCH",
    "REVIEW_LOOKUP",
    "TRAVELLER_LOOKUP",
    "VISA_INFO",
]


def classify_intent(user_query: str) -> IntentType:
    text = user_query.lower()

    # =================================================================
    # 1) VISA INFO — rules between countries
    # =================================================================
    visa_keywords = [
        "visa", "visas", "visa info", "visa information",
        "visa required", "visa requirement", "visa requirements",
        "visa rules", "visa regulations", "visa policy",
        "visa on arrival", "visa-free", "visa free",
        "enter without visa", "enter with visa",
        "allowed to enter", "allowed entry",
        "travel permit", "entry permit",
        "travel to", "travel from",
        "coming from", "going to",
        "border rules", "passport requirement",
        "is a visa needed", "do i need a visa",
    ]
    for kw in visa_keywords:
        if kw in text:
            return "VISA_INFO"

    # =================================================================
    # 2) REVIEW LOOKUP
    # =================================================================
    review_keywords = [
        "reviews", "guest reviews", "customer reviews",
        "show reviews", "list reviews",
        "guest feedback", "comments", "opinions",
        "guest experience",
        "reviews written in", "recent reviews",
    ]
    for kw in review_keywords:
        if kw in text:
            return "REVIEW_LOOKUP"

    # =================================================================
    # 3) HOTEL SEARCH
    # =================================================================
    hotel_keywords = [
        "hotels in", "hotel in", "find hotels", "list hotels",
        "best hotels", "top hotels", "which hotels",
        "star rating", "average review score",
        "rating above", "rating over",
        "cleanliness_base", "comfort_base",
        "with pool", "with wifi", "with spa",
        "beach access", "parking",
    ]
    for kw in hotel_keywords:
        if kw in text:
            return "HOTEL_SEARCH"

    # =================================================================
    # 4) TRAVELLER LOOKUP
    # =================================================================
    traveller_keywords = [
        "traveller", "travellers", "travelers",
        "female travellers", "male travellers",
        "solo travellers", "family travellers",
        "who stayed", "who visited", "who wrote reviews",
        "travellers from",
    ]
    for kw in traveller_keywords:
        if kw in text:
            return "TRAVELLER_LOOKUP"

    raise ValueError(f"Could not classify intent for query: {user_query!r}")


# =====================================================================
# TEST BLOCK — NOW READS FROM queries.txt
# =====================================================================
if __name__ == "__main__":

    queries_path = "queries.txt"

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

    for q in test_queries:
        try:
            print("\nQUERY:", q)
            print("PREDICTED INTENT:", classify_intent(q))
        except ValueError as e:
            print("ERROR:", e)
        print("--------------------------------------------------")
