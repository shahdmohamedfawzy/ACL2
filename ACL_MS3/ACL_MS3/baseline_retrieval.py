"""
=======================================================================
MS3 – Step 2.a: Baseline Graph Retrieval Layer (Continent-Based)
=======================================================================
"""

# =====================================================================
# IMPORTS
# =====================================================================
import os
from neo4j import GraphDatabase
from typing import List, Dict
from intent_classifier import classify_intent
from entity_extractor import extract_entities


# =====================================================================
# LOAD CONFIG.TXT
# =====================================================================

CONFIG_FILE = "config.txt"

def load_config():
    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError("❌ config.txt not found in project folder!")

    config = {}
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                config[k] = v

    return config["URI"], config["USERNAME"], config["PASSWORD"]


# =====================================================================
# BASELINE CYPHER QUERIES (UPDATED FOR CONTINENTS)
# =====================================================================

CYPHER_QUERIES = {

    # =========================================================================
    # HOTEL SEARCH QUERIES
    # =========================================================================
    
    # Query 1: "List all hotels in Rome with a star rating above 3."
    "HOTEL_SEARCH_CITY_STAR_ABOVE": """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City {name: $city})
        WHERE h.star_rating > $star_rating
        RETURN h.name AS hotel_name, h.star_rating AS star_rating, 
               h.average_reviews_score AS avg_score, c.name AS city
        ORDER BY h.star_rating DESC, h.name ASC
    """,

    # Query 2: "Show the top 3 hotels in Asia with the highest average review scores?"
    "HOTEL_SEARCH_CONTINENT_TOP_BY_METRIC": """
        MATCH (h:Hotel)-[:LOCATED_IN]->(city:City)-[:LOCATED_IN]->(country:Country)-[:PART_OF]->(cont:Continent {name: $continent})
        RETURN h.name AS hotel_name, h.star_rating AS star_rating,
               h[$metric] AS metric_value, city.name AS city, country.name AS country
        ORDER BY h[$metric] DESC
        LIMIT $limit
    """,
    
    
    # Query 3: "Show all hotels in Africa suitable for a senior trip."
"HOTEL_SEARCH_CONTINENT_BY_AGE_GROUP": """
    MATCH (t:Traveller {age: $age_group})-[:STAYED_AT]->(h:Hotel)-[:LOCATED_IN]->(city:City)-[:LOCATED_IN]->(country:Country)-[:PART_OF]->(cont:Continent {name: $continent})
    RETURN DISTINCT 
           h.name AS hotel_name,
           h.star_rating AS star_rating,
           h.average_reviews_score AS avg_score,
           t.age AS traveller_age_group,
           city.name AS city,
           country.name AS country
    ORDER BY h.average_reviews_score DESC
""",

    # Query 4: "Which hotels in Africa were visited by female solo travellers?"
    "HOTEL_SEARCH_CONTINENT_BY_GENDER_AND_TYPE": """
        MATCH (t:Traveller {gender: $gender, type: $traveller_type})
              -[:STAYED_AT]->(h:Hotel)-[:LOCATED_IN]->(city:City)-[:LOCATED_IN]->(country:Country)-[:PART_OF]->(cont:Continent {name: $continent})
        RETURN DISTINCT h.name AS hotel_name, h.star_rating AS star_rating,
               h.average_reviews_score AS avg_score, city.name AS city, country.name AS country
        ORDER BY h.average_reviews_score DESC
    """,

# Query 6: "Show best hotel in cairo that offer the best value for money."
    "HOTEL_SEARCH_CITY_BEST_VALUE": """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City {name: $city})
        RETURN h.name AS hotel_name, h.star_rating AS star_rating,
               h.value_for_money_base AS value_for_money_score, c.name AS city
        ORDER BY h.value_for_money_base DESC
        LIMIT 1
    """,
    # Query 7: "Which hotel in Barcelona have the best cleanliness scores?"
    "HOTEL_SEARCH_CITY_BEST_CLEANLINESS": """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City {name: $city})
        RETURN h.name AS hotel_name, h.cleanliness_base AS cleanliness_score,
               h.average_reviews_score AS avg_score, c.name AS city
        ORDER BY h.cleanliness_base DESC
        LIMIT 1
    """,

    # Query 8: "Which hotels in Europe are pet-friendly?"
    "HOTEL_SEARCH_CONTINENT_BY_AMENITY": """
        MATCH (h:Hotel)-[:LOCATED_IN]->(city:City)-[:LOCATED_IN]->(country:Country)-[:PART_OF]->(cont:Continent {name: $continent})
        WHERE h.amenity = $amenity
        RETURN h.name AS hotel_name, h.star_rating AS star_rating,
               h.average_reviews_score AS avg_score, h.amenity AS amenity, 
               city.name AS city, country.name AS country
        ORDER BY h.average_reviews_score DESC
    """,

    # =========================================================================
    # VISA INFO QUERIES
    # =========================================================================
    
    # Query 9: "List all countries that allow travellers from Turkey to enter without a visa."
    "VISA_LIST_NO_REQUIREMENT": """
        MATCH (from:Country {name: $from_country})
        MATCH (to:Country)
        WHERE NOT (from)-[:NEEDS_VISA]->(to) AND from <> to
        RETURN to.name AS country
        ORDER BY country ASC
    """,

    # =========================================================================
    # REVIEW LOOKUP QUERIES
    # =========================================================================
    
    # Query 10: "Show all reviews from the year 2020 for The Gateway Royale."
    "REVIEWS_FOR_HOTEL_YEAR": """
        MATCH (r:Review)-[:REVIEWED]->(h:Hotel {name: $hotel_name})
        WHERE r.date STARTS WITH $year
        RETURN r.review_id AS review_id, r.text AS review_text,
               r.date AS review_date, r.score_overall AS score,
               h.name AS hotel_name
        ORDER BY r.date DESC
        LIMIT 50
    """,
}


# =====================================================================
# TEMPLATE SELECTOR (UPDATED FOR CONTINENTS + SEMANTIC DETECTION)
# =====================================================================

def select_cypher_template(intent: str, params: dict) -> str:
    """
    Select the appropriate Cypher query template based on intent and entities.
    Returns "SEMANTIC_ONLY" for queries that should use embeddings instead of Cypher.
    """
    # --------------------------
    # HOTEL SEARCH
    # --------------------------
    if intent == "HOTEL_SEARCH":
        
        # CONTINENT-BASED QUERIES (Priority!)
        if "continent" in params:
            # Query 8: Pet-friendly hotels in continent
            if "amenity" in params:
                return "HOTEL_SEARCH_CONTINENT_BY_AMENITY"
            
            # Query 2: Top N hotels by metric in continent
            if "metric" in params and "limit" in params:
                return "HOTEL_SEARCH_CONTINENT_TOP_BY_METRIC"
            
            # Query 4: Gender + traveller type in continent
            if "gender" in params and "traveller_type" in params:
                return "HOTEL_SEARCH_CONTINENT_BY_GENDER_AND_TYPE"
            
            # Query 5: Traveller type in continent (honeymoon)
            # SEMANTIC QUERY - Skip baseline, use embeddings only
            if "traveller_type" in params:
                return "SEMANTIC_ONLY"
            
            # Query 3: Age group in continent (senior trip)
            # SEMANTIC QUERY - Skip baseline, use embeddings only
            # Query 3: Age group in continent (senior trip)
            if "age_group" in params:
              return "HOTEL_SEARCH_CONTINENT_BY_AGE_GROUP"

        
        # CITY-BASED QUERIES (Fallback)
        if "city" in params:
            # Query 1: Star rating above threshold
            if "star_rating" in params:
                return "HOTEL_SEARCH_CITY_STAR_ABOVE"
            
            # Query 7: Best cleanliness score
            if "metric" in params and params["metric"] == "cleanliness_base":
                return "HOTEL_SEARCH_CITY_BEST_CLEANLINESS"
            
            # Query 6: Best value for money
            if params.get("metric") in ["score_value_for_money", "value_for_money_base"]:
                return "HOTEL_SEARCH_CITY_BEST_VALUE"
        
        raise ValueError(f"HOTEL_SEARCH but insufficient entities. Params: {params}")

    # --------------------------
    # REVIEW LOOKUP
    # --------------------------
    if intent == "REVIEW_LOOKUP":
        # Query 10: Hotel name + year
        if "hotel_name" in params and "year" in params:
            return "REVIEWS_FOR_HOTEL_YEAR"

        raise ValueError(f"REVIEW_LOOKUP but insufficient entities. Params: {params}")

    # --------------------------
    # VISA INFO
    # --------------------------
    if intent == "VISA_INFO":
        # Query 9: From country only (list countries without visa requirement)
        if "from_country" in params:
            return "VISA_LIST_NO_REQUIREMENT"
        
        raise ValueError(f"VISA_INFO missing required entities. Params: {params}")

    # --------------------------
    # TRAVELLER LOOKUP (not in current queries)
    # --------------------------
    if intent == "TRAVELLER_LOOKUP":
        raise ValueError("TRAVELLER_LOOKUP intent not supported in current queries")

    raise ValueError(f"Unknown intent: {intent}")


# =====================================================================
# NEO4J RETRIEVER
# =====================================================================

class Neo4jRetriever:

    def __init__(self):
        uri, username, password = load_config()
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self):
        self.driver.close()

    def run(self, cypher_key: str, params: dict):
        """Execute Cypher query and return results"""
        # Check if this is a semantic-only query
        if cypher_key == "SEMANTIC_ONLY":
            print("  ℹ️  This is a SEMANTIC query - baseline returns empty, embeddings will handle it")
            return []  # Return empty list - embeddings will handle it
        
        cypher = CYPHER_QUERIES[cypher_key]
        with self.driver.session() as session:
            result = session.run(cypher, params)
            return [record.data() for record in result]


# =====================================================================
# BASELINE PIPELINE
# =====================================================================

def run_baseline_pipeline(user_query: str, retriever: Neo4jRetriever):
    """
    Complete pipeline: classify intent → extract entities → select query → execute
    """
    # Step 1: Classify intent
    intent = classify_intent(user_query)
    
    # Step 2: Extract entities
    entities = extract_entities(user_query)
    
    # Step 3: Prepare params
    params = entities.copy()
    
    # Fix year to string if present
    if "year" in params:
        params["year"] = str(params["year"])
    
    # Fix star_rating to float if present
    if "star_rating" in params:
        params["star_rating"] = float(params["star_rating"])
    
    # Add default limit if not present but needed
    if "limit" not in params and ("top" in user_query.lower() or "best" in user_query.lower()):
        import re
        top_match = re.search(r"top\s+(\d+)", user_query.lower())
        if top_match:
            params["limit"] = int(top_match.group(1))
        else:
            params["limit"] = 10  # default

    # Step 4: Select appropriate Cypher template
    cypher_key = select_cypher_template(intent, params)

    # Step 5: Execute query
    result = retriever.run(cypher_key, params)

    return {
        "query": user_query,
        "intent": intent,
        "entities": entities,
        "params": params,
        "template_used": cypher_key,
        "result": result,
        "result_count": len(result)
    }


# =====================================================================
# WRAPPER FOR LLM LAYER
# =====================================================================

def get_baseline_results_for_llm(user_query: str, retriever) -> List[Dict]:
    """
    Wrapper to get baseline results in format for LLM layer
    """
    pipeline_output = run_baseline_pipeline(user_query, retriever)
    return pipeline_output["result"]


# =====================================================================
# TESTING WITH NEW QUERIES (FROM queries.txt)
# =====================================================================

if __name__ == "__main__":

    # Load queries from queries.txt
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

    retriever = Neo4jRetriever()

    print("=" * 80)
    print("BASELINE RETRIEVAL TESTING (CONTINENT-BASED)")
    print("=" * 80)

    for i, q in enumerate(test_queries, 1):
        print(f"\n{i}. QUERY: {q}")
        print("-" * 80)
        
        try:
            out = run_baseline_pipeline(q, retriever)
            print(f"✓ Intent: {out['intent']}")
            print(f"✓ Template: {out['template_used']}")
            print(f"✓ Params: {out['params']}")
            print(f"✓ Results Found: {out['result_count']}")
            
            # Show first 3 results as preview
            if out['result']:
                print(f"\n  Preview (first 3 results):")
                for idx, record in enumerate(out['result'][:3], 1):
                    print(f"    {idx}. {record}")
            else:
                print("  ⚠️  No results found (may be handled by embeddings)")
                
        except Exception as e:
            print(f"✗ ERROR: {e}")
        
        print("=" * 80)

    retriever.close()
    print("\n✅ All tests completed!")