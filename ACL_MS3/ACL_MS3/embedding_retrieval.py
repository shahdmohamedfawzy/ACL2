# feature_embeddings.py
# csv_embedding.py
"""
=======================================================================
Embed Review Text from CSV (Review IDs 50001-50500)
=======================================================================
Reads reviews.csv and embeds only the review_text column
"""

# =====================================================================
# IMPORTS
# =====================================================================
import pandas as pd
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# =====================================================================
# LOAD CSV DATA
# =====================================================================
def load_reviews_from_csv(csv_path="reviews.csv", start_id=50001, end_id=50500):
    """
    Load reviews from CSV and filter by review_id range.
    
    Args:
        csv_path: Path to the CSV file
        start_id: Starting review_id (inclusive)
        end_id: Ending review_id (inclusive)
    
    Returns:
        DataFrame with filtered reviews
    """
    print(f"Loading reviews from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Filter by review_id range
    df_filtered = df[(df['review_id'] >= start_id) & (df['review_id'] <= end_id)].copy()
    
    print(f"✓ Loaded {len(df_filtered)} reviews (IDs {start_id}-{end_id})")
    print(f"  Columns: {list(df_filtered.columns)}")
    
    return df_filtered


# =====================================================================
# EMBEDDING MODELS
# =====================================================================
def load_embedding_models():
    """Load two different embedding models for comparison"""
    print("\nLoading embedding models...")
    
    # Model 1: all-MiniLM-L6-v2 (faster, smaller)
    model1 = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("✓ Model 1 loaded: all-MiniLM-L6-v2")
    
    # Model 2: all-mpnet-base-v2 (more powerful)
    model2 = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    print("✓ Model 2 loaded: all-mpnet-base-v2")
    
    return {"model1": model1, "model2": model2}


# =====================================================================
# BUILD FAISS INDEX
# =====================================================================
def build_faiss_index(embeddings):
    """Build FAISS index for fast similarity search"""
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    print(f"✓ FAISS index built with {index.ntotal} vectors")
    return index


# =====================================================================
# SEMANTIC SEARCH
# =====================================================================
def semantic_search(query, embedder, index, df, k=5):
    """
    Perform semantic similarity search using embeddings.
    
    Args:
        query: User's search query
        embedder: SentenceTransformer model
        index: FAISS index
        df: Original DataFrame with reviews
        k: Number of results to return
    
    Returns:
        List of top-k matching reviews with similarity scores
    """
    # Embed the query
    query_emb = embedder.encode([query], convert_to_numpy=True)
    
    # Search in FAISS index
    distances, indices = index.search(query_emb, k)
    
    # Prepare results
    results = []
    for i, idx in enumerate(indices[0]):
        row = df.iloc[idx]
        similarity_score = 1 / (1 + distances[0][i])  # Convert distance to similarity
        
        results.append({
            "review_id": row["review_id"],
            "review_text": row["review_text"],
            "score_overall": row["score_overall"],
            "similarity_score": similarity_score
        })
    
    return results


# =====================================================================
# MAIN PIPELINE
# =====================================================================
def main():
    print("=" * 80)
    print("EMBEDDING REVIEW TEXT FROM CSV")
    print("=" * 80)
    
    # Step 1: Load CSV data
    print("\n[1] Loading reviews from CSV...")
    df = load_reviews_from_csv("reviews.csv", start_id=50001, end_id=50503)
    
    # Step 2: Extract review texts
    print("\n[2] Extracting review texts...")
    review_texts = df['review_text'].tolist()
    print(f"✓ Extracted {len(review_texts)} review texts")
    print(f"Example: '{review_texts[0][:100]}...'")
    
    # Step 3: Load embedding models
    print("\n[3] Loading embedding models...")
    models = load_embedding_models()
    
    # Step 4: Encode review texts with both models
    print("\n[4] Encoding review texts into embeddings...")
    
    # Model 1
    print("  → Encoding with Model 1 (all-MiniLM-L6-v2)...")
    embeddings_m1 = models["model1"].encode(review_texts, convert_to_numpy=True, show_progress_bar=True)
    print(f"    ✓ Shape: {embeddings_m1.shape}")
    
    # Model 2
    print("  → Encoding with Model 2 (all-mpnet-base-v2)...")
    embeddings_m2 = models["model2"].encode(review_texts, convert_to_numpy=True, show_progress_bar=True)
    print(f"    ✓ Shape: {embeddings_m2.shape}")
    
    # Step 5: Show one complete vector
    print("\n[5] Showing one complete embedding vector (Model 1)...")
    print(f"Review ID: {df.iloc[0]['review_id']}")
    print(f"Review text: '{review_texts[0][:100]}...'")
    print(f"\nComplete vector (384 dimensions):")
    print(embeddings_m1[0])
    print(f"\nVector shape: {embeddings_m1[0].shape}")
    print(f"Min value: {np.min(embeddings_m1[0]):.6f}")
    print(f"Max value: {np.max(embeddings_m1[0]):.6f}")
    print(f"Mean value: {np.mean(embeddings_m1[0]):.6f}")
    
    # Step 6: Build FAISS indices
    print("\n[6] Building FAISS indices...")
    index_m1 = build_faiss_index(embeddings_m1)
    index_m2 = build_faiss_index(embeddings_m2)
    
    # Step 7: Test semantic search with multiple queries
    print("\n[7] Testing semantic search...")
    
    test_queries = [
    "List all hotels in Rome with a star rating above 3.",
    "Show the top 3 hotels in Asia with the highest average review scores?",
    "Show all hotels in Africa that are suitable for a senior trip.",
    "Which hotels in Africa were visited by female solo travellers?",
    "Which hotels in Europe are suitable for a honeymoon?",
    "Show best hotel in Cairo that offer the best value for money.",
    "Which hotel in Barcelona have the best cleanliness scores?",
    "Which hotels in Europe are pet friendly?",
    "I need places in South America where I can stay active during my trip. Which hotels have dedicated exercise spaces?",
    "Show all reviews from the year 2020 for The Gateway Royale."
    ]
    
    for query_num, test_query in enumerate(test_queries, 1):
        print("\n" + "=" * 80)
        print(f"QUERY {query_num}: '{test_query}'")
        print("=" * 80)
        
        # Model 1 Results
        print("\n--- MODEL 1 (all-MiniLM-L6-v2) ---")
        results_m1 = semantic_search(test_query, models["model1"], index_m1, df, k=5)
        for i, r in enumerate(results_m1, 1):
            print(f"\n{i}. Review ID: {r['review_id']} | Similarity: {r['similarity_score']:.4f}")
            print(f"   Overall Score: {r['score_overall']}")
            print(f"   Review: {r['review_text'][:200]}...")
        
        # Model 2 Results
        print("\n--- MODEL 2 (all-mpnet-base-v2) ---")
        results_m2 = semantic_search(test_query, models["model2"], index_m2, df, k=5)
        for i, r in enumerate(results_m2, 1):
            print(f"\n{i}. Review ID: {r['review_id']} | Similarity: {r['similarity_score']:.4f}")
            print(f"   Overall Score: {r['score_overall']}")
            print(f"   Review: {r['review_text'][:200]}...")
    
    print("\n" + "=" * 80)
    print("✅ Embedding completed!")
    print(f"Total reviews embedded: {len(review_texts)}")
    print(f"Model 1 dimensions: {embeddings_m1.shape[1]}")
    print(f"Model 2 dimensions: {embeddings_m2.shape[1]}")


# =====================================================================
# RUN
# =====================================================================
if __name__ == "__main__":
    main()


# At the end of embedding_retrieval.py, add:

def get_embedding_results_for_llm(query: str, embedder, index, df, k=5) -> List[Dict]:
    """
    Wrapper to get embedding results in format for LLM layer
    """
    return semantic_search(query, embedder, index, df, k)