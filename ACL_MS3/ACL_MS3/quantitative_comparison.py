"""
Enhanced Quantitative LLM Comparison Script
Adds: Token count, Correctness checking, Better visualizations
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from baseline_retrieval import Neo4jRetriever, get_baseline_results_for_llm
from embedding_retrieval import (
    load_reviews_from_csv, 
    load_embedding_models, 
    build_faiss_index, 
    get_embedding_results_for_llm
)
from llm_layer import rag_answer, compare_models
import time
from typing import Dict, List
import re

# =====================================================================
# CORRECTNESS CHECKING FUNCTIONS
# =====================================================================

def check_correctness(query: str, query_idx: int, answer: str, baseline_results: List[Dict]) -> Dict:
    """
    Check if the answer is correct based on baseline results
    Returns dict with correctness info
    """
    
    # Query 5: Honeymoon in Europe - special case (check for London hotel)
    if query_idx == 5:
        # Look for "The Royal Compass" or any hotel in London
        london_hotels = ["royal compass", "london"]
        is_correct = any(hotel.lower() in answer.lower() for hotel in london_hotels)
        
        return {
            'is_correct': is_correct,
            'expected': "Hotel in London (e.g., The Royal Compass)",
            'found': "London hotel mentioned" if is_correct else "No London hotel found",
            'match_type': 'location_check'
        }
    
    # Query 10: Reviews from 2020 - special case (check for review IDs)
    elif query_idx == 10:
        # Extract review IDs from baseline
        baseline_review_ids = set()
        for result in baseline_results:
            if 'review_id' in result:
                baseline_review_ids.add(str(result['review_id']))
        
        # Check if answer contains any of the review IDs
        found_review_ids = set()
        for review_id in baseline_review_ids:
            if review_id in answer:
                found_review_ids.add(review_id)
        
        is_correct = len(found_review_ids) > 0
        
        return {
            'is_correct': is_correct,
            'expected': f"{len(baseline_review_ids)} review IDs from baseline",
            'found': f"{len(found_review_ids)} review IDs mentioned",
            'match_type': 'review_id_check',
            'baseline_ids': baseline_review_ids,
            'found_ids': found_review_ids
        }
    
    # All other queries: Check for hotel names from baseline
    else:
        # Extract hotel names from baseline
        baseline_hotels = set()
        for result in baseline_results:
            if 'hotel_name' in result:
                baseline_hotels.add(result['hotel_name'].lower())
        
        # Check if answer contains any baseline hotel names
        found_hotels = set()
        for hotel in baseline_hotels:
            if hotel in answer.lower():
                found_hotels.add(hotel)
        
        is_correct = len(found_hotels) > 0
        
        return {
            'is_correct': is_correct,
            'expected': f"{len(baseline_hotels)} hotels from baseline",
            'found': f"{len(found_hotels)} hotels mentioned",
            'match_type': 'hotel_name_check',
            'baseline_hotels': baseline_hotels,
            'found_hotels': found_hotels
        }


def estimate_tokens(text: str) -> int:
    """
    Estimate token count (rough approximation: 1 token ≈ 4 characters)
    """
    return len(text) // 4


# =====================================================================
# INITIALIZE SYSTEM
# =====================================================================

print("\n" + "=" * 100)
print("INITIALIZING ENHANCED QUANTITATIVE LLM COMPARISON")
print("=" * 100)

neo4j_retriever = Neo4jRetriever()

# Load embedding setup
df = load_reviews_from_csv("reviews.csv", start_id=50001, end_id=50500)
models = load_embedding_models()
embedder = models["model2"]
review_texts = df['review_text'].tolist()
embeddings = embedder.encode(review_texts, convert_to_numpy=True)
index = build_faiss_index(embeddings)

# Load queries
queries_path = "queries.txt"
with open(queries_path, "r", encoding="utf-8") as f:
    test_queries = [line.strip().strip('"') for line in f.readlines() if line.strip()]

print(f"\n✓ Loaded {len(test_queries)} queries")
print(f"✓ Testing 3 models: Gemma 2B, Mistral 7B, Llama 3B")

# =====================================================================
# COLLECT METRICS FOR ALL QUERIES
# =====================================================================

all_results = []
model_names = ["gemma-2b", "mistral-7b", "llama-3b"]

for query_idx, query in enumerate(test_queries, 1):
    print(f"\n{'=' * 100}")
    print(f"Query {query_idx}/{len(test_queries)}: {query}")
    print("=" * 100)
    
    # Get retrieval results
    baseline_results = get_baseline_results_for_llm(query, neo4j_retriever)
    embedding_results = get_embedding_results_for_llm(query, embedder, index, df, k=5)
    
    print(f"  → Baseline results: {len(baseline_results)}")
    print(f"  → Embedding results: {len(embedding_results)}")
    
    # Test each model
    for model_name in model_names:
        print(f"\n  Testing {model_name}...")
        
        try:
            result = rag_answer(query, baseline_results, embedding_results, model_name)
            
            # Check correctness
            correctness_info = check_correctness(query, query_idx, result["answer"], baseline_results)
            
            # Estimate tokens
            tokens_used = estimate_tokens(result["answer"])
            
            # Extract metrics
            metrics = {
                "query_id": query_idx,
                "query": query,
                "model": model_name,
                "response_time": result["response_time"],
                "success": result["success"],
                "answer_length": len(result["answer"]) if result["success"] else 0,
                "tokens_used": tokens_used,
                "is_correct": correctness_info["is_correct"],
                "correctness_details": f"{correctness_info['found']} (expected: {correctness_info['expected']})",
                "answer": result["answer"][:300] + "..." if len(result["answer"]) > 300 else result["answer"],
                "baseline_count": len(baseline_results),
                "embedding_count": len(embedding_results)
            }
            
            all_results.append(metrics)
            
            status = "✓" if correctness_info["is_correct"] else "✗"
            print(f"    {status} Correct: {correctness_info['is_correct']}")
            print(f"    ⏱️  Time: {metrics['response_time']:.2f}s")
            print(f"    📊 Tokens: {metrics['tokens_used']}")
            print(f"    📝 {correctness_info['found']}")
            
        except Exception as e:
            print(f"    ✗ Error: {str(e)}")
            all_results.append({
                "query_id": query_idx,
                "query": query,
                "model": model_name,
                "response_time": 0,
                "success": False,
                "answer_length": 0,
                "tokens_used": 0,
                "is_correct": False,
                "correctness_details": f"Error: {str(e)}",
                "answer": f"Error: {str(e)}",
                "baseline_count": len(baseline_results),
                "embedding_count": len(embedding_results)
            })

# =====================================================================
# CREATE DATAFRAME AND SAVE RESULTS
# =====================================================================

df_results = pd.DataFrame(all_results)

# Save detailed results
df_results.to_csv("llm_quantitative_results_detailed.csv", index=False)
print(f"\n✓ Saved detailed results to llm_quantitative_results_detailed.csv")

# =====================================================================
# CALCULATE SUMMARY STATISTICS
# =====================================================================

summary = df_results.groupby("model").agg({
    "response_time": "mean",
    "success": lambda x: (x.sum() / len(x)) * 100,
    "answer_length": "mean",
    "tokens_used": "mean",
    "is_correct": lambda x: (x.sum() / len(x)) * 100  # Accuracy percentage
}).round(2)

summary.columns = ["Avg Response Time (s)", "Success Rate (%)", "Avg Answer Length", "Avg Tokens Used", "Accuracy (%)"]
summary = summary.reset_index()
summary.columns = ["Model", "Avg Response Time (s)", "Success Rate (%)", "Avg Answer Length", "Avg Tokens Used", "Accuracy (%)"]

print("\n" + "=" * 100)
print("QUANTITATIVE COMPARISON SUMMARY")
print("=" * 100)
print(summary.to_string(index=False))

# Save summary
summary.to_csv("llm_quantitative_summary.csv", index=False)
print(f"\n✓ Saved summary to llm_quantitative_summary.csv")

# =====================================================================
# VISUALIZATION 1: MAIN METRICS (4 BAR CHARTS)
# =====================================================================

print("\n" + "=" * 100)
print("GENERATING COMPARISON VISUALIZATIONS")
print("=" * 100)

# Custom color scheme for presentation
colors = ['#0b2f3d', '#ed8c02', '#234e5f']

sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("LLM Quantitative Comparison - All Metrics", fontsize=18, fontweight='bold', y=0.995)

# Graph 1: Response Time
axes[0, 0].bar(summary["Model"], summary["Avg Response Time (s)"], color=colors)
axes[0, 0].set_title("Average Response Time", fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel("Seconds", fontsize=12)
axes[0, 0].set_xlabel("Model", fontsize=12)
for i, v in enumerate(summary["Avg Response Time (s)"]):
    axes[0, 0].text(i, v + 0.05, f"{v:.2f}s", ha='center', fontweight='bold')

# Graph 2: Accuracy (NEW!)
axes[0, 1].bar(summary["Model"], summary["Accuracy (%)"], color=colors)
axes[0, 1].set_title("Accuracy (Correct Answers)", fontsize=14, fontweight='bold')
axes[0, 1].set_ylabel("Percentage (%)", fontsize=12)
axes[0, 1].set_xlabel("Model", fontsize=12)
axes[0, 1].set_ylim(0, 110)
for i, v in enumerate(summary["Accuracy (%)"]):
    axes[0, 1].text(i, v + 2, f"{v:.0f}%", ha='center', fontweight='bold')

# Graph 3: Tokens Used (NEW!)
axes[1, 0].bar(summary["Model"], summary["Avg Tokens Used"], color=colors)
axes[1, 0].set_title("Average Tokens Used", fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel("Tokens", fontsize=12)
axes[1, 0].set_xlabel("Model", fontsize=12)
for i, v in enumerate(summary["Avg Tokens Used"]):
    axes[1, 0].text(i, v + 5, f"{v:.0f}", ha='center', fontweight='bold')

# Graph 4: Success Rate
axes[1, 1].bar(summary["Model"], summary["Success Rate (%)"], color=colors)
axes[1, 1].set_title("Success Rate (No Errors)", fontsize=14, fontweight='bold')
axes[1, 1].set_ylabel("Percentage (%)", fontsize=12)
axes[1, 1].set_xlabel("Model", fontsize=12)
axes[1, 1].set_ylim(0, 110)
for i, v in enumerate(summary["Success Rate (%)"]):
    axes[1, 1].text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig("llm_quantitative_comparison_all_metrics.png", dpi=300, bbox_inches='tight')
print("✓ Saved: llm_quantitative_comparison_all_metrics.png")


# =====================================================================
# VISUALIZATION 3: CORRECTNESS BAR CHART (CLEARER VIEW)
# =====================================================================

# Count correct answers per model
correct_counts = df_results.groupby('model')['is_correct'].sum()
total_queries = len(test_queries)

plt.figure(figsize=(12, 6))
bars = plt.bar(correct_counts.index, correct_counts.values, color=colors, edgecolor='black', linewidth=2)

plt.title("Total Correct Answers (out of 10 queries)", fontsize=16, fontweight='bold', pad=15)
plt.xlabel("Model", fontsize=12, fontweight='bold')
plt.ylabel("Number of Correct Answers", fontsize=12, fontweight='bold')
plt.ylim(0, total_queries + 1)

# Add value labels on bars
for i, (bar, count) in enumerate(zip(bars, correct_counts.values)):
    plt.text(bar.get_x() + bar.get_width()/2, count + 0.2, 
             f'{count}/{total_queries}\n({count/total_queries*100:.0f}%)', 
             ha='center', fontsize=12, fontweight='bold')

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("llm_correctness_bar_chart.png", dpi=300, bbox_inches='tight')
print("✓ Saved: llm_correctness_bar_chart.png")

# =====================================================================
# VISUALIZATION 4: QUERY-BY-QUERY CORRECTNESS COMPARISON
# =====================================================================

fig, ax = plt.subplots(figsize=(14, 8))

# Prepare data
query_ids = sorted(df_results['query_id'].unique())
x = range(len(query_ids))
width = 0.25

# Plot bars for each model
for i, model in enumerate(model_names):
    model_data = df_results[df_results['model'] == model].sort_values('query_id')
    correctness = model_data['is_correct'].astype(int).values
    offset = (i - 1) * width
    bars = ax.bar([xi + offset for xi in x], correctness, width, 
                   label=model, color=colors[i], alpha=0.8, edgecolor='black')
    
    # Add checkmarks/crosses on bars
    for j, (bar, correct) in enumerate(zip(bars, correctness)):
        symbol = '✓' if correct else '✗'
        color = 'green' if correct else 'red'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
                symbol, ha='center', va='center', fontsize=14, 
                fontweight='bold', color=color)

ax.set_xlabel('Query ID', fontsize=12, fontweight='bold')
ax.set_ylabel('Correct (1) / Incorrect (0)', fontsize=12, fontweight='bold')
ax.set_title('Query-by-Query Correctness: All Models', fontsize=16, fontweight='bold', pad=15)
ax.set_xticks([xi for xi in x])
ax.set_xticklabels([f'Q{qid}' for qid in query_ids])
ax.set_ylim(0, 1.3)
ax.set_yticks([0, 1])
ax.set_yticklabels(['Wrong', 'Correct'])
ax.legend(loc='upper right', fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("llm_query_by_query_correctness.png", dpi=300, bbox_inches='tight')
print("✓ Saved: llm_query_by_query_correctness.png")

# =====================================================================
# PRINT DETAILED CORRECTNESS BREAKDOWN
# =====================================================================

print("\n" + "=" * 100)
print("DETAILED CORRECTNESS BREAKDOWN")
print("=" * 100)

for query_idx in range(1, len(test_queries) + 1):
    query_data = df_results[df_results['query_id'] == query_idx]
    query_text = query_data.iloc[0]['query']
    
    print(f"\nQuery {query_idx}: {query_text}")
    print("-" * 100)
    
    for _, row in query_data.iterrows():
        status = "✓ CORRECT" if row['is_correct'] else "✗ WRONG"
        print(f"  {row['model']:<15} {status:<15} | {row['correctness_details']}")

# =====================================================================
# CLOSE CONNECTION
# =====================================================================

neo4j_retriever.close()

print("\n" + "=" * 100)
print("✅ ENHANCED QUANTITATIVE COMPARISON COMPLETE!")
print("=" * 100)
print("\nGenerated files:")
print("  1. llm_quantitative_results_detailed.csv (all data with correctness)")
print("  2. llm_quantitative_summary.csv (includes accuracy)")
print("  3. llm_quantitative_comparison_all_metrics.png (4 bar charts)")
print("  5. llm_correctness_bar_chart.png (total correct per model)")
print("  6. llm_query_by_query_correctness.png (grouped bar chart)")
print("\n" + "=" * 100)