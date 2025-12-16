"""
Test LLM Layer with actual baseline + embedding results
Tests ALL queries from queries.txt
"""

from baseline_retrieval import Neo4jRetriever, get_baseline_results_for_llm
from embedding_retrieval import load_reviews_from_csv, load_embedding_models, build_faiss_index, get_embedding_results_for_llm
from llm_layer import rag_answer, compare_models, combine_results, build_prompt

# Initialize retrievers
print("\n" + "=" * 100)
print("INITIALIZING SYSTEM")
print("=" * 100)

neo4j_retriever = Neo4jRetriever()

# Load embedding setup
df = load_reviews_from_csv("reviews.csv", start_id=50001, end_id=50500)
models = load_embedding_models()
embedder = models["model2"]
review_texts = df['review_text'].tolist()
embeddings = embedder.encode(review_texts, convert_to_numpy=True)
index = build_faiss_index(embeddings)

# Load all queries from queries.txt
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

print(f"\n✓ Loaded {len(test_queries)} queries from {queries_path}")
print("=" * 100)

# Process each query
for query_num, test_query in enumerate(test_queries, 1):
    
    print("\n\n" + "█" * 100)
    print(f"QUERY {query_num}/{len(test_queries)}: {test_query}")
    print("█" * 100)
    
    # STEP 1: BASELINE RETRIEVAL
    print("\n" + "=" * 100)
    print("STEP 1: BASELINE RETRIEVAL (FROM KNOWLEDGE GRAPH)")
    print("=" * 100)
    baseline_results = get_baseline_results_for_llm(test_query, neo4j_retriever)
    print(f"\n✓ Number of results from baseline: {len(baseline_results)}")
    
    if baseline_results:
        print("\nBaseline Results:")
        for i, result in enumerate(baseline_results[:3], 1):  # Show first 3
            print(f"  {i}. {result}")
    else:
        print("\n  ⚠️  No baseline results (semantic query - will use embeddings)")
    
    # STEP 2: EMBEDDING RETRIEVAL
    print("\n" + "=" * 100)
    print("STEP 2: EMBEDDING RETRIEVAL (FROM REVIEW SIMILARITY)")
    print("=" * 100)
    embedding_results = get_embedding_results_for_llm(test_query, embedder, index, df, k=5)
    print(f"\n✓ Number of results from embeddings: {len(embedding_results)}")
    
    print("\nTop 5 Embedding Results:")
    for i, result in enumerate(embedding_results[:5], 1):
        print(f"  {i}. Review ID: {result['review_id']} | Similarity: {result['similarity_score']:.4f}")
        print(f"     Text: {result['review_text'][:100]}...")
    
    # STEP 3: COMBINE
    print("\n" + "=" * 100)
    print("STEP 3: COMBINE BASELINE + EMBEDDINGS")
    print("=" * 100)
    combined_context = combine_results(baseline_results, embedding_results)
    print(f"\n✓ Combined {len(baseline_results)} baseline + {len(embedding_results)} embedding results")
    print("\nCombined Context (first 500 chars):")
    print(combined_context[:500] + "...")
    
    # STEP 4: BUILD PROMPT
    print("\n" + "=" * 100)
    print("STEP 4: BUILD STRUCTURED PROMPT")
    print("=" * 100)
    full_prompt = build_prompt(test_query, combined_context)
    print("\n✓ Prompt built with Context + Persona + Task structure")
    print(f"\n✓ Prompt length: {len(full_prompt)} characters")
    
    # STEP 5: LLM COMPARISON
    print("\n" + "=" * 100)
    print("STEP 5: LLM MODEL COMPARISON")
    print("=" * 100)
    
    compare_results_dict = compare_models(test_query, baseline_results, embedding_results)
    
    # Show each model's output
    print("\n" + "=" * 100)
    print("RESULTS FROM EACH MODEL")
    print("=" * 100)
    
    for model_name, result in compare_results_dict.items():
        print("\n" + "─" * 100)
        print(f"🤖 MODEL: {model_name.upper()}")
        print("─" * 100)
        print(f"⏱️  Response Time: {result['response_time']:.2f} seconds")
        print(f"✅ Success: {result['success']}")
        print(f"\n📝 ANSWER:\n")
        if result['success']:
            # Show full answer but limit to 500 chars for readability
            if len(result['answer']) > 500:
                print(result['answer'][:500] + "...\n[truncated for display]")
            else:
                print(result['answer'])
        else:
            print(result['answer'])
        print("─" * 100)
    
    # Summary for this query
    print("\n" + "=" * 100)
    print(f"SUMMARY FOR QUERY {query_num}")
    print("=" * 100)
    
    print(f"\n{'Model':<15} {'Time (sec)':<15} {'Success':<10} {'Answer Length':<15}")
    print("-" * 55)
    for model_name, result in compare_results_dict.items():
        answer_len = len(result['answer']) if result['success'] else 0
        print(f"{model_name:<15} {result['response_time']:<15.2f} {str(result['success']):<10} {answer_len:<15}")
    
    fastest = min(compare_results_dict.items(), key=lambda x: x[1]['response_time'])
    print(f"\n🏆 Fastest: {fastest[0].upper()} ({fastest[1]['response_time']:.2f}s)")
    
    longest = max(compare_results_dict.items(), key=lambda x: len(x[1]['answer']) if x[1]['success'] else 0)
    print(f"📊 Most detailed: {longest[0].upper()} ({len(longest[1]['answer'])} chars)")
    
    print("\n" + "=" * 100)
    print(f"END OF QUERY {query_num}")
    print("=" * 100)


# FINAL OVERALL SUMMARY
print("\n\n" + "█" * 100)
print("FINAL OVERALL SUMMARY - ALL QUERIES")
print("█" * 100)

print(f"\n✅ Successfully tested {len(test_queries)} queries")
print("\nQueries tested:")
for i, q in enumerate(test_queries, 1):
    print(f"  {i}. {q}")

neo4j_retriever.close()

print("\n✅ ALL TESTS COMPLETED!")
print("=" * 100)