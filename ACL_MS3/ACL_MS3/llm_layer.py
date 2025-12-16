"""
=======================================================================
MS3 – Step 3: LLM Layer (Exactly like Lab 8)
=======================================================================
"""

from huggingface_hub import InferenceClient
from typing import Dict, Any, List
import time

# =====================================================================
# HUGGINGFACE API SETUP
# =====================================================================

HF_TOKEN = "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"

# Initialize THREE different models (like Lab 8 requirement)
clients = {
    "gemma-2b": InferenceClient(model="google/gemma-2-2b-it", token=HF_TOKEN),
    "mistral-7b": InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.2", token=HF_TOKEN),
    "llama-3b": InferenceClient(model="meta-llama/Llama-3.2-3B-Instruct", token=HF_TOKEN)
}


# =====================================================================
# COMBINE BASELINE + EMBEDDINGS RESULTS
# =====================================================================

def combine_results(baseline_results: List[Dict], embedding_results: List[Dict]) -> str:
    """
    Combine baseline Cypher results and embedding-based results.
    Remove duplicates and create unified context.
    """
    all_results = []
    seen_ids = set()
    
    # Add baseline results
    for item in baseline_results:
        # Create composite ID for better duplicate detection
        if 'review_id' in item:
            item_id = f"review_{item['review_id']}"
        elif 'hotel_name' in item:
            item_id = f"hotel_{item['hotel_name']}"
        else:
            item_id = str(hash(str(item)))
        
        if item_id not in seen_ids:
            all_results.append(item)
            seen_ids.add(item_id)
    
    # Add embedding results (avoid duplicates)
    for item in embedding_results:
        # Create same composite ID structure
        if 'review_id' in item:
            item_id = f"review_{item['review_id']}"
        elif 'hotel_name' in item:
            item_id = f"hotel_{item['hotel_name']}"
        else:
            item_id = str(hash(str(item)))
        
        if item_id not in seen_ids:
            all_results.append(item)
            seen_ids.add(item_id)
    
    # Format results into readable context
    if not all_results:
        return "No relevant information found in the knowledge graph."
    
    context_parts = []
    for idx, item in enumerate(all_results, 1):  # Limit to top 10
        context_parts.append(f"{idx}. {item}")
    
    return "\n".join(context_parts)


# =====================================================================
# STRUCTURED PROMPT (Context + Persona + Task)
# =====================================================================

def build_prompt(user_query: str, kg_context: str) -> str:
    """
    Build structured prompt with Context, Persona, Task,
    plus extended MS3/Lab8 constraints and strict anti-hallucination rules.
    """

    prompt = f"""You are a helpful travel assistant with access to a comprehensive travel knowledge graph database.
Your job is to answer hotel-related queries with PERFECT grounding, absolute accuracy, and no hallucinations.

CONTEXT - Here is the relevant information retrieved from the knowledge graph:

{kg_context}

TASK:
- Answer the user's question using ONLY the information provided above.
- Be specific and include all relevant details (hotel names, ratings, cities, scores, etc.)
- Format your answer in a clear, helpful way.
- If the context contains the answer, provide it completely.
-focus on the context info even if it is 1 hotel only .


==========================
ADDITIONAL MANDATORY RULES
==========================

STRICT GROUNDEDNESS RULES:
- Do NOT guess, infer, assume, or invent anything that is not explicitly written in the context.
- Do NOT compute new values, do NOT average metrics, do NOT merge repeated data unless context explicitly provides a merged value.
- If a property (city, country, star rating, metric) does not appear in the context, YOU MUST NOT invent or assume it.
-if there is any data in the form of ,hotel_name,star_rating,avg_score,traveller_age_group,city,country in the context it is always relevant TAKE IT.
-if the context is Fully empty dont invent anything.

ANTI-DUPLICATION RULES:
- ABSOLUTE RULE: Never list the same hotel name twice under any circumstances.
- Even if the same hotel appears multiple times in the context, you must treat it as ONE unique hotel.
- If the user asks for "top N", you MUST return exactly N DISTINCT hotels IF context contains at least N distinct hotels.
- Never fill missing positions by repeating a hotel.

STRICT REVIEW OUTPUT RULES:
- If the user asks for reviews of a hotel, and the context contains MULTIPLE reviews for that hotel:
  **You MUST return multiple reviews, not only one.**
- If the context contains 3 reviews, show all 3.
- If the context contains many reviews, show several clearly (3–5 depending on context order).
- NEVER output only one review when the context provides more than one.

OUTPUT BEHAVIOR RULES:
- Do NOT say based on the provided context.
- Use clear numbered or bulleted lists and natural english language .
- Include ONLY and ALL details that appear explicitly in the context in natural way .
- No creative writing, no assumptions, no "likely", "probably", or "estimated".
- No internal reasoning, chain-of-thought, or explanations of how you derived answers.
- DO NOT say based on the given context and provide the answer directly wothout changing it to natural language.
- At the end, in a short, direct, straight-to-the-point, natural English way, explain why this hotel is what the user asked for based on the provided context.


PERSONA RULES:
- You are a precise Graph-RAG system.
- You ALWAYS rely on structured graph data over language priors.
- You act as a hotel recommender that prioritizes factual correctness and distinctness.

FAILURE CONDITION:
if the context is truly FULLY empty , reply with EXACTLY:
  "The knowledge graph does not contain the answer."



USER QUESTION: {user_query}

YOUR ANSWER (follow ALL rules above):"""

    return prompt
    



# =====================================================================
# CALL LLM (Like Lab 8)
# =====================================================================

def call_llm(model_name: str, prompt: str, max_tokens: int = 500) -> Dict[str, Any]:
    """
    Call HuggingFace LLM and return response with metrics
    """
    client = clients[model_name]
    
    start_time = time.time()
    
    try:
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        
        end_time = time.time()
        
        answer = response.choices[0].message["content"]
        
        return {
            "model": model_name,
            "answer": answer,
            "response_time": end_time - start_time,
            "success": True
        }
    
    except Exception as e:
        end_time = time.time()
        return {
            "model": model_name,
            "answer": f"Error: {str(e)}",
            "response_time": end_time - start_time,
            "success": False
        }


# =====================================================================
# COMPLETE RAG PIPELINE
# =====================================================================

def rag_answer(user_query: str, baseline_results: List[Dict], 
               embedding_results: List[Dict], model_name: str = "gemma-2b") -> Dict[str, Any]:
    """
    Complete RAG pipeline:
    1. Combine baseline + embedding results
    2. Build structured prompt
    3. Call LLM
    4. Return answer with metrics
    """
    # Step 1: Combine results
    kg_context = combine_results(baseline_results, embedding_results)
    
    # Step 2: Build prompt
    prompt = build_prompt(user_query, kg_context)
    
    # Step 3: Call LLM
    result = call_llm(model_name, prompt)
    
    # Step 4: Add context info
    result["kg_context"] = kg_context
    result["prompt"] = prompt
    
    return result


# =====================================================================
# COMPARE MULTIPLE MODELS
# =====================================================================

def compare_models(user_query: str, baseline_results: List[Dict], 
                   embedding_results: List[Dict]) -> Dict[str, Dict]:
    """
    Compare all three models on the same query
    """
    kg_context = combine_results(baseline_results, embedding_results)
    prompt = build_prompt(user_query, kg_context)
    
    results = {}
    
    print("\n" + "=" * 80)
    print(f"COMPARING MODELS FOR QUERY: {user_query}")
    print("=" * 80)
    
    for model_name in ["gemma-2b", "mistral-7b", "llama-3b"]:
        print(f"\nTesting {model_name}...")
        result = call_llm(model_name, prompt)
        results[model_name] = result
        
        print(f"✓ Response time: {result['response_time']:.2f}s")
        if result['success']:
            print(f"✓ Answer preview: {result['answer'][:100]}...")
        else:
            print(f"✗ Error: {result['answer']}")
    
    return results


# =====================================================================
# TESTING
# =====================================================================

if __name__ == "__main__":
    # Example test data (replace with actual results from your baseline/embedding)
    
    test_baseline_results = [
        {"hotel_name": "The Royal Compass", "star_rating": 5, "city": "London"},
        {"hotel_name": "Colosseum Gardens", "star_rating": 4, "city": "Rome"}
    ]
    
    test_embedding_results = [
        {"hotel_name": "The Royal Compass", "review_text": "Perfect honeymoon hotel", "similarity_score": 0.95},
        {"hotel_name": "Nile Grandeur", "review_text": "Best value in Cairo", "similarity_score": 0.87}
    ]
    
    test_query = "Which hotels in Europe are suitable for a honeymoon?"
    
    # Test single model
    print("\n" + "=" * 80)
    print("SINGLE MODEL TEST")
    print("=" * 80)
    result = rag_answer(test_query, test_baseline_results, test_embedding_results, "gemma-2b")
    print(f"\nAnswer: {result['answer']}")
    print(f"Response time: {result['response_time']:.2f}s")
    
    # Test all models
    print("\n" + "=" * 80)
    print("MULTI-MODEL COMPARISON")
    print("=" * 80)
    compare_results = compare_models(test_query, test_baseline_results, test_embedding_results)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for model_name, result in compare_results.items():
        print(f"\n{model_name}:")
        print(f"  Time: {result['response_time']:.2f}s")
        print(f"  Success: {result['success']}")
        if result['success']:
            print(f"  Answer: {result['answer'][:150]}...")