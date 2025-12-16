"""
=======================================================================
MS3 – Step 3: LLM Layer (Graph-RAG Integration)
=======================================================================
Combines baseline Cypher retrieval + embedding retrieval → LLM generation
"""

# =====================================================================
# IMPORTS
# =====================================================================
from huggingface_hub import InferenceClient
from typing import Dict, List, Any, Optional
import os
import json
from datetime import datetime

# Import your existing modules
from baseline_retrieval import run_baseline_pipeline, Neo4jRetriever
# from embedding_retrieval import semantic_search, load_embedding_models, build_faiss_index


# =====================================================================
# LLM MODEL CONFIGURATION
# =====================================================================

# Store your HuggingFace token securely
# For production, use environment variables or Kaggle secrets
HF_TOKEN = os.getenv("HF_TOKEN", "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")

# Define available models for comparison
LLM_MODELS = {
    "gemma-2b": "google/gemma-2-2b-it",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
    "llama-3b": "meta-llama/Llama-3.2-3B-Instruct"
}


# =====================================================================
# LLM CLIENT WRAPPER
# =====================================================================

class LLMClient:
    """Wrapper for HuggingFace Inference API with multiple model support"""
    
    def __init__(self, model_name: str = "gemma-2b", token: str = None):
        """
        Initialize LLM client
        
        Args:
            model_name: Key from LLM_MODELS dict or full model path
            token: HuggingFace API token
        """
        self.model_name = model_name
        self.model_path = LLM_MODELS.get(model_name, model_name)
        self.token = token or HF_TOKEN
        
        if not self.token or self.token == "your_token_here":
            raise ValueError("❌ HuggingFace token not set! Set HF_TOKEN environment variable.")
        
        self.client = InferenceClient(
            model=self.model_path,
            token=self.token
        )
        
        print(f"✓ LLM Client initialized: {self.model_path}")
    
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        """
        Generate response from LLM
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
        
        Returns:
            Generated text response
        """
        try:
            response = self.client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message["content"]
        except Exception as e:
            return f"❌ LLM Error: {str(e)}"


# =====================================================================
# CONTEXT COMBINATION
# =====================================================================

def combine_retrieval_results(
    baseline_results: List[Dict],
    embedding_results: Optional[List[Dict]] = None
) -> str:
    """
    Combine baseline Cypher results and embedding results into unified context
    
    Args:
        baseline_results: Results from Cypher queries
        embedding_results: Results from semantic search (optional)
    
    Returns:
        Formatted context string for LLM
    """
    context_parts = []
    
    # Add baseline results
    if baseline_results:
        context_parts.append("=== STRUCTURED DATA FROM KNOWLEDGE GRAPH ===")
        for idx, record in enumerate(baseline_results, 1):
            context_parts.append(f"\n{idx}. {json.dumps(record, indent=2)}")
    
    # Add embedding results if available
    if embedding_results:
        context_parts.append("\n\n=== SEMANTICALLY SIMILAR REVIEWS ===")
        for idx, record in enumerate(embedding_results, 1):
            review_text = record.get('review_text', 'N/A')
            similarity = record.get('similarity_score', 0)
            context_parts.append(f"\n{idx}. [Similarity: {similarity:.3f}]")
            context_parts.append(f"   Review: {review_text[:300]}...")
    
    return "\n".join(context_parts)


# =====================================================================
# STRUCTURED PROMPT BUILDER
# =====================================================================

def build_structured_prompt(
    user_query: str,
    context: str,
    intent: str,
    persona: str = "helpful travel assistant"
) -> str:
    """
    Build structured prompt with Context + Persona + Task
    
    Args:
        user_query: Original user question
        context: Retrieved KG context
        intent: Classified intent
        persona: Assistant persona/role
    
    Returns:
        Structured prompt for LLM
    """
    prompt = f"""You are a {persona} specializing in hotel recommendations and travel information.

CONTEXT (Retrieved from Knowledge Graph):
{context}

TASK:
Answer the following user question using ONLY the information provided in the context above.
If the context does not contain enough information to answer the question, clearly state that.
Do not make up or hallucinate information. Be specific and cite relevant details from the context.

USER QUESTION: {user_query}

DETECTED INTENT: {intent}

ANSWER:"""
    
    return prompt


# =====================================================================
# GRAPH-RAG PIPELINE
# =====================================================================

class GraphRAGPipeline:
    """Complete Graph-RAG pipeline integrating baseline + embeddings + LLM"""
    
    def __init__(self, 
                 neo4j_retriever: Neo4jRetriever,
                 llm_model: str = "gemma-2b",
                 use_embeddings: bool = True):
        """
        Initialize Graph-RAG pipeline
        
        Args:
            neo4j_retriever: Neo4j retriever instance
            llm_model: LLM model to use
            use_embeddings: Whether to use embedding-based retrieval
        """
        self.neo4j_retriever = neo4j_retriever
        self.llm_client = LLMClient(model_name=llm_model)
        self.use_embeddings = use_embeddings
        
        # Initialize embedding components if needed
        if self.use_embeddings:
            # You'll need to implement this based on your embedding_retrieval.py
            pass
    
    def retrieve_baseline(self, user_query: str) -> Dict[str, Any]:
        """Run baseline Cypher retrieval"""
        return run_baseline_pipeline(user_query, self.neo4j_retriever)
    
    def retrieve_embeddings(self, user_query: str, k: int = 5) -> List[Dict]:
        """Run embedding-based retrieval (placeholder)"""
        # TODO: Implement using your embedding_retrieval.py
        # return semantic_search(user_query, embedder, index, df, k=k)
        return []
    
    def answer(self, 
               user_query: str,
               max_tokens: int = 512,
               temperature: float = 0.2) -> Dict[str, Any]:
        """
        Complete Graph-RAG answer generation
        
        Args:
            user_query: User's question
            max_tokens: Max tokens for LLM response
            temperature: LLM sampling temperature
        
        Returns:
            Dict containing query, context, prompt, and answer
        """
        # Step 1: Baseline retrieval
        baseline_output = self.retrieve_baseline(user_query)
        
        # Step 2: Embedding retrieval (if enabled)
        embedding_results = []
        if self.use_embeddings:
            try:
                embedding_results = self.retrieve_embeddings(user_query)
            except Exception as e:
                print(f"⚠️ Embedding retrieval failed: {e}")
        
        # Step 3: Combine contexts
        context = combine_retrieval_results(
            baseline_output['result'],
            embedding_results if embedding_results else None
        )
        
        # Step 4: Build structured prompt
        prompt = build_structured_prompt(
            user_query=user_query,
            context=context,
            intent=baseline_output['intent'],
            persona="helpful travel assistant"
        )
        
        # Step 5: Generate answer with LLM
        answer = self.llm_client.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return {
            "query": user_query,
            "intent": baseline_output['intent'],
            "baseline_results": baseline_output['result'],
            "baseline_count": baseline_output['result_count'],
            "embedding_results": embedding_results,
            "embedding_count": len(embedding_results),
            "context": context,
            "prompt": prompt,
            "answer": answer,
            "model_used": self.llm_client.model_path,
            "timestamp": datetime.now().isoformat()
        }


# =====================================================================
# MODEL COMPARISON UTILITY
# =====================================================================

class ModelComparator:
    """Compare multiple LLM models on the same queries"""
    
    def __init__(self, neo4j_retriever: Neo4jRetriever):
        self.neo4j_retriever = neo4j_retriever
        self.models_to_compare = ["gemma-2b", "mistral-7b", "llama-3b"]
    
    def compare_models(self, user_query: str) -> Dict[str, Any]:
        """
        Run same query through multiple models
        
        Args:
            user_query: Question to test
        
        Returns:
            Dict with responses from each model
        """
        results = {
            "query": user_query,
            "models": {}
        }
        
        for model_name in self.models_to_compare:
            print(f"\n🤖 Testing model: {model_name}")
            try:
                pipeline = GraphRAGPipeline(
                    neo4j_retriever=self.neo4j_retriever,
                    llm_model=model_name,
                    use_embeddings=False
                )
                
                response = pipeline.answer(user_query)
                
                results["models"][model_name] = {
                    "answer": response["answer"],
                    "model_path": response["model_used"],
                    "timestamp": response["timestamp"]
                }
                
                print(f"✓ {model_name} completed")
                
            except Exception as e:
                print(f"❌ {model_name} failed: {e}")
                results["models"][model_name] = {
                    "answer": f"Error: {str(e)}",
                    "error": True
                }
        
        return results


# =====================================================================
# TESTING
# =====================================================================

if __name__ == "__main__":
    
    print("=" * 80)
    print("MS3 STEP 3: GRAPH-RAG LLM LAYER")
    print("=" * 80)
    
    # Initialize Neo4j retriever
    retriever = Neo4jRetriever()
    
    # Test queries
    test_queries = [
        "Which hotels in Europe are suitable for a honeymoon?",
        "Which hotels in Europe are pet-friendly?",
        "Which hotels in Barcelona have the best cleanliness scores?",
        "Show all hotels in Africa that are suitable for a senior trip."
    ]
    
    # Test 1: Single model Graph-RAG
    print("\n" + "=" * 80)
    print("TEST 1: SINGLE MODEL GRAPH-RAG")
    print("=" * 80)
    
    pipeline = GraphRAGPipeline(
        neo4j_retriever=retriever,
        llm_model="gemma-2b",
        use_embeddings=False
    )
    
    for query in test_queries[:2]:  # Test first 2 queries
        print(f"\n📝 Query: {query}")
        print("-" * 80)
        
        result = pipeline.answer(query)
        
        print(f"\n✓ Intent: {result['intent']}")
        print(f"✓ Baseline Results: {result['baseline_count']}")
        print(f"✓ Model: {result['model_used']}")
        print(f"\n🤖 ANSWER:\n{result['answer']}")
        print("\n" + "=" * 80)
    
    # Test 2: Model comparison
    print("\n" + "=" * 80)
    print("TEST 2: MODEL COMPARISON")
    print("=" * 80)
    
    comparator = ModelComparator(retriever)
    comparison_query = "Which hotels in Europe are suitable for a honeymoon?"
    
    print(f"\n📝 Query: {comparison_query}")
    comparison_results = comparator.compare_models(comparison_query)
    
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    for model_name, result in comparison_results["models"].items():
        print(f"\n🤖 {model_name.upper()}")
        print("-" * 80)
        print(result["answer"])
    
    retriever.close()
    print("\n✅ Testing completed!")