"""
=======================================================================
MS3 - Enhanced WhatsApp-Style Chat UI for Graph-RAG Travel Assistant
WITH RELATIONSHIP EXTRACTION FROM CYPHER QUERIES
=======================================================================
"""

# =====================================================================
# IMPORTS
# =====================================================================
import streamlit as st
import time
import json
import networkx as nx
import plotly.graph_objects as go
from typing import Dict, List, Tuple
import re

from baseline_retrieval import Neo4jRetriever, run_baseline_pipeline, CYPHER_QUERIES
from embedding_retrieval import (
    load_reviews_from_csv,
    load_embedding_models,
    build_faiss_index,
    get_embedding_results_for_llm
)
from llm_layer import rag_answer

# =====================================================================
# PAGE CONFIG
# =====================================================================
st.set_page_config(
    page_title="AI Travel Assistant",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# GLOBAL CSS (DARK + CLEAN + FIXED SIDEBAR)
# =====================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }

/* Background */
body { background: #0b0f1a; }
.stApp { background: transparent; }

/* App width */
.block-container {
    max-width: 1200px;
    padding-top: 1.5rem;
}

/* Header */
.app-header {
    background: linear-gradient(135deg, #6366f1, #a855f7);
    padding: 16px 20px;
    border-radius: 14px;
    color: white;
    margin-bottom: 14px;
}
.app-title { font-size: 1.3rem; font-weight: 700; }
.app-sub { font-size: 0.8rem; opacity: 0.9; }

/* Chat bubbles */
.user-bubble {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    padding: 10px 14px;
    border-radius: 16px 16px 4px 16px;
    margin: 8px 0 8px auto;
    max-width: 70%;
}
.ai-bubble {
    background: #242a52;
    color: #e5e7eb;
    padding: 10px 14px;
    border-radius: 16px 16px 16px 4px;
    margin: 8px auto 8px 0;
    max-width: 70%;
}
.bubble-label {
    font-size: 0.7rem;
    font-weight: 600;
    opacity: 0.7;
    margin-bottom: 4px;
}

/* Relationship box */
.relationship-box {
    background: #1a1f3a;
    border-left: 4px solid #a855f7;
    border-radius: 8px;
    padding: 12px;
    margin: 8px 0;
    font-family: 'Courier New', monospace;
    color: #e5e7eb;
}

.node-label {
    color: #6366f1;
    font-weight: 600;
}

.rel-label {
    color: #ec4899;
    font-weight: 600;
}

/* Retrieval method badge */
.retrieval-badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 8px;
}

.badge-baseline {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
}

.badge-embeddings {
    background: linear-gradient(135deg, #ec4899, #f43f5e);
    color: white;
}

.badge-both {
    background: linear-gradient(135deg, #10b981, #059669);
    color: white;
}

.badge-model {
    background: linear-gradient(135deg, #f59e0b, #d97706);
    color: white;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #ec4899, #f43f5e);
    border-radius: 12px;
    font-weight: 700;
    height: 48px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f1426;
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] label {
    color: #e5e7eb !important;
}
[data-testid="collapsedControl"] {
    display: none !important;
}

/* Expander styling */
.streamlit-expanderHeader {
    background: #1a1f3a !important;
    border-radius: 8px !important;
    color: #e5e7eb !important;
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# SESSION STATE
# =====================================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =====================================================================
# EXTRACT RELATIONSHIPS FROM CYPHER QUERY
# =====================================================================
def extract_relationships_from_cypher(cypher_query: str) -> List[Dict]:
    """
    Parse a Cypher query and extract all relationship patterns.
    Returns list of {source_node, relationship, target_node, direction}
    UPDATED: Better handling of chained patterns
    """
    relationships = []
    
    # Remove line breaks and extra spaces for easier parsing
    cypher_cleaned = ' '.join(cypher_query.split())
    
    # Pattern 1: (node1)-[:REL_TYPE]->(node2)
    # This pattern handles: (a:Label)-[:REL]->(b:Label)
    pattern1 = r'\((\w+):?(\w*)[^)]*\)-\[:(\w+)\]->\((\w+):?(\w*)[^)]*\)'
    matches1 = re.findall(pattern1, cypher_cleaned)
    
    for match in matches1:
        source_var = match[0]
        source_label = match[1] if match[1] else "Node"
        rel_type = match[2]
        target_var = match[3]
        target_label = match[4] if match[4] else "Node"
        
        relationships.append({
            'source_var': source_var,
            'source_label': source_label,
            'relationship': rel_type,
            'target_var': target_var,
            'target_label': target_label,
            'direction': 'outgoing'
        })
    
    # Pattern 2: (node1)<-[:REL_TYPE]-(node2)
    pattern2 = r'\((\w+):?(\w*)[^)]*\)<-\[:(\w+)\]-\((\w+):?(\w*)[^)]*\)'
    matches2 = re.findall(pattern2, cypher_cleaned)
    
    for match in matches2:
        target_var = match[0]
        target_label = match[1] if match[1] else "Node"
        rel_type = match[2]
        source_var = match[3]
        source_label = match[4] if match[4] else "Node"
        
        relationships.append({
            'source_var': source_var,
            'source_label': source_label,
            'relationship': rel_type,
            'target_var': target_var,
            'target_label': target_label,
            'direction': 'incoming'
        })
    
    # Pattern 3: Handle chained patterns like:
    # (t)-[:STAYED_AT]->(h)-[:LOCATED_IN]->(city)
    # This splits on MATCH and extracts chains
    match_clauses = re.findall(r'MATCH\s+(.*?)(?:RETURN|WHERE|WITH|$)', cypher_cleaned, re.IGNORECASE)
    
    for clause in match_clauses:
        # Find all sequential relationships in the chain
        # Pattern: )- or -> followed by ( 
        chain_pattern = r'\((\w+):?(\w*)[^)]*\)(?:\s*-\[:(\w+)\]->?\s*|\s*<-\[:(\w+)\]-\s*)(?=\()'
        
        # Split the clause into node-relationship pairs
        nodes_and_rels = re.findall(r'\((\w+):?(\w*).*?\)(?:\s*(-\[:\w+\]->?|<-\[:\w+\]-))?', clause)
        
        # Process sequential pairs
        for i in range(len(nodes_and_rels) - 1):
            current = nodes_and_rels[i]
            next_node = nodes_and_rels[i + 1]
            
            if current[2]:  # Has relationship
                rel_match = re.search(r'\[?:(\w+)\]?', current[2])
                if rel_match:
                    rel_type = rel_match.group(1)
                    
                    source_var = current[0]
                    source_label = current[1] if current[1] else "Node"
                    target_var = next_node[0]
                    target_label = next_node[1] if next_node[1] else "Node"
                    
                    direction = 'outgoing' if '->' in current[2] else 'incoming' if '<-' in current[2] else 'bidirectional'
                    
                    # Avoid duplicates
                    already_exists = any(
                        r['source_var'] == source_var and 
                        r['target_var'] == target_var and 
                        r['relationship'] == rel_type 
                        for r in relationships
                    )
                    
                    if not already_exists:
                        relationships.append({
                            'source_var': source_var,
                            'source_label': source_label,
                            'relationship': rel_type,
                            'target_var': target_var,
                            'target_label': target_label,
                            'direction': direction
                        })
    
    return relationships

# =====================================================================
# DISPLAY RELATIONSHIPS AS GRAPH PATTERN
# =====================================================================
def display_relationship_pattern(relationships: List[Dict]) -> str:
    """
    Create a visual representation of the graph pattern from Cypher.
    """
    if not relationships:
        return "No relationships found in query"
    
    pattern_lines = []
    pattern_lines.append("**Graph Pattern from Cypher Query:**")
    pattern_lines.append("")
    
    for rel in relationships:
        source = f"({rel['source_var']}:{rel['source_label']})" if rel['source_label'] else f"({rel['source_var']})"
        target = f"({rel['target_var']}:{rel['target_label']})" if rel['target_label'] else f"({rel['target_var']})"
        
        if rel['direction'] == 'outgoing':
            arrow = f"-[:{rel['relationship']}]->"
        elif rel['direction'] == 'incoming':
            arrow = f"<-[:{rel['relationship']}]-"
        else:
            arrow = f"-[:{rel['relationship']}]-"
        
        pattern_lines.append(f"```")
        pattern_lines.append(f"{source} {arrow} {target}")
        pattern_lines.append(f"```")
    
    return "\n".join(pattern_lines)

# =====================================================================
# GRAPH VISUALIZATION WITH RELATIONSHIPS FROM CYPHER
# =====================================================================
def create_graph_from_cypher(cypher_query: str, query_results: List[Dict]) -> go.Figure:
    """
    Create graph visualization based on Cypher query structure and results.
    """
    G = nx.DiGraph()
    
    # Extract relationships from Cypher
    relationships = extract_relationships_from_cypher(cypher_query)
    
    # Node styling
    node_colors = {
        'Hotel': '#a855f7', 'City': '#6366f1', 'Country': '#8b5cf6',
        'Continent': '#ec4899', 'Traveller': '#f59e0b', 'Review': '#10b981',
        'Query': '#ef4444', 'Node': '#64748b'
    }
    
    node_symbols = {
        'Hotel': 'circle', 'City': 'square', 'Country': 'diamond',
        'Continent': 'star', 'Traveller': 'triangle-up', 'Review': 'hexagon',
        'Query': 'star', 'Node': 'circle'
    }
    
    # Add nodes from query structure
    added_nodes = set()
    for rel in relationships:
        source_id = f"{rel['source_label']}:{rel['source_var']}"
        target_id = f"{rel['target_label']}:{rel['target_var']}"
        
        if source_id not in added_nodes:
            G.add_node(source_id, type=rel['source_label'], label=rel['source_var'])
            added_nodes.add(source_id)
        
        if target_id not in added_nodes:
            G.add_node(target_id, type=rel['target_label'], label=rel['target_var'])
            added_nodes.add(target_id)
        
        # Add edge
        G.add_edge(source_id, target_id, relation=rel['relationship'])
    
    # Add result nodes if any
    if query_results:
        for idx, result in enumerate(query_results[:5]):  # Limit to 5
            if 'hotel_name' in result:
                hotel_id = f"Hotel:result_{idx}"
                G.add_node(hotel_id, type='Hotel', label=result['hotel_name'])
                
                # Connect to query pattern
                for node_id in added_nodes:
                    if 'Hotel' in node_id:
                        G.add_edge(node_id, hotel_id, relation='RESULT')
                        break
    
    if len(G.nodes()) == 0:
        G.add_node("Empty", type="Node", label="No pattern found")
    
    # Layout
    pos = nx.spring_layout(G, k=3, iterations=50)
    
    # Edge traces
    edge_x, edge_y = [], []
    edge_text_x, edge_text_y, edge_text = [], [], []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
        edge_text_x.append(mid_x)
        edge_text_y.append(mid_y)
        edge_text.append(edge[2].get('relation', ''))
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#6366f1'),
        hoverinfo='none', mode='lines', showlegend=False
    )
    
    edge_label_trace = go.Scatter(
        x=edge_text_x, y=edge_text_y,
        mode='text', text=edge_text,
        textposition='middle center',
        textfont=dict(size=10, color='#ec4899', family='Courier New'),
        hoverinfo='none', showlegend=False
    )
    
    # Node traces by type
    node_traces = []
    seen_types = set()
    
    for node, data in G.nodes(data=True):
        node_type = data.get('type', 'Node')
        
        if node_type not in seen_types:
            nodes_of_type = [n for n, d in G.nodes(data=True) if d.get('type') == node_type]
            
            node_x = [pos[n][0] for n in nodes_of_type]
            node_y = [pos[n][1] for n in nodes_of_type]
            
            node_text = []
            for n in nodes_of_type:
                node_data = G.nodes[n]
                text = f"<b>{node_data.get('label', n)}</b><br>Type: {node_type}"
                node_text.append(text)
            
            color = node_colors.get(node_type, '#64748b')
            symbol = node_symbols.get(node_type, 'circle')
            
            trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                text=[G.nodes[n].get('label', str(n))[:20] for n in nodes_of_type],
                textposition="top center",
                textfont=dict(size=10, color='#e5e7eb'),
                hovertext=node_text, hoverinfo='text',
                marker=dict(size=20, color=color, symbol=symbol, line=dict(width=2, color='white')),
                name=node_type, showlegend=True
            )
            node_traces.append(trace)
            seen_types.add(node_type)
    
    # Create figure
    fig = go.Figure(
        data=[edge_trace, edge_label_trace] + node_traces,
        layout=go.Layout(
            title='Knowledge Graph Pattern from Cypher Query',
            titlefont_size=16, showlegend=True, hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            plot_bgcolor='#0b0f1a', paper_bgcolor='#0b0f1a',
            font=dict(color='#e5e7eb'),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=500,
            legend=dict(x=1.02, y=1, bgcolor='rgba(26, 31, 58, 0.8)', 
                       bordercolor='#2d3354', borderwidth=1)
        )
    )
    
    return fig

# =====================================================================
# RECOMMENDATION HELPER
# =====================================================================
def generate_recommendations(baseline_results: List[Dict], embedding_results: List[Dict]) -> List[Dict]:
    """Generate hotel recommendations with explanations."""
    recommendations = []
    all_hotels = {}
    
    for result in baseline_results:
        if 'hotel_name' in result:
            hotel_name = result['hotel_name']
            if hotel_name not in all_hotels:
                all_hotels[hotel_name] = {
                    'hotel_name': hotel_name,
                    'reasons': [],
                    'score': 0,
                    'data': result
                }
            
            hotel = all_hotels[hotel_name]
            
            # Check for None before formatting
            if 'star_rating' in result and result['star_rating'] is not None and result['star_rating'] >= 4:
                hotel['reasons'].append(f"⭐ High star rating ({result['star_rating']} stars)")
                hotel['score'] += result['star_rating']
            
            if 'avg_score' in result and result['avg_score'] is not None:
                hotel['reasons'].append(f"📊 Excellent reviews (avg: {result['avg_score']:.1f}/10)")
                hotel['score'] += result['avg_score']
            
            if 'cleanliness_score' in result and result['cleanliness_score'] is not None:
                hotel['reasons'].append(f"✨ Great cleanliness ({result['cleanliness_score']:.1f}/10)")
            
            if 'value_for_money_score' in result and result['value_for_money_score'] is not None:
                hotel['reasons'].append(f"💰 Good value ({result['value_for_money_score']:.1f}/10)")
    
    for result in embedding_results:
        if 'hotel_name' in result:
            hotel_name = result['hotel_name']
            if hotel_name not in all_hotels:
                all_hotels[hotel_name] = {
                    'hotel_name': hotel_name,
                    'reasons': [],
                    'score': 0,
                    'data': result
                }
            
            hotel = all_hotels[hotel_name]
            if 'similarity_score' in result and result['similarity_score'] is not None:
                hotel['reasons'].append(f"🎯 Semantically relevant (similarity: {result['similarity_score']:.2f})")
                hotel['score'] += result['similarity_score'] * 10
    
    sorted_hotels = sorted(all_hotels.values(), key=lambda x: x['score'], reverse=True)
    return sorted_hotels[:5]

# =====================================================================
# INITIALIZATION
# =====================================================================
@st.cache_resource
def initialize_system():
    neo4j = Neo4jRetriever()
    df = load_reviews_from_csv("reviews.csv", 50001, 50505)
    models = load_embedding_models()
    # Return all models, we'll select the right one later
    return neo4j, models, df

neo4j, models, df = initialize_system()
# =====================================================================
# LEFT PINNED SIDEBAR
# =====================================================================
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("Control retrieval & model behavior")

    # NEW: Retrieval Method Selection
    st.markdown("### 🔍 Retrieval Method")
    retrieval_method = st.radio(
        "Choose retrieval approach:",
        ["Both (Hybrid)", "Baseline Only (Graph)", "Embeddings Only (Semantic)"],
        index=0,
        help="Select which retrieval method(s) to use for answering queries"
    )
    
    # Map radio selection to internal flags
    if retrieval_method == "Both (Hybrid)":
        use_baseline = True
        use_embeddings = True
        method_badge = "both"
    elif retrieval_method == "Baseline Only (Graph)":
        use_baseline = True
        use_embeddings = False
        method_badge = "baseline"
    else:  # Embeddings Only
        use_baseline = False
        use_embeddings = True
        method_badge = "embeddings"
    
    # Show info about selected method
    if method_badge == "both":
        st.info("🔀 Using both graph queries and semantic search for comprehensive results")
    elif method_badge == "baseline":
        st.info("📊 Using only Neo4j Cypher queries on the knowledge graph")
    else:
        st.info("🧠 Using only semantic similarity search with embeddings")

    st.markdown("---")

    # LLM Model Selection (ONLY ONCE!)
    model = st.selectbox(
        "🤖 LLM Model",
        ["gemma-2b", "mistral-7b", "llama-3b"],
        key="llm_model_select"  # Add unique key
    )
    
    st.markdown("---")
    
    # Embedding Model Selection
    st.markdown("### 🧠 Embedding Model")
    embedding_model = st.selectbox(
        "Choose embedding model:",
        ["model1", "model2"],
        index=1,  # Default to model2
        help="Select which embedding model to use for semantic search",
        key="embedding_model_select"  # Add unique key
    )
    
    # Show info about selected embedding model
    embedding_info = {
        "model1": "all-MiniLM-L6-v2 (General Purpose)",
        "model2": "all-mpnet-base-v2 (High Quality)"
    }
    st.caption(f"📌 {embedding_info.get(embedding_model, 'Unknown model')}")
    
    st.markdown("---")
    st.markdown("### 📋 Display Options")
    
    show_kg_context = st.checkbox("📊 Show KG Context", value=True)
    show_cypher = st.checkbox("💻 Show Cypher Queries", value=True)
    show_graph_viz = st.checkbox("🕸️ Show Graph Visualization", value=True)
    show_recommendations = st.checkbox("⭐ Show Recommendations", value=True)

    st.markdown("---")

    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
# =====================================================================
# HEADER
# =====================================================================
st.markdown("""
<div class="app-header">
    <div class="app-title">✈️ AI Travel Assistant</div>
</div>
""", unsafe_allow_html=True)

# =====================================================================
# CHAT DISPLAY
# =====================================================================
if st.session_state.chat_history:
    for chat in st.session_state.chat_history:
        st.markdown(f"""
        <div class="user-bubble">
            <div class="bubble-label">You</div>
            {chat['query']}
        </div>
        """, unsafe_allow_html=True)

        # Show retrieval method badge AND model badge
        # Show retrieval method badge AND model badge
        method_used = chat.get('retrieval_method', 'both')
        model_used = chat.get('model_used', 'Unknown')
        embedding_model_used = chat.get('embedding_model_used', 'N/A')  
        
        badge_class = f"badge-{method_used}"
        method_text = {
            'both': 'Baseline + Embeddings',
            'baseline': 'Baseline Only',
            'embeddings': 'Embeddings Only'
        }.get(method_used, 'Hybrid')
                                                                            
        st.markdown(f"""
        <div class="ai-bubble">
            <div class="bubble-label">
                AI Assistant 
                <span class="retrieval-badge {badge_class}">{method_text}</span>
                <span class="retrieval-badge badge-model">{model_used}</span>
                <span class="retrieval-badge badge-model">{embedding_model_used}</span>
            </div>
            {chat['answer']}
        """, unsafe_allow_html=True)

        with st.expander("📊 Technical Details & Context", expanded=False):
            
            # Show retrieval method info
# Show retrieval method info
            st.markdown(f"**Retrieval Method Used:** {method_text}")
            st.markdown(f"**LLM Model Used:** {model_used}")
            st.markdown(f"**Embedding Model Used:** {embedding_model_used}")  # ADD THIS LINE
            st.markdown("---")
            
            # KG Context Display WITH RELATIONSHIPS
            if show_kg_context:
                st.markdown("### 📚 Knowledge Graph Retrieved Context")
                
                # Extract and display relationships from Cypher (automatically shown with KG Context)
                if "query_info" in chat and chat.get('used_baseline', True):
                    template = chat["query_info"]["template_used"]
                    
                    if template != "SEMANTIC_ONLY" and template in CYPHER_QUERIES:
                        st.markdown("#### 🔗 Relationships in Query")
                        
                        cypher_query = CYPHER_QUERIES[template]
                        relationships = extract_relationships_from_cypher(cypher_query)
                        
                        if relationships:
                            st.markdown("**Graph Pattern Used:**")
                            for rel in relationships:
                                source = f"**({rel['source_var']}:{rel['source_label']})**" if rel['source_label'] else f"**({rel['source_var']})**"
                                target = f"**({rel['target_var']}:{rel['target_label']})**" if rel['target_label'] else f"**({rel['target_var']})**"
                                
                                if rel['direction'] == 'outgoing':
                                    st.markdown(f"""
                                    <div class="relationship-box">
                                        <span class="node-label">{source}</span> 
                                        —[<span class="rel-label">:{rel['relationship']}</span>]→ 
                                        <span class="node-label">{target}</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                                elif rel['direction'] == 'incoming':
                                    st.markdown(f"""
                                    <div class="relationship-box">
                                        <span class="node-label">{target}</span> 
                                        ←[<span class="rel-label">:{rel['relationship']}</span>]— 
                                        <span class="node-label">{source}</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.markdown(f"""
                                    <div class="relationship-box">
                                        <span class="node-label">{source}</span> 
                                        —[<span class="rel-label">:{rel['relationship']}</span>]— 
                                        <span class="node-label">{target}</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            st.markdown("---")
                        else:
                            st.info("No relationships detected in query")
                    else:
                        st.info("Semantic-only query (no graph relationships)")
                elif not chat.get('used_baseline', True):
                    st.info("ℹ️ Baseline retrieval was disabled for this query")
                
                # Display retrieved data
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📹 Nodes Retrieved (Baseline)**")
                    if chat.get('used_baseline', True) and chat["baseline"]:
                        st.json(chat["baseline"])
                    elif not chat.get('used_baseline', True):
                        st.info("Baseline disabled")
                    else:
                        st.info("No baseline results")
                
                with col2:
                    st.markdown("**📹 Semantic Search Results**")
                    if chat.get('used_embeddings', True) and chat["embeddings"]:
                        st.json(chat["embeddings"])
                    elif not chat.get('used_embeddings', True):
                        st.info("Embeddings disabled")
                    else:
                        st.info("No embedding results")
            
            # Cypher Query Display
            if show_cypher and "query_info" in chat and chat.get('used_baseline', True):
                st.markdown("### 💻 Cypher Query Executed")
                query_info = chat["query_info"]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Intent & Template:**")
                    st.code(f"""
Intent: {query_info['intent']}
Template: {query_info['template_used']}
                    """, language="text")
                
                with col2:
                    st.markdown("**Extracted Parameters:**")
                    st.json(query_info['params'])
                
                if query_info['template_used'] != "SEMANTIC_ONLY":
                    st.markdown("**🔍 Full Cypher Query:**")
                    template = query_info['template_used']
                    if template in CYPHER_QUERIES:
                        st.code(CYPHER_QUERIES[template], language="cypher")
                else:
                    st.info("ℹ️ This query uses semantic search only")
            elif show_cypher and not chat.get('used_baseline', True):
                st.info("ℹ️ Cypher queries disabled for this query")
            
            # Graph Visualization
            if show_graph_viz and "query_info" in chat and chat.get('used_baseline', True):
                st.markdown("### 🕸️ Knowledge Graph Visualization")
                try:
                    template = chat["query_info"]["template_used"]
                    
                    if template != "SEMANTIC_ONLY" and template in CYPHER_QUERIES:
                        cypher_query = CYPHER_QUERIES[template]
                        fig = create_graph_from_cypher(cypher_query, chat["baseline"])
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No Cypher query to visualize (semantic-only)")
                
                except Exception as e:
                    st.error(f"Could not generate graph visualization: {e}")
            elif show_graph_viz and not chat.get('used_baseline', True):
                st.info("ℹ️ Graph visualization requires baseline retrieval")
            
            # Recommendations
            if show_recommendations and (chat["baseline"] or chat["embeddings"]):
                st.markdown("### ⭐ Recommended Hotels")
                recommendations = generate_recommendations(chat["baseline"], chat["embeddings"])
                
                if recommendations:
                    for idx, rec in enumerate(recommendations, 1):
                        st.markdown(f"**#{idx} {rec['hotel_name']}** (Score: {rec['score']:.1f})")
                        if rec['reasons']:
                            for reason in rec['reasons']:
                                st.markdown(f"  {reason}")
                        st.markdown("---")
                else:
                    st.info("No recommendations available")
            
            # Performance metrics
            st.markdown("### ⏱️ Performance Metrics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Response Time", f"{chat['time']:.2f}s")
            with col2:
                baseline_count = len(chat["baseline"]) if chat.get('used_baseline', True) else 0
                st.metric("Baseline Results", baseline_count if chat.get('used_baseline', True) else "Disabled")
            with col3:
                embedding_count = len(chat["embeddings"]) if chat["embeddings"] and chat.get('used_embeddings', True) else 0
                st.metric("Embedding Results", embedding_count if chat.get('used_embeddings', True) else "Disabled")

else:
    st.info("Ask a travel question to get started")

# =====================================================================
# INPUT AREA
# =====================================================================
st.markdown("------------------------------------------------------------------------------------------------------")
st.markdown("### 💬 Ask Your Question")

# Predefined suggestions
EXAMPLE_QUERIES = [
    "List all hotels in Rome with a star rating above 3.",
    "Show the top 3 hotels in Asia with the highest average review scores?",
    "Show all hotels in Africa that are suitable for a senior trip.",
    "Which hotels in Africa were visited by female solo travellers?",
    "Which hotels in Europe are suitable for a honeymoon?",
    "Show best hotel in Cairo that offer the best value for money.",
    "Which hotel in Barcelona have the best cleanliness scores?",
    "Which hotels in Europe are pet friendly?",
    "I need places in South America where I can stay active during my trip. Which hotels have dedicated exercise spaces?",
    "Show some reviews from the year 2020 for The Gateway Royale.",
    ]


# Text input for user's question
query = st.text_area(
    "Type your question here:",
    value=st.session_state.get('selected_query', ''),
    height=100,
    placeholder="Welcome to your favorite Travel Assistant!\nFor suggestions, double click on the example question you want below to autofill.",
)

# Show suggestions in an expander
with st.expander("💡 Example Questions (click to see suggestions)", expanded=False):
    st.markdown("**Try asking questions like:**")
    cols = st.columns(2)
    for idx, example in enumerate(EXAMPLE_QUERIES):
        with cols[idx % 2]:
            
            if st.button(f"{example}", key=f"suggestion_{idx}", use_container_width=True):
                st.session_state.selected_query = example

col1, col2 = st.columns([3, 1])
with col1:
    send = st.button("🚀 Send Message", use_container_width=True)
with col2:
    if st.button("🗑️ Clear Input", use_container_width=True):
        if 'selected_query' in st.session_state:
            del st.session_state.selected_query
        st.rerun()


# =====================================================================
# PROCESS QUERY
# =====================================================================
if send and query.strip():
    with st.spinner("✨ Thinking..."):
        start = time.time()

        try:
            # Initialize results
            baseline = []
            baseline_pipeline = None
            embedding_results = []
            
            # Run baseline if enabled
            if use_baseline:
                baseline_pipeline = run_baseline_pipeline(query, neo4j)
                baseline = baseline_pipeline["result"]
            
            # Run embeddings if enabled
            if use_embeddings:
                # Select the chosen embedding model and build index
                selected_embedder = models[embedding_model]
                embeddings = selected_embedder.encode(df["review_text"].tolist(), convert_to_numpy=True)
                index = build_faiss_index(embeddings)
                
                embedding_results = get_embedding_results_for_llm(
                    query, selected_embedder, index, df, k=5
                )
                
            # Generate answer with available results
            result = rag_answer(
                query,
                baseline,
                embedding_results,
                model_name=model
            )

            # Store in chat history with retrieval method info

            chat_entry = {
                "query": query,
                "answer": result["answer"],
                "time": time.time() - start,
                "baseline": baseline,
                "embeddings": embedding_results,
                "retrieval_method": method_badge,
                "used_baseline": use_baseline,
                "used_embeddings": use_embeddings,
                "model_used": model,
                "embedding_model_used": embedding_model if use_embeddings else "N/A"  # ADD THIS LINE
            }
            # Add query info only if baseline was used
            if use_baseline and baseline_pipeline:
                chat_entry["query_info"] = {
                    "intent": baseline_pipeline["intent"],
                    "entities": baseline_pipeline["entities"],
                    "params": baseline_pipeline["params"],
                    "template_used": baseline_pipeline["template_used"]
                }
            
            st.session_state.chat_history.append(chat_entry)

            # Clear the selected query after successful send
            if 'selected_query' in st.session_state:
                del st.session_state.selected_query
            st.rerun()
        
        except Exception as e:
            # Create a chat entry showing the agent couldn't find an answer
            no_answer_entry = {
                "query": query,
                "answer": "I apologize, but the knowledge base doesn't contain information to answer your question. Please try rephrasing your question or check the example questions below for supported queries.",
                "time": time.time() - start,
                "baseline": [],
                "embeddings": [],
                "retrieval_method": method_badge,
                "used_baseline": use_baseline,
                "used_embeddings": use_embeddings,
                "model_used": model,
                "embedding_model_used": embedding_model if use_embeddings else "N/A"
            }
            
            st.session_state.chat_history.append(no_answer_entry)
            
            # Clear the selected query
            if 'selected_query' in st.session_state:
                del st.session_state.selected_query
            st.rerun()

elif send and not query.strip():
    st.warning("⚠️ Please enter a question before sending!")