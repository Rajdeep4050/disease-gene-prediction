import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import networkx as nx
import random

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Disease Gene Prediction",
    layout="wide"
)

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("data/processed/final_dataset.csv")

@st.cache_resource
def load_model():
    return joblib.load("models/random_forest.pkl")

df = load_data()
model = load_model()

# =========================
# BUILD GRAPH (FIXED POSITION)
# =========================
@st.cache_resource
def build_graph(df):
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(row["gene1"], row["gene2"])
    return G

G = build_graph(df)

# =========================
# GET TOP GENES (UI FRIENDLY)
# =========================
@st.cache_data
def get_top_genes(df, n=100):
    degree_counts = pd.concat([df['gene1'], df['gene2']]).value_counts()
    return sorted(list(degree_counts.head(n).index))

# =========================
# FEATURE COMPUTATION
# =========================
def compute_features(g1, g2, G):

    degree1 = G.degree(g1) if g1 in G else 0
    degree2 = G.degree(g2) if g2 in G else 0

    neighbors1 = set(G.neighbors(g1)) if g1 in G else set()
    neighbors2 = set(G.neighbors(g2)) if g2 in G else set()

    common = len(neighbors1 & neighbors2)

    union = len(neighbors1 | neighbors2)
    jaccard = common / union if union != 0 else 0

    return degree1, degree2, common, jaccard


# =========================
# SIDEBAR NAVIGATION
# =========================
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Graph Construction",
        "Label Engineering",
        "Feature Engineering",
        "Model Performance",
        "Prediction Playground"
    ]
)

# =========================
# 1. OVERVIEW
# =========================
if page == "Overview":

    st.title("🧬 Disease-Gene Prediction")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Edges", len(df))
    col2.metric("Positive Cases", int(df['edge_label'].sum()))
    col3.metric("Positive Ratio", f"{df['edge_label'].mean():.2f}")

    st.markdown("""
    STRING → ~472K interactions  
    Open Targets → ~2.7K disease genes  
    
    Key Insight: Graph structure > raw score
    """)

# =========================
# 2. GRAPH CONSTRUCTION
# =========================
elif page == "Graph Construction":

    st.title("🕸️ Graph Construction")

    sample_size = st.slider("Sample Size", 100, 2000, 500)

    sample_df = df.sample(sample_size)

    G_sample = nx.Graph()
    for _, row in sample_df.iterrows():
        G_sample.add_edge(row["gene1"], row["gene2"])

    fig, ax = plt.subplots()
    nx.draw(G_sample, node_size=10, ax=ax)
    st.pyplot(fig)

# =========================
# 3. LABEL ENGINEERING
# =========================
elif page == "Label Engineering":

    st.title("🏷️ Label Engineering")

    label_counts = df['edge_label'].value_counts()

    fig, ax = plt.subplots()
    label_counts.plot(kind='bar', ax=ax)
    st.pyplot(fig)

    st.markdown("""
    AND logic → strong signal  
    OR logic → noisy labels
    """)

# =========================
# 4. FEATURE ENGINEERING
# =========================
elif page == "Feature Engineering":

    st.title("⚙️ Feature Engineering")

    feature = st.selectbox(
        "Select Feature",
        [
            "degree_gene1",
            "degree_gene2",
            "common_neighbors",
            "jaccard_similarity"
        ]
    )

    fig, ax = plt.subplots()
    df[feature].hist(bins=50, ax=ax)
    st.pyplot(fig)

# =========================
# 5. MODEL PERFORMANCE
# =========================
elif page == "Model Performance":

    st.title("🤖 Model Performance")

    models = ["Logistic Regression", "Random Forest"]
    recall = [0.38, 0.76]

    fig, ax = plt.subplots()
    ax.bar(models, recall)
    st.pyplot(fig)

    st.markdown("""
    Accuracy: ~0.84  
    Recall: ~0.76  
    Precision: ~0.50  
    """)
# =========================
# 6. PREDICTION PLAYGROUND
# =========================
elif page == "Prediction Playground":

    st.title("🔮 Disease Interaction Predictor")

    st.markdown("""
    Select two genes to check whether their interaction is related to disease.

    """)


    mode = st.radio("Select Gene Mode", ["Top Genes", "All Genes"])

    if mode == "Top Genes":
        gene_list = get_top_genes(df, 100)
    else:
        gene_list = sorted(list(set(df['gene1']).union(set(df['gene2']))))

    col1, col2 = st.columns(2)

    if "gene1" not in st.session_state:
        st.session_state.gene1 = gene_list[0]

    if "gene2" not in st.session_state:
        st.session_state.gene2 = gene_list[1]

    with col1:
        gene1 = st.selectbox(
            "Select Gene 1",
            gene_list,
            index=gene_list.index(st.session_state.gene1),
            key="gene1_select"
        )

    with col2:
        gene2 = st.selectbox(
            "Select Gene 2",
            gene_list,
            index=gene_list.index(st.session_state.gene2),
            key="gene2_select"
        )

    # Sync session state with UI
    st.session_state.gene1 = gene1
    st.session_state.gene2 = gene2


    if st.button("🎲 Try Random Pair"):
        st.session_state.gene1 = random.choice(gene_list)
        st.session_state.gene2 = random.choice(gene_list)

    if st.button("Predict Interaction"):

        # Compute features
        degree1, degree2, common, jaccard = compute_features(gene1, gene2, G)

        # Create input
        X = pd.DataFrame([{
            "degree_gene1": degree1,
            "degree_gene2": degree2,
            "common_neighbors": common,
            "jaccard_similarity": jaccard
        }])

        # Predict
        prob = model.predict_proba(X)[0][1]

        st.write(f"🔍 Raw probability: {prob:.3f}")
        
        # 🔥 THRESHOLD LOGIC (IMPORTANT FIX)
        threshold_strong = 0.90
        threshold_moderate = 0.70

        if prob >= threshold_strong:
            pred = "strong"
        elif prob >= threshold_moderate:
            pred = "moderate"
        else:
            pred = "weak"
        prob_percent = prob * 100

        # =========================
        # HUMAN-FRIENDLY RESULT
        # =========================
        prob_percent = prob * 100

        if pred == "strong":
            st.success(f"🟢 Strong disease-related interaction ({prob_percent:.1f}%)")

        elif pred == "moderate":
            st.warning(f"🟡 Moderate disease interaction ({prob_percent:.1f}%)")

        else:
            st.error(f"🔴 Weak or no disease relationship ({prob_percent:.1f}%)")
        # =========================
        # EXPLANATION
        # =========================
        st.markdown("### 🧠 Why this prediction?")

        explanation = []

        if common == 0:
            explanation.append("• These genes do NOT share common connections")
        else:
            explanation.append(f"• They share {common} common connections")

        if jaccard < 0.1:
            explanation.append("• Their similarity is very low")
        elif jaccard < 0.3:
            explanation.append("• Their similarity is moderate")
        else:
            explanation.append("• Their similarity is high")

        if degree1 < 5:
            explanation.append("• Gene 1 has few interactions")
        else:
            explanation.append("• Gene 1 is highly connected")

        if degree2 < 5:
            explanation.append("• Gene 2 has few interactions")
        else:
            explanation.append("• Gene 2 is highly connected")

        for line in explanation:
            st.write(line)

        # =========================
        # TECHNICAL DETAILS
        # =========================
        # with st.expander("🔬 Show Technical Details"):
        #     st.write(X)

        st.markdown("### 📊 Confidence Level")
        st.progress(float(prob))
        st.write(f"Confidence: {prob * 100:.1f}%")