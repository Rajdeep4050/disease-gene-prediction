import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import networkx as nx
import random
import plotly.express as px

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

    st.title("🧬 Disease Gene Interaction Prediction")

    st.markdown("""
    ### 🧠 Problem We Are Solving

    In our body, genes interact like people in a social network.

    👉 Some genes are associated with diseases  
    👉 Some genes interact with each other  

    Our goal:

    > **Can we predict if an interaction between two genes is related to disease?**
    """)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Interactions", f"{len(df):,}")
    col2.metric("Disease Interactions", int(df['edge_label'].sum()))
    col3.metric("Positive Ratio", f"{df['edge_label'].mean():.2f}")

    st.markdown("---")

    st.markdown("""
    ### 📊 Data Sources

    - STRING Database → Gene interactions (~472K)
    - Open Targets → Disease-associated genes (~2.7K)

    ### ⚙️ What This App Shows

    - How gene networks are built  
    - How we label disease interactions  
    - What features we use  
    - How well the model performs  
    - Interactive prediction for any gene pair  
    """)

    # st.info("👉 Use the sidebar to explore step-by-step how the model works.")
    



    st.markdown("## 📂 Understanding the Dataset")

    st.markdown("""
    We combine multiple biological datasets to build our model.

    ---

    ### 🧬 1. Gene Labels (gene_labels.csv)

    This dataset tells us which genes are associated with diseases.

    | Column | Meaning |
    |--------|--------|
    | gene_id | Unique gene identifier |
    | label | 1 = disease gene, 0 = normal gene |

    👉 Example:
    """)

    st.code("""
    ENSG00000204480 → 1 (disease-related)
    ENSG00000085063 → 1
    """)

    st.markdown("""
    ---

    ### 🔗 2. Gene Interactions (interactions.csv)

    This dataset tells us which genes interact with each other.

    | Column | Meaning |
    |--------|--------|
    | protein1, protein2 | Interacting proteins |
    | combined_score | Strength of interaction (0–1000) |

    👉 Example:
    """)

    st.code("""
    Gene A ↔ Gene B → score = 825 (strong interaction)
    """)

    st.markdown("""
    ---

    ### ⚙️ 3. Final Dataset (final_dataset.csv)

    This is the dataset used for training the model.

    Each row represents an interaction between two genes.

    | Column | Meaning |
    |--------|--------|
    | gene1, gene2 | Pair of genes |
    | edge_label | 1 if both genes are disease-related |
    | degree | Connectivity of genes |
    | common_neighbors | Shared connections |
    | jaccard_similarity | Similarity between genes |

    👉 Example:
    """)

    st.code("""
    gene1 = ENSG00000004059
    gene2 = ENSG00000168374

    common_neighbors = 3
    jaccard_similarity = 0.09
    edge_label = 0 (not strong disease interaction)
    """)

    st.markdown("""
    ---

    ### 🧠 Key Insight

    We transformed raw biological data into a **graph-based machine learning problem**:

    👉 From:
    - Who is a disease gene?
    - Who interacts?

    👉 To:
    - Can we predict disease-related interactions?

    ---
    """)

# =========================
# 2. GRAPH CONSTRUCTION
# =========================
elif page == "Graph Construction":

    import plotly.express as px

    st.title("🕸️ Understanding Gene Interaction Network")

    st.markdown("""
    Instead of visualizing the entire network (which is too large and complex),
    we analyze **key structural properties** of the graph.

    👉 This helps us understand how genes are connected.
    """)

    st.markdown("---")

    # =========================
    # 1. DEGREE DISTRIBUTION
    # =========================
    st.markdown("## 📊 Gene Connectivity Distribution")

    all_nodes = pd.concat([df["gene1"], df["gene2"]])
    degree_counts = all_nodes.value_counts()

    degree_df = degree_counts.reset_index()
    degree_df.columns = ["Gene", "Degree"]

    fig = px.histogram(
        degree_df,
        x="Degree",
        nbins=50,
        title="Distribution of Gene Connectivity",
        color_discrete_sequence=["#4C78A8"]
    )

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Number of Connections (Degree)",
        yaxis_title="Number of Genes"
    )

    fig.update_traces(
        hovertemplate="Degree: %{x}<br>Count: %{y}"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    ### 🧠 Insight

    - Most genes have **few connections**
    - A small number of genes have **many connections (hub genes)**

    👉 This is a common pattern in real-world networks.
    """)

    st.markdown("---")

    # =========================
    # 2. TOP HUB GENES
    # =========================
    st.markdown("## ⭐ Top Connected Genes (Hubs)")

    top_genes = degree_counts.head(10).reset_index()
    top_genes.columns = ["Gene", "Connections"]

    # Reverse for better display
    top_genes = top_genes.iloc[::-1]

    fig2 = px.bar(
        top_genes,
        x="Connections",
        y="Gene",
        orientation="h",
        title="Top 10 Most Connected Genes",
        color="Connections",
        color_continuous_scale="Oranges"
    )

    fig2.update_layout(
        template="plotly_white",
        xaxis_title="Number of Connections",
        yaxis_title="Gene"
    )

    fig2.update_traces(
        hovertemplate="Gene: %{y}<br>Connections: %{x}"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
    ### 🧠 Insight

    - These genes act as **hubs** in the network  
    - Highly connected genes are often biologically important  
    """)

    st.markdown("---")

    # =========================
    # 3. LOCAL NEIGHBOR EXPLORATION
    # =========================
    st.markdown("## 🔍 Explore a Gene's Neighborhood")

    gene_list = sorted(list(set(df["gene1"]).union(set(df["gene2"]))))

    selected_gene = st.selectbox("Select a gene", gene_list)

    neighbors = df[
        (df["gene1"] == selected_gene) |
        (df["gene2"] == selected_gene)
    ]

    st.write(f"### Connections: {len(neighbors)}")

    st.dataframe(neighbors.head(10))

    st.markdown("""
    ### 🧠 Insight

    - This shows how a gene connects locally  
    - These relationships are used to compute features like:
        - Degree  
        - Common neighbors  
        - Similarity  
    """)

    st.markdown("---")

    # =========================
    # 4. FINAL TAKEAWAY
    # =========================
    st.markdown("## 🧠 Why This Matters")

    st.markdown("""
    Instead of visualizing the full graph, we extract meaningful patterns:

    - Degree → importance of a gene  
    - Shared neighbors → relationship strength  
    - Similarity → how closely genes are related  

    👉 These patterns are used to build features for the model.
    """)

    st.success("✔ Graph structure is converted into meaningful features for prediction")
    
# =========================
# 3. LABEL ENGINEERING
# =========================
elif page == "Label Engineering":

    st.title("🏷️ How We Define Disease Interactions")

    # =========================
    # 1. CONCEPT
    # =========================
    st.markdown("""
    In this step, we define what a **disease-related interaction** means.

    Each gene has a label:
    - 1 → Disease-associated  
    - 0 → Not disease-associated  

    We now assign labels to **interactions between genes (edges)**.
    """)

    st.markdown("---")

    # =========================
    # 2. ALL CASES
    # =========================
    st.markdown("## 🧩 All Possible Interaction Cases")

    cases_df = pd.DataFrame({
        "Gene 1": [0, 1, 0, 1],
        "Gene 2": [0, 0, 1, 1],
        "Edge Label": [0, 0, 0, 1],
        "Meaning": [
            "Both normal → no disease relevance",
            "One disease gene → weak signal",
            "One disease gene → weak signal",
            "Both disease genes → strong interaction"
        ]
    })

    st.dataframe(cases_df)

    st.markdown("""
    👉 Only when **both genes are disease-related (1,1)**  
    do we label it as a strong interaction.
    """)

    st.markdown("---")

    # =========================
    # 3. AND vs OR
    # =========================
    st.markdown("## ⚖️ Why AND and not OR?")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ❌ OR Logic")
        st.code("""
edge_label = (gene1 == 1 OR gene2 == 1)
""")
        st.markdown("""
        - Includes weak interactions  
        - Adds noise  
        - Too many false positives  
        """)

    with col2:
        st.markdown("### ✅ AND Logic")
        st.code("""
edge_label = (gene1 == 1 AND gene2 == 1)
""")
        st.markdown("""
        - Keeps only strong signals  
        - Reduces noise  
        - Improves model quality  
        """)

    st.success("✔ We use AND logic to ensure high-quality learning")

    st.markdown("---")

    # =========================
    # 4. FILTERING
    # =========================
    st.markdown("## 🔍 Data Filtering Strategy")

    st.markdown("""
    We remove interactions where both genes are normal:

    👉 (0,0) → removed  

    Why?
    - These interactions are not informative  
    - They dominate the dataset  
    - They reduce learning signal  

    👉 This helps the model focus on meaningful interactions.
    """)

    st.markdown("---")

    # =========================
    # 5. IMPROVED GRAPH
    # =========================
    st.markdown("## 📊 Final Label Distribution")

    label_counts = df["edge_label"].value_counts().sort_index()

    labels = ["Weak (0)", "Strong (1)"]
    values = [label_counts.get(0, 0), label_counts.get(1, 0)]

    total = sum(values)
    percentages = [(v / total) * 100 for v in values]

    fig, ax = plt.subplots(figsize=(6, 4))

    # Colors
    colors = ["#6baed6", "#fd8d3c"]

    bars = ax.bar(labels, values, color=colors)

    # Add values + percentages on top
    for bar, val, pct in zip(bars, values, percentages):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:,}\n({pct:.1f}%)",
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )

    # Styling
    ax.set_title("Distribution of Disease Interactions", fontsize=14)
    ax.set_ylabel("Number of Interactions")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    st.pyplot(fig)

    st.markdown("---")

    # =========================
    # 6. INSIGHT
    # =========================
    st.markdown("## 🧠 Key Insights")

    st.markdown(f"""
    - Total interactions: **{total:,}**  
    - Strong interactions: **{values[1]:,} ({percentages[1]:.1f}%)**  
    - Weak interactions: **{values[0]:,} ({percentages[0]:.1f}%)**

    👉 The dataset is **highly imbalanced**

    - Only a small percentage of interactions are truly disease-related  
    - Most interactions are weak  

    ### ⚠️ Why This Matters

    - The model may become biased toward predicting weak interactions  
    - Important disease interactions might be missed  

    ### ✅ Solution

    We handle this by:
    - Using `class_weight="balanced"`  
    - Focusing on **recall instead of accuracy**
    """)



# =========================
# 4. FEATURE ENGINEERING
# =========================
elif page == "Feature Engineering":

    st.title("⚙️ Understanding Feature Engineering")

    # =========================
    # 1. INTRO
    # =========================
    st.markdown("""
    We convert the gene interaction network into numerical features that the model can learn from.

    These features capture **how strongly two genes are related**.
    """)

    st.markdown("---")

    # =========================
    # 2. FEATURE EXPLANATION
    # =========================
    st.markdown("## 🔑 Features Used")

    st.markdown("""
    • **Degree** → How connected a gene is  
    • **Common Neighbors** → Shared connections  
    • **Jaccard Similarity** → Similarity of connections  
    """)


    st.markdown("---")

    # =========================
    # 3. ADD COMBINED DEGREE (IMPORTANT)
    # =========================
    df["avg_degree"] = (df["degree_gene1"] + df["degree_gene2"]) / 2

    feature_map = {
        "avg_degree": "Average Degree",
        "degree_gene1": "Degree (Gene 1)",
        "degree_gene2": "Degree (Gene 2)",
        "common_neighbors": "Common Neighbors",
        "jaccard_similarity": "Jaccard Similarity"
    }

    feature = st.selectbox("Select Feature", list(feature_map.keys()))
    display_name = feature_map[feature]

    st.markdown("---")

    # =========================
    # 4. STRONG vs WEAK COMPARISON
    # =========================
    st.markdown("## 🔍 Strong vs Weak Interaction Comparison")

    weak_sample = df[df["edge_label"] == 0].sample(1)
    strong_sample = df[df["edge_label"] == 1].sample(1)

    comparison_df = pd.DataFrame({
        "Feature": [
            "Degree (Gene 1)",
            "Degree (Gene 2)",
            "Common Neighbors",
            "Jaccard Similarity"
        ],
        "Weak Interaction": [
            weak_sample["degree_gene1"].values[0],
            weak_sample["degree_gene2"].values[0],
            weak_sample["common_neighbors"].values[0],
            weak_sample["jaccard_similarity"].values[0]
        ],
        "Strong Interaction": [
            strong_sample["degree_gene1"].values[0],
            strong_sample["degree_gene2"].values[0],
            strong_sample["common_neighbors"].values[0],
            strong_sample["jaccard_similarity"].values[0]
        ]
    })

    st.dataframe(comparison_df)



    # =========================
    # 5. IMPROVED DISTRIBUTION GRAPH
    # =========================
    st.markdown(f"## 📊 Distribution of {display_name}")

    weak_vals = df[df["edge_label"] == 0][feature]
    strong_vals = df[df["edge_label"] == 1][feature]

    fig, ax = plt.subplots(figsize=(6, 4))

    # Histograms
    ax.hist(weak_vals, bins=40, alpha=0.6, label="Weak", color="#6baed6")
    ax.hist(strong_vals, bins=40, alpha=0.6, label="Strong", color="#fd8d3c")

    # Mean lines
    weak_mean = weak_vals.mean()
    strong_mean = strong_vals.mean()

    ax.axvline(weak_mean, linestyle='--', color="#6baed6", label="Weak Mean")
    ax.axvline(strong_mean, linestyle='--', color="#fd8d3c", label="Strong Mean")

    # Styling
    ax.set_title(f"{display_name} Distribution")
    ax.set_xlabel(display_name)
    ax.set_ylabel("Frequency")
    ax.legend()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    st.pyplot(fig)

    st.markdown("---")

    # =========================
    # 6. SUMMARY STATS (IMPORTANT)
    # =========================
    st.markdown("## 📊 Summary Statistics")

    summary_df = pd.DataFrame({
        "Metric": ["Mean", "Median"],
        "Weak": [weak_vals.mean(), weak_vals.median()],
        "Strong": [strong_vals.mean(), strong_vals.median()]
    })

    st.dataframe(summary_df)

    st.markdown("---")

    # =========================
    # 7. INSIGHT (VERY IMPORTANT)
    # =========================
    st.markdown("## 🧠 Key Insight")

    st.markdown(f"""
    - Weak Mean: **{weak_mean:.2f}**  
    - Strong Mean: **{strong_mean:.2f}**

    👉 Even if the distributions look similar,  
    the model learns from **subtle statistical differences**.

    👉 Features like:
    - Common neighbors  
    - Similarity  

    provide strong signals for predicting disease interactions.
    """)

    st.success("✔ These features allow the model to learn complex graph relationships")


# =========================
# 5. MODEL PERFORMANCE
# =========================
elif page == "Model Performance":

    st.title("🤖 Model Comparison & Evolution")

    st.markdown("""
    This section explains how the model improved step-by-step and why certain decisions were made.
    """)

    # =========================
    # 1. MODEL COMPARISON
    # =========================
    st.markdown("## 🔍 Step 1: Model Comparison")

    comparison_df = pd.DataFrame({
        "Metric": ["Precision", "Recall", "F1 Score", "Accuracy"],
        "Logistic Regression": [0.77, 0.36, 0.49, 0.88],
        "Random Forest": [0.49, 0.76, 0.60, 0.83]
    })

    st.dataframe(comparison_df)

    # Bar chart for comparison
    fig1, ax1 = plt.subplots()

    comparison_df.set_index("Metric").plot(kind="bar", ax=ax1)
    ax1.set_title("Logistic Regression vs Random Forest")

    st.pyplot(fig1)

    st.markdown("""
    ### 🧠 Interpretation

    - Logistic Regression:
        - High precision ✔
        - Low recall ❌ (misses disease interactions)

    - Random Forest:
        - Lower precision
        - High recall ✔ (captures disease interactions)

    👉 Since missing disease interactions is costly, we prioritize **recall**.
    """)

    st.markdown("---")

    # =========================
    # 2. MODEL EVOLUTION
    # =========================
    st.markdown("## 🚀 Step 2: Model Evolution")

    evolution_df = pd.DataFrame({
        "Stage": ["Initial Model", "Feature Engineering", "Final Model"],
        "Precision": [1.00, 0.67, 0.49],
        "Recall": [0.97, 0.42, 0.76],
        "F1 Score": [0.98, 0.52, 0.60]
    })

    st.dataframe(evolution_df)

    # 🔥 NEW IMPROVED GRAPH (GROUPED BAR)
    fig2, ax2 = plt.subplots()

    metrics = ["Precision", "Recall", "F1 Score"]
    stages = evolution_df["Stage"]

    precision = evolution_df["Precision"]
    recall = evolution_df["Recall"]
    f1 = evolution_df["F1 Score"]

    x = range(len(stages))

    ax2.bar([i - 0.2 for i in x], precision, width=0.2, label="Precision")
    ax2.bar(x, recall, width=0.2, label="Recall")
    ax2.bar([i + 0.2 for i in x], f1, width=0.2, label="F1 Score")

    ax2.set_xticks(x)
    ax2.set_xticklabels(stages)
    ax2.set_title("Model Performance Evolution")
    ax2.set_ylabel("Score")
    ax2.legend()

    st.pyplot(fig2)

    # =========================
    # 3. EXPLANATION (MOST IMPORTANT)
    # =========================
    st.markdown("## 🧠 What Changed Across Experiments")

    st.markdown("""
    ### 🧪 Initial Model
    - Very high performance (Precision: 1.00, Recall: 0.97)
    - Relied heavily on external signal (combined_score)
    - ❌ Not truly learning graph patterns (misleading performance)

    ---
    ### 🔧 Feature Engineering Phase
    - Removed combined_score
    - Introduced graph-based features:
        - Degree
        - Common neighbors
        - Jaccard similarity

    - Result:
        - Precision dropped (0.67)
        - Recall dropped significantly (0.42)

    👉 Model became realistic but **too conservative**

    ---
    ### 🚀 Final Model
    - Added class balancing (`class_weight="balanced"`)

    - Result:
        - Recall improved: **0.42 → 0.76 🔥**
        - Precision decreased: 0.49

    👉 Model now captures more disease interactions

    ---
    ### ⚠️ Trade-off

    - Precision ↓ (more false positives)
    - Recall ↑ (fewer missed disease interactions)

    👉 This is acceptable because:
    Missing a disease-related interaction is more harmful than a false alarm.
    """)

    st.success("✔ Final Decision: Random Forest optimized for high recall")

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

        # if degree1 < 5:
        #     explanation.append("• Gene 1 has few interactions")
        # else:
        #     explanation.append("• Gene 1 is highly connected")

        # if degree2 < 5:
        #     explanation.append("• Gene 2 has few interactions")
        # else:
        #     explanation.append("• Gene 2 is highly connected")

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