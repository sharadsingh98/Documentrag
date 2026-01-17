
import streamlit as st
from pathlib import Path
import sys
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder
from relationship_detector import RelationshipDetector



# ---------------------------
# Global Page Configuration
# ---------------------------
st.set_page_config(
    page_title="🤖 RAG Analytics Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Global CSS / Theme
# ---------------------------
st.markdown(
    """
    <style>
    /* Global */
    .main {
        background: linear-gradient(180deg, #f9fafb 0%, #ffffff 40%, #f3f4f6 100%);
        color: #111827;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* Override light headings - make them dark for better visibility */
    h1, h2, h3, h4, h5, h6 {
        color: #111827 !important;
        letter-spacing: 0.02em;
        font-weight: 600;
    }
    
    /* Top app bar */
    .top-bar {
        background: rgba(255,255,255,0.98);
        border-radius: 16px;
        padding: 0.85rem 1.4rem;
        margin-bottom: 1.2rem;
        border: 1px solid rgba(209,213,219,0.9);
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 14px 35px rgba(148,163,184,0.35);
    }
    .top-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #111827;
    }
    .top-subtitle {
        font-size: 0.8rem;
        color: #6b7280;
    }
    .top-pill {
        background: linear-gradient(135deg,#22c55e,#16a34a);
        color: #f9fafb;
    }
    .top-chip {
        border: 1px solid rgba(148,163,184,0.55);
        color: #374151;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg,#4f46e5,#6366f1);
        color: #f9fafb;
        font-weight: 600;
        border-radius: 999px;
        border: 0;
        padding: 0.55rem 0.8rem;
        box-shadow: 0 10px 25px rgba(79,70,229,0.45);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg,#6366f1,#818cf8);
        transform: translateY(-1px);
        box-shadow: 0 18px 40px rgba(79,70,229,0.65);
    }

    /* Metric cards */
    .metric-card {
        background: #ffffff;
        padding: 18px 18px 16px 18px;
        border-radius: 16px;
        border: 1px solid rgba(209,213,219,0.9);
        box-shadow: 0 12px 30px rgba(148,163,184,0.35);
        text-align: left;
        color: #111827;
    }
    .metric-card h4 {
        color: #6b7280 !important;
    }
    .metric-card .metric-value {
        color: #111827;
    }
    .metric-card .metric-sub {
        color: #9ca3af;
    }
    .metric-pill {
        background: rgba(239,246,255,0.9);
        border: 1px solid rgba(59,130,246,0.5);
        color: #1d4ed8;
    }

    /* File badges */
    .file-badge {
        border: 1px solid rgba(209,213,219,0.9);
        color: #111827;
    }
    .badge-primary { background: rgba(219,234,254,0.95); }
    .badge-success { background: rgba(220,252,231,0.95); }
    .badge-warning { background: rgba(254,243,199,0.95); }
    .badge-danger  { background: rgba(254,226,226,0.95); }

    .dq-pass { color: #16a34a; }
    .dq-fail { color: #dc2626; }
    .dq-warning { color: #d97706; }

    /* Dark heading class */
    .dark-heading {
        color: #1f2937 !important;
        font-weight: 600;
    }

    /* Expander tweak */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #111827 !important;
    }
    
    /* Ensure all text in expanders is visible */
    .streamlit-expanderContent {
        color: #111827;
    }
    </style>
    """,
    unsafe_allow_html=True,
)




# ---------------------------
# Session State
# ---------------------------
def init_session_state():
    defaults = {
        "rag_system": None,
        "initialized": False,
        "history": [],
        "uploaded_files": {},   # Store multiple CSV files
        "relationships": [],    # Store detected relationships
        "dq_results": {},       # Store DQ results for each file
        "analytics_data": {
            "total_queries": 0,
            "avg_response_time": 0,
            "query_categories": [],
            "response_times": [],
            "timestamps": [],
        },
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource
def initialize_rag():
    try:
        llm = Config.get_llm()
        doc_processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP
        )
        vector_store = VectorStore()
        urls = Config.DEFAULT_URLS
        documents = doc_processor.process_urls(urls)
        vector_store.create_vectorstore(documents)
        graph_builder = GraphBuilder(
            retriever=vector_store.get_retriever(), llm=llm
        )
        graph_builder.build()
        return graph_builder, len(documents)
    except Exception as e:
        st.error(f"Failed to initialize: {str(e)}")
        return None, 0


# ---------------------------
# Relationship Detection
# ---------------------------


def perform_join_analysis(df1, df2, join_col1, join_col2=None):
    try:
        if join_col2 and join_col1 != join_col2:
            inner_join = pd.merge(
                df1,
                df2,
                left_on=join_col1,
                right_on=join_col2,
                how="inner",
                suffixes=("_1", "_2"),
            )
            left_join = pd.merge(
                df1,
                df2,
                left_on=join_col1,
                right_on=join_col2,
                how="left",
                suffixes=("_1", "_2"),
            )
        else:
            inner_join = pd.merge(
                df1, df2, on=join_col1, how="inner", suffixes=("_1", "_2")
            )
            left_join = pd.merge(
                df1, df2, on=join_col1, how="left", suffixes=("_1", "_2")
            )

        stats = {
            "total_records_file1": len(df1),
            "total_records_file2": len(df2),
            "inner_join_records": len(inner_join),
            "left_join_records": len(left_join),
            "unmatched_records": len(left_join) - len(inner_join),
            "match_rate": (len(inner_join) / len(df1) * 100) if len(df1) > 0 else 0,
        }
        return inner_join, stats
    except Exception as e:
        st.error(f"Join analysis error: {str(e)}")
        return None, None


# ---------------------------
# Data Quality
# ---------------------------
def perform_dq_checks(df, file_name):
    dq_report = {
        "file_name": file_name,
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "checks": [],
        "overall_score": 0,
    }

    passed_checks = 0
    total_checks = 0

    # Missing values
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df) * 100).round(2)
    total_checks += 1
    if missing_counts.sum() == 0:
        passed_checks += 1
        status = "PASS"
        details = "No missing values"
    else:
        status = "WARNING" if missing_counts.sum() < len(df) * 0.05 else "FAIL"
        details = missing_pct[missing_pct > 0].to_dict()

    dq_report["checks"].append(
        {"name": "Missing Values", "status": status, "details": details}
    )

    # Duplicates
    duplicate_count = df.duplicated().sum()
    total_checks += 1
    if duplicate_count == 0:
        passed_checks += 1
        status = "PASS"
        details = "No duplicates"
    else:
        status = "FAIL"
        details = f"{duplicate_count} duplicates ({duplicate_count/len(df)*100:.1f}%)"

    dq_report["checks"].append(
        {"name": "Duplicates", "status": status, "details": details}
    )

    # Data types
    type_info = df.dtypes.value_counts().to_dict()
    total_checks += 1
    passed_checks += 1
    dq_report["checks"].append(
        {
            "name": "Data Types",
            "status": "PASS",
            "details": {str(k): v for k, v in type_info.items()},
        }
    )

    # Outliers
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    outlier_info = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = (
            (df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))
        ).sum()
        if outliers > 0:
            outlier_info[col] = outliers

    total_checks += 1
    if not outlier_info:
        passed_checks += 1
        status = "PASS"
    else:
        status = "WARNING"

    dq_report["checks"].append(
        {"name": "Outliers", "status": status, "details": outlier_info or "None detected"}
    )

    dq_report["overall_score"] = (passed_checks / total_checks * 100) if total_checks else 0
    return dq_report


# ---------------------------
# Relationship Network Chart
# ---------------------------
def display_relationship_network(relationships, files_dict):
    if not relationships:
        st.info("No relationships detected between files")
        return

    nodes = list(files_dict.keys())
    edges = [(r["file1"], r["file2"]) for r in relationships]

    n = len(nodes)
    angles = [2 * np.pi * i / n for i in range(n)]
    node_x = [np.cos(angle) * 2 for angle in angles]
    node_y = [np.sin(angle) * 2 for angle in angles]

    edge_traces = []
    for edge in edges:
        idx1 = nodes.index(edge[0])
        idx2 = nodes.index(edge[1])
        edge_trace = go.Scatter(
            x=[node_x[idx1], node_x[idx2], None],
            y=[node_y[idx1], node_y[idx2], None],
            mode="lines",
            line=dict(width=2, color="#64748b"),
            hoverinfo="none",
            showlegend=False,
        )
        edge_traces.append(edge_trace)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        marker=dict(
            size=46,
            color=["#6366f1", "#22c55e", "#f97316", "#ec4899", "#8b5cf6", "#0ea5e9"][:n],
            line=dict(width=3, color="#020617"),
            opacity=0.95,
        ),
        text=nodes,
        textposition="top center",
        textfont=dict(size=12, color="#e5e7eb", family="Inter, system-ui"),
        hoverinfo="text",
        hovertext=[
            f"<b>{node}</b><br>Drag to reposition in the network"
            for node in nodes
        ],
        hoverlabel=dict(bgcolor="#020617", font_size=11, font_family="Inter"),
        showlegend=False,
    )

    fig = go.Figure(data=edge_traces + [node_trace])

    fig.update_layout(
        title={
            "text": "File Relationship Graph<br><sub>Drag nodes to explore the data model</sub>",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 18, "color": "#e5e7eb"},
        },
        showlegend=False,
        hovermode="closest",
        xaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False, range=[-3, 3]
        ),
        yaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False, range=[-3, 3]
        ),
        plot_bgcolor="#020617",
        paper_bgcolor="rgba(0,0,0,0)",
        height=520,
        margin=dict(l=30, r=30, t=80, b=30),
        dragmode="pan",
    )

    config = {
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        "toImageButtonOptions": {
            "format": "png",
            "filename": "relationship_network",
            "height": 520,
            "width": 820,
            "scale": 2,
        },
    }

    st.plotly_chart(fig, use_container_width=True, config=config)
    st.caption("💡 Drag nodes, zoom, and export the graph as an image from the toolbar.")


# ---------------------------
# Analytics
# ---------------------------
def update_analytics(question, response_time):
    st.session_state.analytics_data["total_queries"] += 1
    st.session_state.analytics_data["response_times"].append(response_time)
    st.session_state.analytics_data["timestamps"].append(datetime.now())

    category = "General"
    q = question.lower()
    if any(word in q for word in ["what", "define", "explain"]):
        category = "Informational"
    elif any(word in q for word in ["how", "can you", "show me"]):
        category = "Procedural"
    elif any(word in q for word in ["why", "reason", "because"]):
        category = "Analytical"

    st.session_state.analytics_data["query_categories"].append(category)
    times = st.session_state.analytics_data["response_times"]
    st.session_state.analytics_data["avg_response_time"] = sum(times) / len(times)


def display_analytics_dashboard():
    st.header("📊 Usage Analytics")

    analytics = st.session_state.analytics_data
    if analytics["total_queries"] == 0:
        st.info("No RAG queries yet. Ask something on the RAG Search page to start populating analytics.")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Queries", analytics["total_queries"])
    with c2:
        st.metric("Avg Response Time", f"{analytics['avg_response_time']:.2f}s")
    with c3:
        st.metric("Min Response Time", f"{min(analytics['response_times']):.2f}s")
    with c4:
        st.metric("Max Response Time", f"{max(analytics['response_times']):.2f}s")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        if analytics["query_categories"]:
            category_counts = pd.Series(analytics["query_categories"]).value_counts()
            fig_pie = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="Query Category Mix",
                color_discrete_sequence=px.colors.qualitative.Set3,
            )
            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        fig_hist = px.histogram(
            x=analytics["response_times"],
            nbins=20,
            title="Response Time Distribution",
            color_discrete_sequence=["#22c55e"],
        )
        fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_hist, use_container_width=True)

    if len(analytics["timestamps"]) > 1:
        df_time = pd.DataFrame(
            {
                "Timestamp": analytics["timestamps"],
                "Response Time": analytics["response_times"],
            }
        )
        fig_line = px.line(
            df_time,
            x="Timestamp",
            y="Response Time",
            title="Response Time Over Time",
            markers=True,
        )
        fig_line.update_traces(line_color="#38bdf8")
        fig_line.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_line, use_container_width=True)


# ---------------------------
# Main App
# ---------------------------
def render_top_bar():
    st.markdown(
        """
        <div class="top-bar">
            <div class="top-bar-left">
                <div class="top-pill">RAG Studio • Live</div>
                <div>
                    <div class="top-title">RAG Analytics Dashboard</div>
                    <div class="top-subtitle">Chat with documents, explore CSV relationships, and inspect data quality.</div>
                </div>
            </div>
            <div>
                <span class="top-chip">⚡ Powered by LLM + Vector Search</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    init_session_state()
    render_top_bar()

    # Sidebar navigation
    st.sidebar.title("🧭 Workspace")
    page = st.sidebar.radio(
        "Navigate",
        (
            "🔍 RAG Search",
            "📊 Analytics Dashboard",
            "📁 Multi-File Data Manager",
            "🔗 Relationship Explorer",
        ),
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Tip: Upload CSVs in the Data Manager, then explore their relationships.")

    # PAGE: RAG Search
    if page == "🔍 RAG Search":
        st.header("🔍 RAG Document Search")
        st.caption("Ask natural language questions about your ingested knowledge base.")

        if not st.session_state.initialized:
            with st.spinner("Bootstrapping RAG pipeline, this runs only once…"):
                rag_system, num_chunks = initialize_rag()
                if rag_system:
                    st.session_state.rag_system = rag_system
                    st.session_state.initialized = True
                    st.success(f"System ready. Indexed {num_chunks} document chunks.")
        st.markdown("---")

        with st.form("search_form"):
            question = st.text_input(
                "What would you like to know?",
                placeholder="e.g. Summarise the key risk themes in the latest policy documents",
            )
            submit = st.form_submit_button("🔍 Ask RAG")

        if submit and question:
            if st.session_state.rag_system:
                with st.spinner("Thinking with your documents…"):
                    start_time = time.time()
                    result = st.session_state.rag_system.run(question)
                    elapsed_time = time.time() - start_time
                    update_analytics(question, elapsed_time)

                    st.session_state.history.append(
                        {
                            "question": question,
                            "answer": result["answer"],
                            "time": elapsed_time,
                            "timestamp": datetime.now(),
                        }
                    )

                    st.markdown("### 💡 Answer")
                    st.success(result["answer"])

                    with st.expander("📄 Source Chunks"):
                        for i, doc in enumerate(result["retrieved_docs"], 1):
                            st.text_area(
                                f"Source {i}",
                                doc.page_content[:500] + "...",
                                height=120,
                                disabled=True,
                            )

                    st.caption(f"⏱️ Response time: {elapsed_time:.2f} seconds")

        if st.session_state.history:
            st.markdown("---")
            st.markdown("### 📜 Recent Questions")
            for item in reversed(st.session_state.history[-5:]):
                with st.container():
                    st.markdown(f"**Q:** {item['question']}")
                    st.markdown(f"**A:** {item['answer'][:220]}...")
                    st.caption(
                        f"{item['time']:.2f}s • {item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                    st.markdown("")

    # PAGE: Analytics Dashboard
    elif page == "📊 Analytics Dashboard":
        display_analytics_dashboard()

    # PAGE: Multi-File Data Manager
    elif page == "📁 Multi-File Data Manager":
        st.header("📁 Multi-File Data Manager")
        st.caption("Upload multiple CSVs (accounts, customers, products, etc.) and inspect structure & quality.")
        st.markdown("---")

        uploaded_files = st.file_uploader(
            "Upload CSV files",
            type=["csv"],
            accept_multiple_files=True,
            help="You can drag & drop multiple CSV files here.",
        )

        if uploaded_files:
            for uploaded_file in uploaded_files:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.uploaded_files[uploaded_file.name] = df
                except Exception as e:
                    st.error(f"Error reading {uploaded_file.name}: {str(e)}")

        if st.session_state.uploaded_files:
            st.markdown('<h3 class="dark-heading">🗂️ Loaded Files</h3>',
    unsafe_allow_html=True
        )

            cols = st.columns(min(len(st.session_state.uploaded_files), 4))
            for idx, (file_name, df) in enumerate(st.session_state.uploaded_files.items()):
                with cols[idx % 4]:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-pill">CSV</div>
                            <h4>{file_name}</h4>
                            <div class="metric-value">{len(df):,} rows</div>
                            <div class="metric-sub">{len(df.columns)} columns</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.markdown("---")
            st.markdown("### 👁️ Data Preview & Quality")

            tabs = st.tabs(list(st.session_state.uploaded_files.keys()))
            for tab, (file_name, df) in zip(
                tabs, st.session_state.uploaded_files.items()
            ):
                with tab:
                    col1, col2 = st.columns([2.5, 1.5])

                    with col1:
                        st.subheader("Sample Rows")
                        st.dataframe(df.head(12), use_container_width=True)

                    with col2:
                        st.subheader("Schema")
                        info_df = pd.DataFrame(
                            {
                                "Column": df.columns,
                                "Type": df.dtypes.values,
                                "Non-Null": df.count().values,
                            }
                        )
                        st.dataframe(
                            info_df, use_container_width=True, height=320
                        )

                        if st.button(
                            f"🔍 Run DQ Checks", key=f"dq_{file_name}"
                        ):
                            with st.spinner(f"Profiling {file_name}…"):
                                dq_result = perform_dq_checks(df, file_name)
                                st.session_state.dq_results[file_name] = dq_result

                    if file_name in st.session_state.dq_results:
                        dq = st.session_state.dq_results[file_name]
                        st.markdown("#### Data Quality Summary")
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.metric("DQ Score", f"{dq['overall_score']:.0f}%")
                        with c2:
                            passed = sum(
                                1 for c in dq["checks"] if c["status"] == "PASS"
                            )
                            st.metric("Checks Passed", f"{passed}/{len(dq['checks'])}")
                        with c3:
                            issues = len(
                                [c for c in dq["checks"] if c["status"] != "PASS"]
                            )
                            st.metric("Issues", issues)

                        with st.expander("View Detailed Checks", expanded=False):
                            for check in dq["checks"]:
                                status_class = f"dq-{check['status'].lower()}"
                                st.markdown(
                                    f"**{check['name']}:** "
                                    f"<span class='{status_class}'>{check['status']}</span>",
                                    unsafe_allow_html=True,
                                )
                                st.write(check["details"])
                                st.markdown("---")

    # PAGE: Relationship Explorer
    elif page == "🔗 Relationship Explorer":
        st.header("🔗 Relationship Explorer")
        st.caption("Discover shared keys, foreign keys, and join quality across your CSV files.")
        st.markdown("---")

        if len(st.session_state.uploaded_files) < 2:
            st.warning(
                "Upload at least two CSV files in the Multi-File Data Manager to explore relationships."
            )
            return

        st.markdown("### 📁 Available Files")
        cols = st.columns(len(st.session_state.uploaded_files))
        for idx, (file_name, df) in enumerate(st.session_state.uploaded_files.items()):
            with cols[idx]:
                st.markdown(
                    f"<div class='file-badge badge-success'>📄 {file_name} • {len(df)} rows</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        tab1, tab2 = st.tabs(["🤖 Automatic Detection", "✋ Manual Mapping"])

        with tab1:
            c1, _ = st.columns([1, 3])
            with c1:
                detect_btn = st.button(
                    "🔍 Detect Relationships", type="primary", use_container_width=True
                )

            if detect_btn:
                with st.spinner("Scanning for shared keys and overlapping values…"):
                    detector = RelationshipDetector()
                    relationships = detector.detect(st.session_state.uploaded_files)

                    #relationships = detect_relationships(
                     #   st.session_state.uploaded_files
                    #)
                    st.session_state.relationships = relationships
                    if relationships:
                        st.success(
                            f"Detected {len(relationships)} relationship(s) across your files."
                        )
                    else:
                        st.warning(
                            "No automatic relationships found. Try manual mapping or verify column naming."
                        )

            if st.session_state.relationships:
                st.markdown("### 🕸 Network View")
                display_relationship_network(
                    st.session_state.relationships,
                    st.session_state.uploaded_files,
                )

                st.markdown("### 📋 Relationship Details")
                for idx, rel in enumerate(st.session_state.relationships, 1):
                    label = (
                        f"Relationship {idx}: {rel['file1']} ↔ {rel['file2']} "
                        f"via '{rel['column']}' ↔ '{rel['column_file2']}'"
                    )
                    with st.expander(label, expanded=False):
                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            st.metric("Column (File 1)", rel["column"])
                        with c2:
                            st.metric("Column (File 2)", rel["column_file2"])
                        with c3:
                            st.metric("Matching Values", rel["overlap_count"])
                        with c4:
                            st.metric(
                                "Match Rate",
                                f"{rel['overlap_percentage']:.1f}%",
                            )

                        c5, c6 = st.columns(2)
                        with c5:
                            st.info(
                                f"Unique in {rel['file1']}: {rel['total_unique_file1']}"
                            )
                        with c6:
                            st.info(
                                f"Unique in {rel['file2']}: {rel['total_unique_file2']}"
                            )

                        st.markdown(
                            f"**Type:** "
                            f"<span class='file-badge badge-primary'>{rel['relationship_type']}</span>",
                            unsafe_allow_html=True,
                        )

                        if st.button(
                            "Perform Join Analysis", key=f"join_{idx}"
                        ):
                            df1 = st.session_state.uploaded_files[rel["file1"]]
                            df2 = st.session_state.uploaded_files[rel["file2"]]
                            joined_df, stats = perform_join_analysis(
                                df1, df2, rel["column"], rel["column_file2"]
                            )

                            if joined_df is not None and stats is not None:
                                st.markdown("#### Join Statistics")
                                j1, j2, j3, j4 = st.columns(4)
                                with j1:
                                    st.metric(
                                        "File 1 Records",
                                        stats["total_records_file1"],
                                    )
                                with j2:
                                    st.metric(
                                        "File 2 Records",
                                        stats["total_records_file2"],
                                    )
                                with j3:
                                    st.metric(
                                        "Matched Records",
                                        stats["inner_join_records"],
                                    )
                                with j4:
                                    st.metric(
                                        "Match Rate",
                                        f"{stats['match_rate']:.1f}%",
                                    )

                                st.markdown("#### Inner Join Preview")
                                st.dataframe(
                                    joined_df.head(20), use_container_width=True
                                )

                                fig = go.Figure(
                                    data=[
                                        go.Bar(
                                            x=["Matched", "Unmatched"],
                                            y=[
                                                stats["inner_join_records"],
                                                stats["unmatched_records"],
                                            ],
                                            marker_color=["#22c55e", "#ef4444"],
                                        )
                                    ]
                                )
                                fig.update_layout(
                                    title="Matched vs Unmatched Records",
                                    yaxis_title="Count",
                                    height=320,
                                    paper_bgcolor="rgba(0,0,0,0)",
                                )
                                st.plotly_chart(
                                    fig, use_container_width=True
                                )

                                csv = joined_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Joined CSV",
                                    data=csv,
                                    file_name=f"joined_{rel['file1']}_{rel['file2']}.csv",
                                    mime="text/csv",
                                )
            else:
                st.info("Click **Detect Relationships** above to generate suggestions.")

        with tab2:
            st.markdown("### ✋ Manual Relationship Mapping")
            st.info(
                "Use manual mapping when automatic detection cannot infer the correct join keys."
            )

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**File 1**")
                file1_name = st.selectbox(
                    "Select first file",
                    list(st.session_state.uploaded_files.keys()),
                    key="manual_file1",
                )
                col1_name = None
                if file1_name:
                    df1 = st.session_state.uploaded_files[file1_name]
                    col1_name = st.selectbox(
                        "Select join column", df1.columns.tolist(), key="manual_col1"
                    )
                    if col1_name:
                        st.write(f"Sample values in **{col1_name}**:")
                        st.write(df1[col1_name].dropna().head(5).tolist())

            with col2:
                st.markdown("**File 2**")
                remaining_files = [
                    f for f in st.session_state.uploaded_files.keys() if f != file1_name
                ]
                file2_name = st.selectbox(
                    "Select second file", remaining_files, key="manual_file2"
                )
                col2_name = None
                if file2_name:
                    df2 = st.session_state.uploaded_files[file2_name]
                    col2_name = st.selectbox(
                        "Select join column", df2.columns.tolist(), key="manual_col2"
                    )
                    if col2_name:
                        st.write(f"Sample values in **{col2_name}**:")
                        st.write(df2[col2_name].dropna().head(5).tolist())

            if st.button("🔗 Create Manual Relationship", type="primary"):
                if file1_name and file2_name and col1_name and col2_name:
                    try:
                        df1 = st.session_state.uploaded_files[file1_name]
                        df2 = st.session_state.uploaded_files[file2_name]
                        values1 = set(df1[col1_name].dropna().astype(str).unique())
                        values2 = set(df2[col2_name].dropna().astype(str).unique())
                        overlap = values1.intersection(values2)

                        if len(overlap) > 0:
                            overlap_pct = (
                                len(overlap) / max(len(values1), len(values2)) * 100
                            )
                            manual_rel = {
                                "file1": file1_name,
                                "file2": file2_name,
                                "column": col1_name,
                                "column_file2": col2_name,
                                "overlap_count": len(overlap),
                                "overlap_percentage": overlap_pct,
                                "total_unique_file1": len(values1),
                                "total_unique_file2": len(values2),
                                "relationship_type": "Manual Mapping",
                            }
                            if manual_rel not in st.session_state.relationships:
                                st.session_state.relationships.append(manual_rel)
                                st.success(
                                    f"Relationship created. {len(overlap)} matching values "
                                    f"({overlap_pct:.1f}% match rate)."
                                )
                                st.experimental_rerun()
                        else:
                            st.error(
                                "No overlapping values between the selected columns. Verify data or choose different columns."
                            )
                    except Exception as e:
                        st.error(f"Error creating relationship: {str(e)}")
                else:
                    st.warning(
                        "Choose both files and their join columns before creating a relationship."
                    )


if __name__ == "__main__":
    main()
