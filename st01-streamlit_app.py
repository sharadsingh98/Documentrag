import streamlit as st
from pathlib import Path
import sys
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import threading
import queue
from typing import Dict, List, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder
from relationship_detector import RelationshipDetector
from st_integration import (
    SQLDatabaseManager,
    NaturalLanguageToSQL,
    SQLQueryHistory,
    render_sql_database_page
)


# ---------------------------
# Global Page Configuration
# ---------------------------
# ---------------------------
# Global Page Configuration
# ---------------------------
st.set_page_config(
    page_title="🤖 DataLens Intelligence Studio",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Enhanced Global CSS / Theme - Light Blue Main Area
# ---------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* === Light Blue Background for Main Area === */
    .main {
        background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 50%, #7dd3fc 100%);
        background-attachment: fixed;
        color: #0c4a6e;
    }
    
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 1600px;
    }

    /* === Headings - Deep Blue for Light Background === */
    h1, h2, h3, h4, h5, h6 {
        color: #0c4a6e !important;
        letter-spacing: -0.02em;
        font-weight: 800;
        margin-bottom: 0.8em;
        text-shadow: 0 2px 4px rgba(12, 74, 110, 0.1);
    }

    h1 { 
        font-size: 2.8rem !important;
        background: linear-gradient(135deg, #0c4a6e 0%, #0369a1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    h2 { font-size: 2.2rem !important; }
    h3 { font-size: 1.8rem !important; }
    h4 { font-size: 1.5rem !important; }
    h5 { font-size: 1.3rem !important; }
    h6 { font-size: 1.15rem !important; }

    /* === Text Color for Light Background === */
    p, span, div, label {
        color: #0c4a6e !important;
    }
    
    /* === Sidebar Styling - Dark Theme === */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a1a 0%, #0f172a 100%) !important;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5,
    [data-testid="stSidebar"] h6 {
        color: #ffffff !important;
        text-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
    }
    
    [data-testid="stSidebar"] .css-1d391kg,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        color: #e5e7eb !important;
    }
    
    /* Sidebar radio buttons */
    [data-testid="stSidebar"] [data-testid="stRadio"] label,
    [data-testid="stSidebar"] [data-testid="stRadio"] > div > label {
        color: #ffffff !important;
        font-weight: 500;
    }
    
    /* Sidebar divider */
    [data-testid="stSidebar"] hr {
        border-color: #334155 !important;
        opacity: 0.5;
    }
    
    /* Sidebar captions */
    [data-testid="stSidebar"] .css-16huue1,
    [data-testid="stSidebar"] small,
    [data-testid="stSidebar"] [data-baseweb="caption"] {
        color: #94a3b8 !important;
    }

    /* === Top Bar - White with Blue Accents === */
    .top-bar {
        background: rgba(255, 255, 255, 0.98);
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        border-radius: 24px;
        padding: 1.8rem 2.5rem;
        margin: 0 0 2rem 0;
        border: 2px solid #bae6fd;
        box-shadow: 0 20px 60px -15px rgba(14, 165, 233, 0.3), 
                    0 0 0 1px rgba(186, 230, 253, 0.5) inset;
        display: flex;
        justify-content: space-between;
        align-items: center;
        position: relative;
        overflow: hidden;
    }
    
    .top-bar::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #0ea5e9, #0284c7, #0369a1);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0%, 100% { opacity: 0.6; }
        50% { opacity: 1; }
    }

    .top-pill {
        display: inline-block;
        padding: 0.4rem 1rem;
        background: linear-gradient(135deg, #0ea5e9, #0284c7);
        color: white;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.3);
    }

    .top-title {
        font-size: 1.8rem;
        font-weight: 900;
        background: linear-gradient(135deg, #0c4a6e 0%, #0369a1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }
    
    .top-subtitle {
        font-size: 0.95rem;
        color: #475569;
        font-weight: 500;
        margin-top: 0.4rem;
    }

    .top-chip {
        padding: 0.5rem 1.2rem;
        background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
        color: #0369a1;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid #bae6fd;
    }
    
    /* === Buttons - Blue Theme === */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
        color: white;
        font-weight: 800;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        border-radius: 16px;
        border: none;
        padding: 1rem 1.8rem;
        font-size: 0.95rem;
        box-shadow: 0 10px 30px -8px rgba(14, 165, 233, 0.5),
                    0 0 0 1px rgba(255,255,255,0.3) inset;
        transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
    }
    
    .stButton > button:hover {
        transform: translateY(-4px) scale(1.02);
        box-shadow: 0 20px 40px -10px rgba(14, 165, 233, 0.7);
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
    }

    /* === Metric Cards - White with Blue Accents === */
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem 1.8rem;
        border-radius: 20px;
        border: 2px solid #bae6fd;
        box-shadow: 0 20px 50px -15px rgba(14, 165, 233, 0.2);
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 25px 60px -15px rgba(14, 165, 233, 0.3);
        border-color: #7dd3fc;
    }
    
    .metric-card .metric-value {
        color: #0c4a6e !important;
        font-size: 2.6rem;
        font-weight: 900;
    }

    .metric-pill {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        background: linear-gradient(135deg, #0ea5e9, #0284c7);
        color: white;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
    }

    .metric-sub {
        color: #64748b !important;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }

    [data-testid="stMetricValue"] {
        background: linear-gradient(135deg, #0c4a6e 0%, #0369a1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 900;
    }

    /* === File Badges === */
    .file-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.25rem;
    }

    .badge-success {
        background: linear-gradient(135deg, #22c55e, #16a34a);
        color: white;
        box-shadow: 0 4px 12px rgba(34, 197, 94, 0.3);
    }

    .badge-primary {
        background: linear-gradient(135deg, #0ea5e9, #0284c7);
        color: white;
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.3);
    }

    /* === Streaming Status Indicators === */
    .streaming-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        animation: pulse 2s infinite;
    }
    
    .streaming-active {
        background: linear-gradient(135deg, #22c55e, #16a34a);
        color: white;
        box-shadow: 0 4px 12px rgba(34, 197, 94, 0.4);
    }
    
    .streaming-inactive {
        background: linear-gradient(135deg, #94a3b8, #64748b);
        color: white;
        box-shadow: 0 4px 12px rgba(148, 163, 184, 0.3);
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.85; transform: scale(1.02); }
    }

    /* === Data Quality Badges === */
    .dq-pass {
        color: #16a34a;
        font-weight: 700;
    }

    .dq-warning {
        color: #f59e0b;
        font-weight: 700;
    }

    .dq-fail {
        color: #dc2626;
        font-weight: 700;
    }

    /* === Expanders === */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.9) !important;
        border: 1px solid #bae6fd !important;
        border-radius: 12px !important;
        color: #0c4a6e !important;
        font-weight: 600 !important;
    }

    /* === Info/Warning/Success Boxes === */
    .stAlert {
        border-radius: 16px !important;
        border-left: 4px solid !important;
    }

    /* === Dataframe Styling === */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 8px 24px rgba(14, 165, 233, 0.15);
    }

    /* === Tabs === */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.6);
        padding: 0.5rem;
        border-radius: 16px;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 12px;
        color: #0c4a6e;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0ea5e9, #0284c7);
        color: white !important;
    }

    /* === Form Inputs === */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select {
        border-radius: 12px !important;
        border: 2px solid #bae6fd !important;
        background: white !important;
        color: #0c4a6e !important;
    }

    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #0ea5e9 !important;
        box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.1) !important;
    }

    /* === Dividers === */
    hr {
        border-color: #bae6fd !important;
        opacity: 0.6;
    }

    </style>
    """,
    unsafe_allow_html=True,
)




# ---------------------------
# Streaming Data Manager
# ---------------------------
class StreamingDataManager:
    """Manages streaming data sources and real-time updates"""
    
    def __init__(self):
        self.data_queue = queue.Queue()
        self.streaming_active = False
        self.stream_thread = None
        self.stream_sources = {}
        
    def add_stream_source(self, name: str, source_type: str, config: Dict):
        """Add a new streaming data source"""
        self.stream_sources[name] = {
            'type': source_type,
            'config': config,
            'data': pd.DataFrame(),
            'last_update': None,
            'record_count': 0
        }
    
    def simulate_stream(self, source_name: str, interval: float = 1.0):
        """Simulate streaming data (for demo purposes)"""
        while self.streaming_active:
            try:
                # Generate sample data based on source type
                if source_name in self.stream_sources:
                    source = self.stream_sources[source_name]
                    
                    if source['type'] == 'sales':
                        new_data = {
                            'timestamp': datetime.now(),
                            'transaction_id': f"TXN{np.random.randint(10000, 99999)}",
                            'amount': round(np.random.uniform(10, 1000), 2),
                            'product': np.random.choice(['Product A', 'Product B', 'Product C', 'Product D']),
                            'region': np.random.choice(['North', 'South', 'East', 'West']),
                            'customer_id': f"CUST{np.random.randint(1000, 9999)}"
                        }
                    elif source['type'] == 'iot':
                        new_data = {
                            'timestamp': datetime.now(),
                            'device_id': f"DEV{np.random.randint(100, 999)}",
                            'temperature': round(np.random.uniform(18, 28), 2),
                            'humidity': round(np.random.uniform(30, 70), 2),
                            'status': np.random.choice(['active', 'active', 'active', 'warning'])
                        }
                    elif source['type'] == 'logs':
                        new_data = {
                            'timestamp': datetime.now(),
                            'level': np.random.choice(['INFO', 'INFO', 'INFO', 'WARNING', 'ERROR']),
                            'service': np.random.choice(['API', 'Database', 'Cache', 'Auth']),
                            'message': f"Event {np.random.randint(1000, 9999)}",
                            'response_time_ms': np.random.randint(10, 500)
                        }
                    else:
                        new_data = {
                            'timestamp': datetime.now(),
                            'value': np.random.random() * 100
                        }
                    
                    self.data_queue.put((source_name, new_data))
                    source['record_count'] += 1
                    source['last_update'] = datetime.now()
                    
                time.sleep(interval)
            except Exception as e:
                print(f"Streaming error: {e}")
                break
    
    def start_streaming(self, source_name: str):
        """Start streaming for a specific source"""
        if not self.streaming_active:
            self.streaming_active = True
            self.stream_thread = threading.Thread(
                target=self.simulate_stream,
                args=(source_name, 1.0),
                daemon=True
            )
            self.stream_thread.start()
    
    def stop_streaming(self):
        """Stop all streaming"""
        self.streaming_active = False
        if self.stream_thread:
            self.stream_thread.join(timeout=2)
    
    def get_stream_data(self, source_name: str, max_records: int = 1000) -> pd.DataFrame:
        """Get accumulated streaming data"""
        if source_name in self.stream_sources:
            source = self.stream_sources[source_name]
            
            # Process queued data
            while not self.data_queue.empty():
                try:
                    src_name, new_data = self.data_queue.get_nowait()
                    if src_name == source_name:
                        new_df = pd.DataFrame([new_data])
                        if source['data'].empty:
                            source['data'] = new_df
                        else:
                            source['data'] = pd.concat([source['data'], new_df], ignore_index=True)
                            # Keep only last N records
                            if len(source['data']) > max_records:
                                source['data'] = source['data'].tail(max_records)
                except queue.Empty:
                    break
            
            return source['data']
        return pd.DataFrame()
    
    def get_stream_stats(self, source_name: str) -> Dict:
        """Get statistics for a streaming source"""
        if source_name in self.stream_sources:
            source = self.stream_sources[source_name]
            df = source['data']
            
            return {
                'total_records': source['record_count'],
                'buffered_records': len(df),
                'last_update': source['last_update'],
                'is_active': self.streaming_active,
                'source_type': source['type']
            }
        return {}


# ---------------------------
# Session State
# ---------------------------
def init_session_state():
    defaults = {
        "rag_system": None,
        "initialized": False,
        "history": [],
        "uploaded_files": {},
        "relationships": [],
        "dq_results": {},
        "analytics_data": {
            "total_queries": 0,
            "avg_response_time": 0,
            "query_categories": [],
            "response_times": [],
            "timestamps": [],
        },
        "streaming_manager": StreamingDataManager(),
        "db_manager": None,
        "query_history": None,
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
        st.info("No RAG queries yet. Ask something on the Intelligent Search page to start populating analytics.")
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
# Streaming Data Page
# ---------------------------
def render_streaming_data_page():
    st.header("🌊 Real-Time Streaming Analytics")
    st.caption("Monitor and analyze live data streams")
    st.markdown("---")
    
    streaming_mgr = st.session_state.streaming_manager
    
    # Sidebar controls
    with st.sidebar:
        st.subheader("📡 Stream Sources")
        
        # Add new stream source
        with st.expander("➕ Add Stream Source"):
            stream_name = st.text_input("Source Name", "My Stream")
            stream_type = st.selectbox(
                "Stream Type",
                ["sales", "iot", "logs", "custom"]
            )
            
            if st.button("Create Stream"):
                streaming_mgr.add_stream_source(
                    stream_name,
                    stream_type,
                    {}
                )
                st.success(f"Stream '{stream_name}' created!")
        
        # List existing streams
        if streaming_mgr.stream_sources:
            st.markdown("---")
            st.markdown("**Active Sources:**")
            for source_name in streaming_mgr.stream_sources.keys():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"📊 {source_name}")
                with col2:
                    if streaming_mgr.streaming_active:
                        st.markdown("🟢")
                    else:
                        st.markdown("⚪")
    
    # Main content
    if not streaming_mgr.stream_sources:
        st.info("👈 Add a stream source from the sidebar to begin")
        return
    
    # Stream controls
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        selected_stream = st.selectbox(
            "Select Stream",
            list(streaming_mgr.stream_sources.keys())
        )
    with col2:
        if not streaming_mgr.streaming_active:
            if st.button("▶️ Start Stream", type="primary"):
                streaming_mgr.start_streaming(selected_stream)
                st.rerun()
        else:
            if st.button("⏹️ Stop Stream", type="secondary"):
                streaming_mgr.stop_streaming()
                st.rerun()
    with col3:
        auto_refresh = st.checkbox("Auto Refresh", value=True)
    
    if selected_stream:
        stats = streaming_mgr.get_stream_stats(selected_stream)
        
        # Status badge
        if stats.get('is_active'):
            status_html = '<span class="streaming-badge streaming-active">🔴 LIVE</span>'
        else:
            status_html = '<span class="streaming-badge streaming-inactive">⚫ STOPPED</span>'
        
        st.markdown(status_html, unsafe_allow_html=True)
        
        # Metrics
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Records", f"{stats.get('total_records', 0):,}")
        with c2:
            st.metric("Buffered", f"{stats.get('buffered_records', 0):,}")
        with c3:
            last_update = stats.get('last_update')
            if last_update:
                seconds_ago = (datetime.now() - last_update).total_seconds()
                st.metric("Last Update", f"{seconds_ago:.1f}s ago")
            else:
                st.metric("Last Update", "Never")
        with c4:
            st.metric("Type", stats.get('source_type', 'N/A'))
        
        st.markdown("---")
        
        # Get streaming data
        stream_df = streaming_mgr.get_stream_data(selected_stream)
        
        if not stream_df.empty:
            tab1, tab2, tab3 = st.tabs(["📊 Live Data", "📈 Visualizations", "📋 Analytics"])
            
            with tab1:
                st.subheader("Live Data Stream")
                st.dataframe(
                    stream_df.tail(100).sort_values('timestamp', ascending=False),
                    use_container_width=True,
                    height=400
                )
                
                # Export option
                csv = stream_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Stream Data",
                    csv,
                    f"stream_{selected_stream}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )
            
            with tab2:
                st.subheader("Real-Time Visualizations")
                
                # Time series visualization
                if 'timestamp' in stream_df.columns:
                    numeric_cols = stream_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
                    
                    if numeric_cols:
                        viz_col = st.selectbox("Select metric to visualize", numeric_cols)
                        
                        # Create time series chart
                        fig_ts = go.Figure()
                        fig_ts.add_trace(go.Scatter(
                            x=stream_df['timestamp'],
                            y=stream_df[viz_col],
                            mode='lines+markers',
                            name=viz_col,
                            line=dict(color='#22c55e', width=2),
                            marker=dict(size=4)
                        ))
                        
                        fig_ts.update_layout(
                            title=f"{viz_col} Over Time",
                            xaxis_title="Time",
                            yaxis_title=viz_col,
                            height=400,
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0.05)"
                        )
                        
                        st.plotly_chart(fig_ts, use_container_width=True)
                        
                        # Distribution
                        fig_dist = px.histogram(
                            stream_df,
                            x=viz_col,
                            nbins=30,
                            title=f"{viz_col} Distribution"
                        )
                        fig_dist.update_layout(paper_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(fig_dist, use_container_width=True)
                    
                    # Categorical analysis
                    categorical_cols = stream_df.select_dtypes(include=['object']).columns.tolist()
                    if categorical_cols:
                        cat_col = st.selectbox("Select category", categorical_cols)
                        
                        value_counts = stream_df[cat_col].value_counts()
                        fig_pie = px.pie(
                            values=value_counts.values,
                            names=value_counts.index,
                            title=f"{cat_col} Distribution"
                        )
                        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(fig_pie, use_container_width=True)
            
            with tab3:
                st.subheader("Stream Analytics")
                
                # Statistical summary
                st.markdown("#### Statistical Summary")
                numeric_df = stream_df.select_dtypes(include=['float64', 'int64'])
                if not numeric_df.empty:
                    st.dataframe(numeric_df.describe(), use_container_width=True)
                
                # Recent activity
                st.markdown("#### Recent Activity (Last 5 minutes)")
                recent_df = stream_df[
                    stream_df['timestamp'] > datetime.now() - timedelta(minutes=5)
                ]
                st.metric("Records (5 min)", len(recent_df))
                
                # Anomaly detection (simple threshold-based)
                if not numeric_df.empty:
                    st.markdown("#### Anomaly Detection")
                    for col in numeric_df.columns:
                        mean = stream_df[col].mean()
                        std = stream_df[col].std()
                        anomalies = stream_df[
                            (stream_df[col] > mean + 2*std) | 
                            (stream_df[col] < mean - 2*std)
                        ]
                        
                        if len(anomalies) > 0:
                            st.warning(f"⚠️ {len(anomalies)} anomalies detected in {col}")
        else:
            st.info("Waiting for data... Start the stream to see real-time updates.")
        
        # Auto-refresh
        if auto_refresh and streaming_mgr.streaming_active:
            time.sleep(2)
            st.rerun()


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
                    <div class="top-title">DataLens Intelligence Studio</div>
                    <div class="top-subtitle">Chat with documents, explore CSV relationships, query databases, and analyze streaming data.</div>
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

    # Initialize SQL components if not already done
    if st.session_state.db_manager is None:
        st.session_state.db_manager = SQLDatabaseManager()
    if st.session_state.query_history is None:
        st.session_state.query_history = SQLQueryHistory()

    # Sidebar navigation
    st.sidebar.title("🧭 Workspace")
    page = st.sidebar.radio(
        "Navigate",
        (
            "🔍 Intelligent Search",
            "📊 Analytics Studio",
            "📁 Data Catalog",
            "🔗 Data Lineage",
            "💾 Database Studio",
            "🌊 Streaming Analytics",
        ),
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Multi-source intelligence platform for comprehensive data analysis")

    # PAGE: Intelligent Search
    if page == "🔍 Intelligent Search":
        st.header("🔍 DataLens Intelligence")
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
            submit = st.form_submit_button("🔍 Ask Questions")

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

    # PAGE: Analytics Studio
    elif page == "📊 Analytics Studio":
        display_analytics_dashboard()

    # PAGE: Data Catalog
    elif page == "📁 Data Catalog":
        st.header("📁 Data Catalog")
        st.caption("Upload multiple files (CSV, JSON, Parquet) and inspect structure & quality.")
        st.markdown("---")

        uploaded_files = st.file_uploader(
            "Upload data files",
            type=["csv", "json", "parquet"],
            accept_multiple_files=True,
            help="You can drag & drop multiple CSV, JSON, or Parquet files here.",
        )

        if uploaded_files:
            for uploaded_file in uploaded_files:
                try:
                    file_extension = uploaded_file.name.split('.')[-1].lower()
                    
                    if file_extension == 'csv':
                        df = pd.read_csv(uploaded_file)
                    elif file_extension == 'json':
                        df = pd.read_json(uploaded_file)
                    elif file_extension == 'parquet':
                        df = pd.read_parquet(uploaded_file)
                    else:
                        st.error(f"Unsupported file type: {file_extension}")
                        continue
                    
                    st.session_state.uploaded_files[uploaded_file.name] = df
                except Exception as e:
                    st.error(f"Error reading {uploaded_file.name}: {str(e)}")

        if st.session_state.uploaded_files:
            st.markdown('<h3 class="dark-heading">🗂️ Loaded Files</h3>', unsafe_allow_html=True)

            cols = st.columns(min(len(st.session_state.uploaded_files), 4))
            for idx, (file_name, df) in enumerate(st.session_state.uploaded_files.items()):
                # Determine file type for display
                file_extension = file_name.split('.')[-1].upper()
                
                with cols[idx % 4]:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-pill">{file_extension}</div>
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
            for tab, (file_name, df) in zip(tabs, st.session_state.uploaded_files.items()):
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
                        st.dataframe(info_df, use_container_width=True, height=320)

                        if st.button(f"🔍 Run DQ Checks", key=f"dq_{file_name}"):
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
                            passed = sum(1 for c in dq["checks"] if c["status"] == "PASS")
                            st.metric("Checks Passed", f"{passed}/{len(dq['checks'])}")
                        with c3:
                            issues = len([c for c in dq["checks"] if c["status"] != "PASS"])
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

    # PAGE: Data Lineage
    elif page == "🔗 Data Lineage":
        st.header("🔗 Data Lineage")
        st.caption("Discover shared keys, foreign keys, and join quality across your CSV files.")
        st.markdown("---")

        if len(st.session_state.uploaded_files) < 2:
            st.warning("Upload at least two CSV files in the Data Catalog to explore relationships.")
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
                detect_btn = st.button("🔍 Detect Relationships", type="primary", use_container_width=True)

            if detect_btn:
                with st.spinner("Scanning for shared keys and overlapping values…"):
                    detector = RelationshipDetector()
                    relationships = detector.detect(st.session_state.uploaded_files)

                    st.session_state.relationships = relationships
                    if relationships:
                        st.success(f"Detected {len(relationships)} relationship(s) across your files.")
                    else:
                        st.warning("No automatic relationships found. Try manual mapping or verify column naming.")

            if st.session_state.relationships:
                st.markdown("### 🕸 Network View")
                display_relationship_network(st.session_state.relationships, st.session_state.uploaded_files)

                st.markdown("### 📋 Relationship Details")
                for idx, rel in enumerate(st.session_state.relationships, 1):
                    label = f"Relationship {idx}: {rel['file1']} ↔ {rel['file2']} via '{rel['column']}' ↔ '{rel['column_file2']}'"
                    with st.expander(label, expanded=False):
                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            st.metric("Column (File 1)", rel["column"])
                        with c2:
                            st.metric("Column (File 2)", rel["column_file2"])
                        with c3:
                            st.metric("Matching Values", rel["overlap_count"])
                        with c4:
                            st.metric("Match Rate", f"{rel['overlap_percentage']:.1f}%")

                        c5, c6 = st.columns(2)
                        with c5:
                            st.info(f"Unique in {rel['file1']}: {rel['total_unique_file1']}")
                        with c6:
                            st.info(f"Unique in {rel['file2']}: {rel['total_unique_file2']}")

                        st.markdown(
                            f"**Type:** <span class='file-badge badge-primary'>{rel['relationship_type']}</span>",
                            unsafe_allow_html=True,
                        )

                        if st.button("Perform Join Analysis", key=f"join_{idx}"):
                            df1 = st.session_state.uploaded_files[rel["file1"]]
                            df2 = st.session_state.uploaded_files[rel["file2"]]
                            joined_df, stats = perform_join_analysis(
                                df1, df2, rel["column"], rel["column_file2"]
                            )

                            if joined_df is not None and stats is not None:
                                st.markdown("#### Join Statistics")
                                j1, j2, j3, j4 = st.columns(4)
                                with j1:
                                    st.metric("File 1 Records", stats["total_records_file1"])
                                with j2:
                                    st.metric("File 2 Records", stats["total_records_file2"])
                                with j3:
                                    st.metric("Matched Records", stats["inner_join_records"])
                                with j4:
                                    st.metric("Match Rate", f"{stats['match_rate']:.1f}%")

                                st.markdown("#### Inner Join Preview")
                                st.dataframe(joined_df.head(20), use_container_width=True)

                                fig = go.Figure(
                                    data=[
                                        go.Bar(
                                            x=["Matched", "Unmatched"],
                                            y=[stats["inner_join_records"], stats["unmatched_records"]],
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
                                st.plotly_chart(fig, use_container_width=True)

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
            st.info("Use manual mapping when automatic detection cannot infer the correct join keys.")

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
                    col1_name = st.selectbox("Select join column", df1.columns.tolist(), key="manual_col1")
                    if col1_name:
                        st.write(f"Sample values in **{col1_name}**:")
                        st.write(df1[col1_name].dropna().head(5).tolist())

            with col2:
                st.markdown("**File 2**")
                remaining_files = [f for f in st.session_state.uploaded_files.keys() if f != file1_name]
                file2_name = st.selectbox("Select second file", remaining_files, key="manual_file2")
                col2_name = None
                if file2_name:
                    df2 = st.session_state.uploaded_files[file2_name]
                    col2_name = st.selectbox("Select join column", df2.columns.tolist(), key="manual_col2")
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
                            overlap_pct = len(overlap) / max(len(values1), len(values2)) * 100
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
                                st.success(f"Relationship created. {len(overlap)} matching values ({overlap_pct:.1f}% match rate).")
                                st.rerun()
                        else:
                            st.error("No overlapping values between the selected columns.")
                    except Exception as e:
                        st.error(f"Error creating relationship: {str(e)}")
                else:
                    st.warning("Choose both files and their join columns before creating a relationship.")

    # PAGE: Database Studio
    elif page == "💾 Database Studio":
        render_sql_database_page()

    # PAGE: Streaming Analytics
    elif page == "🌊 Streaming Analytics":
        render_streaming_data_page()


if __name__ == "__main__":
    main()