"""
Enhanced SQL Database Integration Module for RAG Analytics Studio
Enables natural language queries to SQL databases with advanced analytics
"""

import streamlit as st
import pandas as pd
import sqlite3
import sqlalchemy
from sqlalchemy import create_engine, inspect, text
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re

from src.config.config import Config


class SQLDatabaseManager:
    """Manages SQL database connections and metadata"""
    
    def __init__(self):
        self.engine = None
        self.connection_string = None
        self.db_type = None
        
    def connect(self, connection_string: str, db_type: str = "sqlite") -> bool:
        """
        Connect to a database
        
        Args:
            connection_string: Database connection string
            db_type: Type of database (sqlite, postgresql, mysql, mssql)
        
        Returns:
            bool: True if connection successful
        """
        try:
            self.engine = create_engine(connection_string)
            self.connection_string = connection_string
            self.db_type = db_type
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            st.error(f"Connection failed: {str(e)}")
            return False
    
    def get_tables(self) -> List[str]:
        """Get list of all tables in the database"""
        if not self.engine:
            return []
        inspector = inspect(self.engine)
        return inspector.get_table_names()
    
    def get_table_schema(self, table_name: str) -> pd.DataFrame:
        """Get schema information for a specific table"""
        if not self.engine:
            return pd.DataFrame()
        
        inspector = inspect(self.engine)
        columns = inspector.get_columns(table_name)
        
        schema_data = []
        for col in columns:
            schema_data.append({
                'Column': col['name'],
                'Type': str(col['type']),
                'Nullable': col['nullable'],
                'Default': col.get('default', None)
            })
        
        return pd.DataFrame(schema_data)
    
    def get_sample_data(self, table_name: str, limit: int = 5) -> pd.DataFrame:
        """Get sample rows from a table"""
        if not self.engine:
            return pd.DataFrame()
        
        query = f"SELECT * FROM {table_name} LIMIT {limit}"
        return pd.read_sql(query, self.engine)
    
    def get_table_stats(self, table_name: str) -> Dict:
        """Get statistics about a table"""
        if not self.engine:
            return {}
        
        with self.engine.connect() as conn:
            # Row count
            count_query = f"SELECT COUNT(*) as count FROM {table_name}"
            row_count = conn.execute(text(count_query)).fetchone()[0]
            
            # Column count
            inspector = inspect(self.engine)
            columns = inspector.get_columns(table_name)
            
            return {
                'row_count': row_count,
                'column_count': len(columns),
                'columns': [col['name'] for col in columns]
            }
    
    def execute_query(self, query: str) -> Tuple[pd.DataFrame, Optional[str]]:
        """
        Execute a SQL query and return results
        
        Returns:
            Tuple of (DataFrame, error_message)
        """
        if not self.engine:
            return pd.DataFrame(), "No database connection"
        
        try:
            df = pd.read_sql(query, self.engine)
            return df, None
        except Exception as e:
            return pd.DataFrame(), str(e)
    
    def get_foreign_keys(self, table_name: str) -> List[Dict]:
        """Get foreign key relationships for a table"""
        if not self.engine:
            return []
        
        inspector = inspect(self.engine)
        fks = inspector.get_foreign_keys(table_name)
        return fks
    
    def get_indexes(self, table_name: str) -> List[Dict]:
        """Get indexes for a table"""
        if not self.engine:
            return []
        
        inspector = inspect(self.engine)
        indexes = inspector.get_indexes(table_name)
        return indexes


class NaturalLanguageToSQL:
    """Converts natural language queries to SQL using LLM"""
    
    def __init__(self, llm, db_manager: SQLDatabaseManager):
        self.llm = llm
        self.db_manager = db_manager
        
    def get_database_context(self) -> str:
        """Generate database schema context for the LLM"""
        if not self.db_manager.engine:
            return "No database connected"
        
        context = "Database Schema Information:\n\n"
        tables = self.db_manager.get_tables()
        
        for table in tables:
            stats = self.db_manager.get_table_stats(table)
            schema_df = self.db_manager.get_table_schema(table)
            
            context += f"Table: {table}\n"
            context += f"Row Count: {stats['row_count']}\n"
            context += f"Columns:\n"
            
            for _, row in schema_df.iterrows():
                context += f"  - {row['Column']} ({row['Type']})"
                if not row['Nullable']:
                    context += " NOT NULL"
                context += "\n"
            
            # Add foreign keys
            fks = self.db_manager.get_foreign_keys(table)
            if fks:
                context += "Foreign Keys:\n"
                for fk in fks:
                    context += f"  - {fk['constrained_columns']} -> {fk['referred_table']}.{fk['referred_columns']}\n"
            
            # Add sample data
            sample_df = self.db_manager.get_sample_data(table, limit=3)
            if not sample_df.empty:
                context += f"Sample Data:\n{sample_df.to_string()}\n"
            
            context += "\n"
        
        return context
    
    def generate_sql(self, natural_language_query: str) -> Tuple[str, str]:
        """
        Convert natural language to SQL query
        
        Returns:
            Tuple of (sql_query, explanation)
        """
        db_context = self.get_database_context()
        
        prompt = f"""You are a SQL expert. Convert the following natural language question into a SQL query.

{db_context}

Natural Language Question: {natural_language_query}

Instructions:
1. Generate a valid SQL query that answers the question
2. Use proper SQL syntax for the database type
3. Include appropriate JOINs, WHERE clauses, GROUP BY, ORDER BY as needed
4. Return only valid SQL that can be executed directly
5. Provide a brief explanation of what the query does

Format your response as:
SQL:
<your sql query here>

EXPLANATION:
<brief explanation of the query>
"""
        
        try:
            response = self.llm.invoke(prompt)
            
            # Extract SQL and explanation
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            sql_part = ""
            explanation_part = ""
            
            if "SQL:" in response_text and "EXPLANATION:" in response_text:
                parts = response_text.split("EXPLANATION:")
                sql_part = parts[0].replace("SQL:", "").strip()
                explanation_part = parts[1].strip()
            else:
                sql_part = response_text.strip()
                explanation_part = "Query generated from natural language"
            
            # Clean up SQL
            sql_part = sql_part.replace("```sql", "").replace("```", "").strip()
            
            return sql_part, explanation_part
            
        except Exception as e:
            return "", f"Error generating SQL: {str(e)}"
    
    def validate_and_fix_sql(self, sql_query: str, error_message: str) -> str:
        """Use LLM to fix SQL errors"""
        
        prompt = f"""The following SQL query produced an error. Please fix it.

Original SQL:
{sql_query}

Error Message:
{error_message}

Database Context:
{self.get_database_context()}

Please provide the corrected SQL query only, without any explanation.
"""
        
        try:
            response = self.llm.invoke(prompt)
            fixed_sql = response.content if hasattr(response, 'content') else str(response)
            fixed_sql = fixed_sql.replace("```sql", "").replace("```", "").strip()
            return fixed_sql
        except:
            return sql_query
    
    def generate_insights(self, df: pd.DataFrame, query: str) -> str:
        """Generate natural language insights from query results"""
        
        if df.empty:
            return "No data returned from the query."
        
        # Create summary statistics
        summary = f"Query returned {len(df)} rows and {len(df.columns)} columns.\n\n"
        
        # Analyze numeric columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if numeric_cols:
            summary += "Key Statistics:\n"
            for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                summary += f"- {col}: "
                summary += f"Min={df[col].min():.2f}, "
                summary += f"Max={df[col].max():.2f}, "
                summary += f"Avg={df[col].mean():.2f}\n"
        
        prompt = f"""Analyze the following SQL query results and provide 2-3 key business insights.

Query: {query}

Data Summary:
{summary}

Sample Data:
{df.head(5).to_string()}

Provide concise, actionable insights that a business user would find valuable.
"""
        
        try:
            response = self.llm.invoke(prompt)
            insights = response.content if hasattr(response, 'content') else str(response)
            return insights
        except Exception as e:
            return f"Could not generate insights: {str(e)}"


class SQLQueryHistory:
    """Manages query history and analytics"""
    
    def __init__(self):
        if 'sql_history' not in st.session_state:
            st.session_state.sql_history = []
    
    def add_query(self, nl_query: str, sql_query: str, 
                  execution_time: float, success: bool, 
                  rows_returned: int = 0):
        """Add a query to history"""
        st.session_state.sql_history.append({
            'timestamp': datetime.now(),
            'nl_query': nl_query,
            'sql_query': sql_query,
            'execution_time': execution_time,
            'success': success,
            'rows_returned': rows_returned
        })
    
    def get_history(self) -> List[Dict]:
        """Get query history"""
        return st.session_state.sql_history
    
    def get_stats(self) -> Dict:
        """Get statistics from query history"""
        history = self.get_history()
        if not history:
            return {
                'total_queries': 0,
                'successful_queries': 0,
                'avg_execution_time': 0,
                'total_rows_returned': 0
            }
        
        successful = [q for q in history if q['success']]
        
        return {
            'total_queries': len(history),
            'successful_queries': len(successful),
            'avg_execution_time': sum(q['execution_time'] for q in successful) / len(successful) if successful else 0,
            'total_rows_returned': sum(q['rows_returned'] for q in successful)
        }
    
    def export_history(self) -> pd.DataFrame:
        """Export query history as DataFrame"""
        if not self.get_history():
            return pd.DataFrame()
        
        return pd.DataFrame(self.get_history())


def create_sample_database():
    """Create a sample SQLite database for demo purposes"""
    conn = sqlite3.connect('demo_database.db')
    cursor = conn.cursor()
    
    # Create customers table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS customers (
            customer_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT,
            city TEXT,
            country TEXT,
            signup_date DATE
        )
    ''')
    
    # Create orders table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS orders (
            order_id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            order_date DATE,
            total_amount DECIMAL(10,2),
            status TEXT,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        )
    ''')
    
    # Create products table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS products (
            product_id INTEGER PRIMARY KEY,
            product_name TEXT NOT NULL,
            category TEXT,
            price DECIMAL(10,2),
            stock_quantity INTEGER
        )
    ''')
    
    # Insert sample data
    customers_data = [
        (1, 'John Doe', 'john@example.com', 'New York', 'USA', '2023-01-15'),
        (2, 'Jane Smith', 'jane@example.com', 'London', 'UK', '2023-02-20'),
        (3, 'Bob Johnson', 'bob@example.com', 'Sydney', 'Australia', '2023-03-10'),
        (4, 'Alice Brown', 'alice@example.com', 'Toronto', 'Canada', '2023-04-05'),
        (5, 'Charlie Wilson', 'charlie@example.com', 'Berlin', 'Germany', '2023-05-12')
    ]
    
    cursor.executemany('INSERT OR REPLACE INTO customers VALUES (?,?,?,?,?,?)', customers_data)
    
    orders_data = [
        (1, 1, '2023-06-01', 150.00, 'Completed'),
        (2, 1, '2023-06-15', 200.50, 'Completed'),
        (3, 2, '2023-06-10', 99.99, 'Completed'),
        (4, 3, '2023-06-20', 350.00, 'Pending'),
        (5, 4, '2023-06-25', 175.25, 'Completed'),
        (6, 2, '2023-07-01', 425.00, 'Completed'),
        (7, 5, '2023-07-05', 89.99, 'Cancelled')
    ]
    
    cursor.executemany('INSERT OR REPLACE INTO orders VALUES (?,?,?,?,?)', orders_data)
    
    products_data = [
        (1, 'Laptop', 'Electronics', 999.99, 50),
        (2, 'Mouse', 'Electronics', 29.99, 200),
        (3, 'Keyboard', 'Electronics', 79.99, 150),
        (4, 'Monitor', 'Electronics', 299.99, 75),
        (5, 'Desk Chair', 'Furniture', 199.99, 30)
    ]
    
    cursor.executemany('INSERT OR REPLACE INTO products VALUES (?,?,?,?,?)', products_data)
    
    conn.commit()
    conn.close()
    
    return 'sqlite:///demo_database.db'


def render_sql_database_page():
    """Render the SQL Database interface page"""
    st.header("💾 SQL Database Query Interface")
    st.caption("Connect to your database and query it using natural language")
    st.markdown("---")
    
    # Initialize components
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = SQLDatabaseManager()
    
    if 'query_history' not in st.session_state:
        st.session_state.query_history = SQLQueryHistory()
    
    # Sidebar for database connection
    with st.sidebar:
        st.subheader("🔌 Database Connection")
        
        db_type = st.selectbox(
            "Database Type",
            ["SQLite", "PostgreSQL", "MySQL", "SQL Server"],
            key="db_type_select"
        )
        
        if db_type == "SQLite":
            use_demo = st.checkbox("Use Demo Database", value=True)
            
            if use_demo:
                if st.button("Create/Connect Demo DB"):
                    conn_string = create_sample_database()
                    if st.session_state.db_manager.connect(conn_string, "sqlite"):
                        st.success("Connected to demo database!")
                        st.rerun()
            else:
                db_file = st.text_input("Database File Path", "database.db")
                if st.button("Connect"):
                    conn_string = f"sqlite:///{db_file}"
                    if st.session_state.db_manager.connect(conn_string, "sqlite"):
                        st.success("Connected!")
                        st.rerun()
        
        elif db_type == "PostgreSQL":
            host = st.text_input("Host", "localhost")
            port = st.text_input("Port", "5432")
            database = st.text_input("Database", "mydb")
            username = st.text_input("Username", "postgres")
            password = st.text_input("Password", type="password")
            
            if st.button("Connect"):
                conn_string = f"postgresql://{username}:{password}@{host}:{port}/{database}"
                if st.session_state.db_manager.connect(conn_string, "postgresql"):
                    st.success("Connected!")
                    st.rerun()
        
        elif db_type == "MySQL":
            host = st.text_input("Host", "localhost")
            port = st.text_input("Port", "3306")
            database = st.text_input("Database", "mydb")
            username = st.text_input("Username", "root")
            password = st.text_input("Password", type="password")
            
            if st.button("Connect"):
                conn_string = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
                if st.session_state.db_manager.connect(conn_string, "mysql"):
                    st.success("Connected!")
                    st.rerun()
        
        # Database overview
        if st.session_state.db_manager.engine:
            st.markdown("---")
            st.subheader("📊 Database Overview")
            tables = st.session_state.db_manager.get_tables()
            st.metric("Total Tables", len(tables))
            
            if tables:
                with st.expander("Tables"):
                    for table in tables:
                        stats = st.session_state.db_manager.get_table_stats(table)
                        st.write(f"**{table}** ({stats['row_count']} rows)")
    
    # Main content area
    if not st.session_state.db_manager.engine:
        st.info("👈 Connect to a database using the sidebar to get started")
        
        # Show connection guide
        with st.expander("📖 Connection Guide"):
            st.markdown("""
            ### How to Connect to Your Database
            
            **SQLite (Local File)**
            - Perfect for local development and testing
            - No server required
            - Use the demo database to get started quickly
            
            **PostgreSQL**
            - Connection string format: `postgresql://user:password@host:port/database`
            - Common ports: 5432
            - Example: `postgresql://postgres:mypass@localhost:5432/mydb`
            
            **MySQL**
            - Connection string format: `mysql+pymysql://user:password@host:port/database`
            - Common ports: 3306
            - Example: `mysql+pymysql://root:mypass@localhost:3306/mydb`
            
            **Security Note:** Connection credentials are only stored in session memory and are not persisted.
            """)
        
        return
    
    # Tabs for different functionalities
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Natural Language Query",
        "📋 Database Explorer",
        "📊 Query Analytics",
        "💻 SQL Editor",
        "🔗 Schema Diagram"
    ])
    
    with tab1:
        render_nl_query_interface()
    
    with tab2:
        render_database_explorer()
    
    with tab3:
        render_query_analytics()
    
    with tab4:
        render_sql_editor()
    
    with tab5:
        render_schema_diagram()


def render_nl_query_interface():
    """Render natural language query interface"""
    st.subheader("Ask Questions in Natural Language")
    
    # Initialize NL to SQL converter
    llm = Config.get_llm()
    nl_sql = NaturalLanguageToSQL(llm, st.session_state.db_manager)
    
    # Example queries
    with st.expander("💡 Example Questions"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Basic Queries:**
            - Show me all customers from the USA
            - List all products with stock quantity less than 100
            - What are the top 5 orders by total amount?
            """)
        
        with col2:
            st.markdown("""
            **Advanced Queries:**
            - What is the average order amount by customer?
            - Show customers who have never placed an order
            - Compare sales by country for completed orders
            """)
    
    nl_query = st.text_area(
        "Your Question:",
        placeholder="e.g., Show me the total sales by country for completed orders",
        height=100
    )
    
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        execute_btn = st.button("🚀 Execute Query", type="primary", use_container_width=True)
    with col2:
        generate_insights = st.checkbox("Generate Insights", value=True)
    
    if execute_btn and nl_query:
        import time
        start_time = time.time()
        
        with st.spinner("🤖 Translating to SQL..."):
            sql_query, explanation = nl_sql.generate_sql(nl_query)
        
        if sql_query:
            st.markdown("### Generated SQL Query")
            st.code(sql_query, language="sql")
            st.info(f"📝 {explanation}")
            
            with st.spinner("⚙️ Executing query..."):
                result_df, error = st.session_state.db_manager.execute_query(sql_query)
                execution_time = time.time() - start_time
            
            if error:
                st.error(f"❌ Error: {error}")
                
                if st.button("🔧 Try to Fix SQL"):
                    with st.spinner("Fixing SQL..."):
                        fixed_sql = nl_sql.validate_and_fix_sql(sql_query, error)
                        st.code(fixed_sql, language="sql")
                        
                        result_df, error = st.session_state.db_manager.execute_query(fixed_sql)
                        if not error:
                            st.success("✅ Fixed and executed successfully!")
                            display_query_results(
                                result_df, execution_time, fixed_sql, 
                                nl_query, nl_sql if generate_insights else None
                            )
                        else:
                            st.error(f"Still failed: {error}")
                
                st.session_state.query_history.add_query(
                    nl_query, sql_query, execution_time, False, 0
                )
            else:
                display_query_results(
                    result_df, execution_time, sql_query, 
                    nl_query, nl_sql if generate_insights else None
                )
                st.session_state.query_history.add_query(
                    nl_query, sql_query, execution_time, True, len(result_df)
                )


def display_query_results(df: pd.DataFrame, execution_time: float, 
                          sql_query: str, nl_query: str,
                          nl_sql: Optional[NaturalLanguageToSQL] = None):
    """Display query results with visualizations and insights"""
    st.success(f"✅ Query executed in {execution_time:.2f}s")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows Returned", len(df))
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        st.metric("Execution Time", f"{execution_time:.2f}s")
    
    # Generate insights if requested
    if nl_sql and not df.empty:
        with st.expander("🧠 AI-Generated Insights", expanded=True):
            with st.spinner("Analyzing results..."):
                insights = nl_sql.generate_insights(df, nl_query)
                st.markdown(insights)
    
    st.markdown("### 📊 Results")
    st.dataframe(df, use_container_width=True, height=400)
    
    # Auto-visualization suggestions
    if len(df) > 0:
        st.markdown("### 📈 Visualizations")
        
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            if numeric_cols and categorical_cols:
                viz_type = st.selectbox(
                    "Chart Type",
                    ["Bar Chart", "Line Chart", "Pie Chart", "Scatter Plot", "Box Plot"]
                )
                
                if viz_type == "Bar Chart":
                    x_col = st.selectbox("X-axis (Category)", categorical_cols)
                    y_col = st.selectbox("Y-axis (Value)", numeric_cols)
                    
                    fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)
                
                elif viz_type == "Line Chart":
                    x_col = st.selectbox("X-axis", df.columns.tolist())
                    y_col = st.selectbox("Y-axis", numeric_cols)
                    
                    fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} over {x_col}", markers=True)
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)
                
                elif viz_type == "Pie Chart":
                    names_col = st.selectbox("Categories", categorical_cols)
                    values_col = st.selectbox("Values", numeric_cols)
                    
                    fig = px.pie(df, names=names_col, values=values_col, title=f"{values_col} by {names_col}")
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)
                
                elif viz_type == "Scatter Plot":
                    x_col = st.selectbox("X-axis", numeric_cols)
                    y_col = st.selectbox("Y-axis", [c for c in numeric_cols if c != x_col])
                    color_col = st.selectbox("Color by (optional)", ["None"] + categorical_cols)
                    
                    if color_col != "None":
                        fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=f"{y_col} vs {x_col}")
                    else:
                        fig = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
                    
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)
                
                elif viz_type == "Box Plot":
                    y_col = st.selectbox("Numeric Column", numeric_cols)
                    x_col = st.selectbox("Group by (optional)", ["None"] + categorical_cols)
                    
                    if x_col != "None":
                        fig = px.box(df, x=x_col, y=y_col, title=f"{y_col} Distribution by {x_col}")
                    else:
                        fig = px.box(df, y=y_col, title=f"{y_col} Distribution")
                    
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)
        
        with viz_col2:
            if numeric_cols:
                st.markdown("#### Summary Statistics")
                st.dataframe(df[numeric_cols].describe(), use_container_width=True)
    
    # Download results
    col1, col2 = st.columns(2)
    with col1:
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results (CSV)",
            data=csv,
            file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Export with SQL query
        export_data = f"-- Query executed: {datetime.now()}\n"
        export_data += f"-- Question: {nl_query}\n\n"
        export_data += f"{sql_query}\n\n"
        export_data += "-- Results:\n"
        export_data += df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download with SQL",
            data=export_data,
            file_name=f"query_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )


def render_database_explorer():
    """Render database schema explorer"""
    st.subheader("Database Schema Explorer")
    
    tables = st.session_state.db_manager.get_tables()
    
    if not tables:
        st.info("No tables found in the database")
        return
    
    selected_table = st.selectbox("Select Table", tables)
    
    if selected_table:
        stats = st.session_state.db_manager.get_table_stats(selected_table)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", f"{stats['row_count']:,}")
        with col2:
            st.metric("Total Columns", stats['column_count'])
        with col3:
            # Calculate approximate size
            sample_df = st.session_state.db_manager.get_sample_data(selected_table, 1)
            if not sample_df.empty:
                est_size = len(sample_df.to_csv()) * stats['row_count'] / 1024  # KB
                st.metric("Est. Size", f"{est_size:.1f} KB")
        
        # Schema
        st.markdown("### 📋 Schema")
        schema_df = st.session_state.db_manager.get_table_schema(selected_table)
        st.dataframe(schema_df, use_container_width=True)
        
        # Foreign Keys
        fks = st.session_state.db_manager.get_foreign_keys(selected_table)
        if fks:
            st.markdown("### 🔗 Foreign Keys")
            fk_data = []
            for fk in fks:
                fk_data.append({
                    'Column': ', '.join(fk['constrained_columns']),
                    'References': f"{fk['referred_table']}.{', '.join(fk['referred_columns'])}"
                })
            st.dataframe(pd.DataFrame(fk_data), use_container_width=True)
        
        # Indexes
        indexes = st.session_state.db_manager.get_indexes(selected_table)
        if indexes:
            st.markdown("### 📇 Indexes")
            idx_data = []
            for idx in indexes:
                idx_data.append({
                    'Name': idx['name'],
                    'Columns': ', '.join(idx['column_names']),
                    'Unique': idx.get('unique', False)
                })
            st.dataframe(pd.DataFrame(idx_data), use_container_width=True)
        
        # Sample data
        st.markdown("### 👁️ Sample Data")
        sample_size = st.slider("Number of rows", 5, 100, 10)
        sample_df = st.session_state.db_manager.get_sample_data(selected_table, sample_size)
        st.dataframe(sample_df, use_container_width=True)
        
        # Quick stats for numeric columns
        numeric_cols = sample_df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            st.markdown("### 📊 Column Statistics")
            
            # Get full table stats for selected columns
            selected_cols = st.multiselect(
                "Select columns to analyze",
                numeric_cols.tolist(),
                default=numeric_cols.tolist()[:3]
            )
            
            if selected_cols:
                query = f"SELECT {', '.join(selected_cols)} FROM {selected_table}"
                full_df, _ = st.session_state.db_manager.execute_query(query)
                
                if not full_df.empty:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.dataframe(full_df[selected_cols].describe(), use_container_width=True)
                    
                    with col2:
                        # Distribution chart
                        fig = make_subplots(
                            rows=len(selected_cols), cols=1,
                            subplot_titles=selected_cols
                        )
                        
                        for idx, col in enumerate(selected_cols, 1):
                            fig.add_trace(
                                go.Histogram(x=full_df[col], name=col, showlegend=False),
                                row=idx, col=1
                            )
                        
                        fig.update_layout(
                            height=200*len(selected_cols),
                            title_text="Distributions",
                            paper_bgcolor="rgba(0,0,0,0)"
                        )
                        st.plotly_chart(fig, use_container_width=True)


def render_query_analytics():
    """Render query history and analytics"""
    st.subheader("Query Analytics Dashboard")
    
    stats = st.session_state.query_history.get_stats()
    
    if stats['total_queries'] == 0:
        st.info("No queries executed yet. Run some queries to see analytics.")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Queries", stats['total_queries'])
    with col2:
        st.metric("Successful", stats['successful_queries'])
    with col3:
        st.metric("Avg Execution Time", f"{stats['avg_execution_time']:.2f}s")
    with col4:
        success_rate = (stats['successful_queries'] / stats['total_queries'] * 100)
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    st.markdown("---")
    
    # Export history
    if st.button("📥 Export Query History"):
        history_df = st.session_state.query_history.export_history()
        csv = history_df.to_csv(index=False)
        st.download_button(
            "Download History CSV",
            csv,
            f"query_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv"
        )
    
    # Query history
    st.markdown("### 📜 Recent Queries")
    history = st.session_state.query_history.get_history()
    
    for i, query in enumerate(reversed(history[-10:])):
        with st.expander(
            f"{'✅' if query['success'] else '❌'} {query['nl_query'][:80]}... "
            f"({query['timestamp'].strftime('%Y-%m-%d %H:%M')})"
        ):
            st.markdown(f"**Natural Language:** {query['nl_query']}")
            st.code(query['sql_query'], language="sql")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Execution Time", f"{query['execution_time']:.2f}s")
            with col2:
                st.metric("Rows Returned", query['rows_returned'])
            with col3:
                st.metric("Status", "Success" if query['success'] else "Failed")


def render_sql_editor():
    """Render direct SQL editor"""
    st.subheader("Direct SQL Editor")
    st.caption("Write and execute SQL queries directly")
    
    # SQL templates
    with st.expander("📚 SQL Templates"):
        template = st.selectbox(
            "Select a template",
            [
                "-- Select template",
                "SELECT * FROM table_name LIMIT 10;",
                "SELECT column1, COUNT(*) as count FROM table_name GROUP BY column1;",
                "SELECT t1.*, t2.* FROM table1 t1 JOIN table2 t2 ON t1.id = t2.id;",
                "SELECT * FROM table_name WHERE column_name > value ORDER BY column_name DESC;",
            ]
        )
        
        if template != "-- Select template":
            st.code(template, language="sql")
    
    sql_query = st.text_area(
        "SQL Query:",
        height=150,
        placeholder="SELECT * FROM customers WHERE country = 'USA';",
        key="direct_sql_editor"
    )
    
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        execute_btn = st.button("▶️ Execute", type="primary", use_container_width=True)
    with col2:
        explain_btn = st.button("📖 Explain Query", use_container_width=True)
    
    if explain_btn and sql_query:
        llm = Config.get_llm()
        prompt = f"""Explain the following SQL query in simple terms:

{sql_query}

Provide:
1. What the query does
2. Which tables/columns are involved
3. Any joins or filters
4. Expected output"""
        
        with st.spinner("Analyzing query..."):
            try:
                response = llm.invoke(prompt)
                explanation = response.content if hasattr(response, 'content') else str(response)
                st.info(explanation)
            except Exception as e:
                st.error(f"Could not explain query: {str(e)}")
    
    if execute_btn and sql_query:
        import time
        start_time = time.time()
        
        with st.spinner("Executing query..."):
            result_df, error = st.session_state.db_manager.execute_query(sql_query)
            execution_time = time.time() - start_time
        
        if error:
            st.error(f"❌ Error: {error}")
        else:
            display_query_results(result_df, execution_time, sql_query, "Direct SQL Query")


def render_schema_diagram():
    """Render database schema diagram"""
    st.subheader("Database Schema Diagram")
    st.caption("Visual representation of your database structure")
    
    tables = st.session_state.db_manager.get_tables()
    
    if not tables:
        st.info("No tables found in database")
        return
    
    # Create network graph of tables and relationships
    nodes = []
    edges = []
    
    for table in tables:
        stats = st.session_state.db_manager.get_table_stats(table)
        nodes.append({
            'name': table,
            'rows': stats['row_count'],
            'columns': stats['column_count']
        })
        
        # Get foreign keys
        fks = st.session_state.db_manager.get_foreign_keys(table)
        for fk in fks:
            edges.append({
                'from': table,
                'to': fk['referred_table'],
                'columns': f"{', '.join(fk['constrained_columns'])} -> {', '.join(fk['referred_columns'])}"
            })
    
    # Create visualization
    import numpy as np
    
    n = len(nodes)
    angles = [2 * np.pi * i / n for i in range(n)]
    node_x = [np.cos(angle) * 3 for angle in angles]
    node_y = [np.sin(angle) * 3 for angle in angles]
    
    # Create edge traces
    edge_traces = []
    for edge in edges:
        from_idx = [i for i, node in enumerate(nodes) if node['name'] == edge['from']][0]
        to_idx = [i for i, node in enumerate(nodes) if node['name'] == edge['to']][0]
        
        edge_trace = go.Scatter(
            x=[node_x[from_idx], node_x[to_idx], None],
            y=[node_y[from_idx], node_y[to_idx], None],
            mode='lines',
            line=dict(width=2, color='#94a3b8'),
            hoverinfo='text',
            hovertext=edge['columns'],
            showlegend=False
        )
        edge_traces.append(edge_trace)
    
    # Create node trace
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        marker=dict(
            size=[node['rows']/10 + 20 for node in nodes],  # Size based on row count
            color=['#6366f1', '#22c55e', '#f97316', '#ec4899', '#8b5cf6', '#0ea5e9'][:n],
            line=dict(width=2, color='#ffffff')
        ),
        text=[node['name'] for node in nodes],
        textposition='top center',
        hoverinfo='text',
        hovertext=[f"<b>{node['name']}</b><br>{node['rows']:,} rows<br>{node['columns']} columns" 
                   for node in nodes],
        showlegend=False
    )
    
    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace])
    
    fig.update_layout(
        title="Database Schema Relationships",
        showlegend=False,
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='#020617',
        paper_bgcolor='rgba(0,0,0,0)',
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Table details
    st.markdown("### 📊 Table Details")
    
    table_data = []
    for node in nodes:
        table_data.append({
            'Table': node['name'],
            'Rows': f"{node['rows']:,}",
            'Columns': node['columns']
        })
    
    st.dataframe(pd.DataFrame(table_data), use_container_width=True)
    
    # Relationship details
    if edges:
        st.markdown("### 🔗 Relationships")
        
        rel_data = []
        for edge in edges:
            rel_data.append({
                'From': edge['from'],
                'To': edge['to'],
                'Relationship': edge['columns']
            })
        
        st.dataframe(pd.DataFrame(rel_data), use_container_width=True)