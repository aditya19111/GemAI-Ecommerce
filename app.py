import streamlit as st
import streamlit as st
import google.generativeai as genai
import pandas as pd
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
import re
import altair as alt

# Load environment variables from .env file
load_dotenv()

# --- Caching Function for Enhanced Dashboard Data ---
@st.cache_data(ttl=600) # Cache the data for 10 minutes
def load_dashboard_data():
    """
    Connects to the Supabase DB and fetches data for all dashboard charts.
    """
    try:
        engine = create_engine(os.getenv("SUPABASE_URI"))
        with engine.connect() as connection:
            # 1. KPIs
            kpi_query = text("""
                SELECT
                    COUNT(DISTINCT o.order_id) AS total_orders,
                    SUM(oi.quantity * oi.price_per_unit) AS total_revenue,
                    COUNT(DISTINCT o.customer_id) AS unique_customers
                FROM orders AS o JOIN order_items AS oi ON o.order_id = oi.order_id;
            """)
            kpi_df = pd.read_sql_query(kpi_query, connection)

            # 2. Inventory by Warehouse (for Pie Chart)
            inventory_warehouse_query = text("""
                SELECT w.warehouse_location, SUM(i.quantity) as total_quantity
                FROM inventory i JOIN warehouses w ON i.warehouse_id = w.warehouse_id
                GROUP BY w.warehouse_location;
            """)
            inventory_warehouse_df = pd.read_sql_query(inventory_warehouse_query, connection)

            # 3. Top 10 Best-Selling Products by Revenue (for Horizontal Bar Chart)
            top_products_query = text("""
                SELECT
                    p.product_name,
                    SUM(oi.quantity * oi.price_per_unit) as revenue
                FROM products p
                JOIN order_items oi ON p.product_id = oi.product_id
                GROUP BY p.product_name
                ORDER BY revenue DESC
                LIMIT 10;
            """)
            top_products_df = pd.read_sql_query(top_products_query, connection)

            # 4. Revenue by Category (for Bar Chart)
            category_revenue_query = text("""
                SELECT p.category, SUM(oi.quantity * oi.price_per_unit) as revenue
                FROM products p JOIN order_items oi ON p.product_id = oi.product_id
                GROUP BY p.category ORDER BY revenue DESC;
            """)
            category_revenue_df = pd.read_sql_query(category_revenue_query, connection)
            
            # 5. Order Status Breakdown (for Donut Chart)
            order_status_query = text("""
                SELECT status, COUNT(*) as count FROM orders GROUP BY status;
            """)
            order_status_df = pd.read_sql_query(order_status_query, connection)
            
            return {
                "kpi": kpi_df,
                "inventory_warehouse": inventory_warehouse_df,
                "top_products": top_products_df,
                "category_revenue": category_revenue_df,
                "order_status": order_status_df
            }
    except Exception as e:
        st.error(f"Error loading dashboard data: {e}")
        return None

# --- Database Schema ---
DB_SCHEMA = """
CREATE TABLE customers (
    customer_id VARCHAR(255) PRIMARY KEY,
    first_name VARCHAR(255),
    last_name VARCHAR(255),
    email VARCHAR(255),
    login_count INT,
    avg_session_duration DECIMAL(5,2),
    last_session_duration DECIMAL(5,2),
    is_active BOOLEAN,
    preferred_device VARCHAR(50),
    preferred_login_time VARCHAR(50)
);
CREATE TABLE products (
  product_id TEXT PRIMARY KEY,
  product_name TEXT,
  category TEXT,
  price REAL,
  description TEXT
);
CREATE TABLE warehouses (
  warehouse_id TEXT PRIMARY KEY,
  warehouse_location TEXT
);
CREATE TABLE inventory (
  inventory_id TEXT PRIMARY KEY,
  product_id TEXT,
  warehouse_id TEXT,
  quantity INTEGER,
  reorder_level INTEGER,
  last_stock_update TIMESTAMP,
  FOREIGN KEY (product_id) REFERENCES products(product_id),
  FOREIGN KEY (warehouse_id) REFERENCES warehouses(warehouse_id)
);
CREATE TABLE orders (
  order_id TEXT PRIMARY KEY,
  customer_id TEXT,
  order_date TIMESTAMP,
  status TEXT,
  estimated_delivery_date DATE,
  actual_delivery_date DATE,
  FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
CREATE TABLE order_items (
  order_item_id INTEGER PRIMARY KEY,
  order_id TEXT,
  product_id TEXT,
  quantity INTEGER,
  price_per_unit REAL,
  FOREIGN KEY (order_id) REFERENCES orders(order_id),
  FOREIGN KEY (product_id) REFERENCES products(product_id)
);
CREATE TABLE transportation (
    shipment_id VARCHAR(255) PRIMARY KEY,
    order_id VARCHAR(255) REFERENCES orders(order_id),
    carrier VARCHAR(255),
    tracking_number VARCHAR(255),
    current_status VARCHAR(100),
    status_details TEXT,
    last_updated TIMESTAMP,
    delivery_distance_km DECIMAL(7,2),
    delivery_cost_usd DECIMAL(10,2)
);
"""

# --- AI & SQL Functions ---
def get_sql_query(user_question):
    """
    Uses Gemini Pro to convert a natural language question into an SQL query.
    """
    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel('gemini-2.5-pro')
        prompt = f"""
        You are an expert PostgreSQL database analyst. Your task is to convert a user's question into a valid and performant SQL query.
        You must only use the tables and columns provided in the schema below. All table names are lowercase.
        ### PostgreSQL Schema:
        {DB_SCHEMA}
        ### Instructions:
        1. Your final output must be ONLY the raw SQL query inside a single markdown code block.
        2. Do not include any explanations, greetings, or other conversational text.
        3. If a user asks about a specific person by name (e.g., "Larry Williams"), you MUST join the 'customers' table with the 'orders' table and use a WHERE clause to filter by 'first_name' and 'last_name'.
        ### Example:
        User Question: "How many orders have been cancelled?"
        SQL Query:
        ```sql
        SELECT COUNT(*) FROM orders WHERE status = 'Cancelled';
        ```
        ### User Question:
        "{user_question}"
        ### SQL Query:
        """
        response = model.generate_content(prompt)
        
        # Robustly parse the SQL query from the response
        sql_match = re.search(r"```sql\n(.*?)\n```", response.text, re.DOTALL)
        if sql_match:
            sql_query = sql_match.group(1).strip()
        else:
            # Fallback for cases where markdown is missing
            select_match = re.search(r"SELECT.*?;", response.text, re.DOTALL | re.IGNORECASE)
            if select_match:
                sql_query = select_match.group(0).strip()
            else:
                return None # No query found
        return sql_query
    except Exception as e:
        print(f"Error in get_sql_query: {e}") # Print error to terminal
        return None

def execute_sql_query(sql_query):
    """
    Connects to the Supabase DB and executes the given SQL query after a security check.
    """
    # Security Check
    dangerous_keywords = ['DELETE', 'UPDATE', 'INSERT', 'DROP', 'ALTER', 'TRUNCATE', 'CREATE', 'GRANT', 'REVOKE']
    query_upper = sql_query.upper()
    for keyword in dangerous_keywords:
        if f' {keyword} ' in query_upper or query_upper.startswith(keyword + ' '):
            st.error(f"Execution blocked: The generated query contains a dangerous keyword ('{keyword}'). Only SELECT queries are allowed.")
            return None
    try:
        engine = create_engine(os.getenv("SUPABASE_URI"))
        with engine.connect() as connection:
            result_df = pd.read_sql_query(text(sql_query), connection)
            return result_df
    except Exception as e:
        st.error(f"An error occurred while executing the query: {e}")
        return None

def get_natural_language_response(user_question, result_df):
    """
    Uses Gemini Pro to convert the database result into a natural language response.
    """
    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel('gemini-2.5-pro')
        data_string = result_df.to_string(index=False)
        prompt = f"""
        You are an expert data analyst. Your task is to provide a concise, insightful, and professional summary of the data provided, in response to a user's question.
        **Guidelines:**
        1. Start with a direct answer to the user's question.
        2. If the data is a single number (e.g., a total), state it clearly.
        3. If the data is a table, summarize the key findings. Do not just list the rows. For example, instead of listing all top 5 customers, say "The top 5 customers by revenue are..." and then list them.
        4. If the query returns no results, state that "No data was found for this request."
        5. Use markdown and bolding to highlight key numbers, names, or insights.
        ### User's Original Question:
        "{user_question}"
        ### Data from Database:
        ```
        {data_string}
        ```
        ### Your Professional Summary:
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"An error occurred while generating the response: {e}")
        return "Sorry, I encountered an issue while summarizing the data."


# --- Main Streamlit App ---
st.set_page_config(page_title="E-commerce AI Dashboard", layout="wide")
st.title("📈 E-commerce AI Dashboard & Assistant")
st.markdown("Live business intelligence powered by Gemini. Ask any question about your data below.")

dashboard_data = load_dashboard_data()
if dashboard_data:
    st.header("Live Business Metrics")
    kpi_df = dashboard_data["kpi"]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total Orders", value=f"{kpi_df['total_orders'].iloc[0]:,}")
    with col2:
        st.metric(label="Total Revenue", value=f"${kpi_df['total_revenue'].iloc[0]:,.2f}")
    with col3:
        st.metric(label="Unique Customers", value=f"{kpi_df['unique_customers'].iloc[0]:,}")

    st.header("Visual Insights")
    col1, col2 = st.columns(2)
    with col1:
        # Pie Chart: Inventory by Warehouse
        st.subheader("Inventory Distribution by Warehouse")
        inventory_df = dashboard_data["inventory_warehouse"]
        pie_chart = alt.Chart(inventory_df).mark_arc(outerRadius=120).encode(
            theta=alt.Theta(field="total_quantity", type="quantitative", stack=True),
            color=alt.Color(field="warehouse_location", type="nominal", title="Warehouse"),
            tooltip=["warehouse_location", "total_quantity"]
        ).properties(width=400, height=400)
        st.altair_chart(pie_chart, use_container_width=True)
        
        # Bar Chart: Revenue by Category
        st.subheader("Revenue by Product Category")
        category_df = dashboard_data["category_revenue"]
        bar_chart = alt.Chart(category_df).mark_bar().encode(
            x=alt.X('revenue:Q', title='Total Revenue ($)'),
            y=alt.Y('category:N', sort='-x', title='Product Category'),
            tooltip=['category', 'revenue']
        ).properties(width=400, height=350).interactive()
        st.altair_chart(bar_chart, use_container_width=True)

    with col2:
        # Horizontal Bar Chart: Top 10 Best-Selling Products
        st.subheader("Top 10 Best-Selling Products by Revenue")
        top_products_df = dashboard_data["top_products"]
        top_products_chart = alt.Chart(top_products_df).mark_bar().encode(
            x=alt.X('revenue:Q', title='Total Revenue ($)'),
            y=alt.Y('product_name:N', sort='-x', title='Product Name'),
            tooltip=['product_name', 'revenue']
        ).properties(width=400, height=400).interactive()
        st.altair_chart(top_products_chart, use_container_width=True)
        
        # Donut Chart: Order Status
        st.subheader("Order Status Breakdown")
        status_df = dashboard_data["order_status"]
        donut_chart = alt.Chart(status_df).mark_arc(innerRadius=80, outerRadius=120).encode(
            theta='count',
            color=alt.Color('status:N', title='Status'),
            tooltip=['status', 'count']
        ).properties(width=400, height=350)
        st.altair_chart(donut_chart, use_container_width=True)

else:
    st.warning("Could not load dashboard data. Please check database connection.")

# --- Data Guide ---
with st.expander("Explore Your Data: Schema and Examples"):
    st.markdown("""
        **To ask effective questions, it helps to know what data is available. Here is the structure of your database:**

        - **`customers`**: Customer profiles (`customer_id`, `first_name`, `last_name`, `email`, `login_count`, `avg_session_duration`, `is_active`, `preferred_device`).
        - **`products`**: Product catalog (`product_id`, `product_name`, `category`, `price`).
        - **`orders`**: Records of each sale (`order_id`, `customer_id`, `order_date`, `status`).
        - **`order_items`**: Links products to orders (`order_id`, `product_id`, `quantity`).
        - **`inventory`**: Tracks stock levels (`product_id`, `warehouse_id`, `quantity`).
        - **`warehouses`**: Location of warehouses (`warehouse_id`, `warehouse_location`).
        - **`transportation`**: Shipping details (`order_id`, `carrier`, `current_status`, `delivery_distance_km`, `delivery_cost_usd`).

        **Example Questions using new data:**
        - `Show me the top 10 most active customers by login count.`
        - `What is the average delivery cost for orders shipped by FedEx?`
        - `Which customers prefer to use a 'Mobile' device?`
        - `What is the total delivery distance for all completed orders?`
    """)

# --- Chatbot Interface ---
st.header("Ask a Custom Question")
if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
if prompt := st.chat_input("Your question here... e.g., 'What was our total revenue last month?'"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Analyzing your question and querying the database..."):
            sql_query = get_sql_query(prompt)
            if sql_query:
                with st.expander("View Generated SQL Query"):
                    st.code(sql_query, language="sql")
                result_df = execute_sql_query(sql_query)
                if result_df is not None:
                    with st.expander("View Raw Data"):
                        st.dataframe(result_df)
                    if result_df.empty:
                        final_response = "The query returned no results. Please check your question or the data."
                    else:
                        final_response = get_natural_language_response(prompt, result_df)
                    st.markdown(final_response)
                    st.session_state.messages.append({"role": "assistant", "content": final_response})
            else:
                final_response = "I couldn't generate an SQL query for your question. Please try rephrasing it."
                st.error(final_response)
                st.session_state.messages.append({"role": "assistant", "content": final_response})
