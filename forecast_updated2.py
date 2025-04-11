import streamlit as st
import pandas as pd
import altair as alt
import subprocess
import json
import re
from datetime import datetime, timedelta
import random

# ----------------------------
# Page Config
# ----------------------------

st.set_page_config(page_title="📦 Retail AI Forecast Dashboard", layout="wide")
st.title("📦 Enhanced Retail Forecast Dashboard")

# ----------------------------
# CSV Upload
# ----------------------------

st.sidebar.header("📁 Upload Your Retail CSV")
uploaded_file = st.sidebar.file_uploader("Upload retail forecasting CSV", type=["csv"])

st.sidebar.markdown("---")
use_model = st.sidebar.checkbox("🧠 Run Local Forecast Instead")

# ----------------------------
# Helper Functions
# ----------------------------

def run_forecast_model():
    # Generate sample data instead of using Ollama
    sample_data = generate_sample_data()
    with open("forecast_output_raw.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(sample_data))
    return json.dumps(sample_data)

def generate_sample_data():
    # Generate some sample retail data
    products = ["P001", "P002", "P003", "P004", "P005"]
    stores = ["S001", "S002", "S003"]
    segments = ["Premium", "Budget", "Regular"]
    promos = ["None", "10% Off", "BOGO", "Holiday Special"]
    factors = ["Normal", "Holiday", "Weather Event", "Competitor Sale"]
    
    start_date = datetime.now() - timedelta(days=60)
    
    data = []
    for i in range(200):
        date = start_date + timedelta(days=i//4)
        for product in products:
            # Create realistic looking sales patterns
            base_sales = random.randint(50, 200)
            # Add weekly pattern
            day_factor = 1.0 + (0.3 if date.weekday() in [4, 5] else 0)
            # Add some trend
            trend_factor = 1.0 + (i/400)
            # Add some noise
            noise = random.uniform(0.8, 1.2)
            
            sales_qty = int(base_sales * day_factor * trend_factor * noise)
            
            data.append({
                "date": date.strftime("%Y-%m-%d"),
                "product id": product,
                "store id": random.choice(stores),
                "sales quantity": sales_qty,
                "price": round(random.uniform(10, 100), 2),
                "customer segments": random.choice(segments),
                "promotions": random.choice(promos),
                "external factors": random.choice(factors)
            })
    
    return data

def extract_valid_json(text):
    match = re.search(r'({.*?}|\[.*?\])', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            st.error("⚠️ Found JSON-like text, but it's invalid.")
            return None
    return None

def load_and_clean_data(df):
    df.columns = df.columns.str.strip().str.lower()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors='coerce')
    return df

# ----------------------------
# Visualizations
# ----------------------------
def visualize(df):
    st.subheader("📅 Full Retail Data Table")
    st.dataframe(df)

    # --- Time Series: Sales by Product ---
    if all(col in df.columns for col in ["date", "product id", "sales quantity"]):
        st.subheader("📈 Sales Trend by Product")
        chart = alt.Chart(df).mark_line(point=True).encode(
            x='date:T',
            y='sales quantity:Q',
            color='product id:N',
            tooltip=['date:T', 'product id:N', 'sales quantity:Q']
        ).interactive()
        st.altair_chart(chart, use_container_width=True)

    # --- Time Series: Price Trend ---
    if "price" in df.columns:
        st.subheader("💸 Price Trend Over Time")
        price_chart = alt.Chart(df).mark_line(point=True).encode(
            x='date:T',
            y='price:Q',
            color='product id:N',
            tooltip=['date:T', 'product id:N', 'price:Q']
        ).interactive()
        st.altair_chart(price_chart, use_container_width=True)

    # --- Bar: Total Sales by Product ---
    st.subheader("🧮 Total Sales by Product")
    product_summary = df.groupby("product id")["sales quantity"].sum().reset_index().sort_values(by="sales quantity", ascending=False)
    st.dataframe(product_summary)

    bar = alt.Chart(product_summary).mark_bar().encode(
        x=alt.X("product id:N", sort='-y'),
        y="sales quantity:Q",
        color="product id:N"
    )
    st.altair_chart(bar, use_container_width=True)

    # --- Bar: Sales by Customer Segments ---
    if "customer segments" in df.columns:
        st.subheader("👥 Sales by Customer Segments")
        seg = df.groupby("customer segments")["sales quantity"].sum().reset_index()
        seg_chart = alt.Chart(seg).mark_bar().encode(
            x='customer segments:N',
            y='sales quantity:Q',
            color='customer segments:N'
        )
        st.altair_chart(seg_chart, use_container_width=True)

    # --- Bar: Promotions ---
    if "promotions" in df.columns:
        st.subheader("🏷️ Sales by Promotion")
        promo = df.groupby("promotions")["sales quantity"].mean().reset_index()
        promo_chart = alt.Chart(promo).mark_bar().encode(
            x='promotions:N',
            y='sales quantity:Q',
            color='promotions:N'
        )
        st.altair_chart(promo_chart, use_container_width=True)

    # --- Store-wise Sales ---
    if "store id" in df.columns:
        st.subheader("🏬 Store-wise Sales")
        store_summary = df.groupby("store id")["sales quantity"].sum().reset_index()
        store_chart = alt.Chart(store_summary).mark_bar().encode(
            x="store id:N",
            y="sales quantity:Q",
            tooltip=["store id:N", "sales quantity:Q"],
            color="store id:N"
        )
        st.altair_chart(store_chart, use_container_width=True)

    # --- External Factor Impact ---
    if "external factors" in df.columns:
        st.subheader("🌦️ Sales by External Factors")
        ext = df.groupby("external factors")["sales quantity"].sum().reset_index()
        ext_chart = alt.Chart(ext).mark_bar().encode(
            x="external factors:N",
            y="sales quantity:Q",
            color="external factors:N"
        )
        st.altair_chart(ext_chart, use_container_width=True)

    # --- Download Button ---
    st.subheader("📥 Download Cleaned CSV")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", data=csv, file_name="retail_forecast_cleaned.csv", mime="text/csv")

# ----------------------------
# Local Simple Chatbot
# ----------------------------

# Simple rule-based chatbot that works with the dataframe
def local_chatbot(question, df):
    # Store variables that we'll use for analysis
    if df is None:
        return "Please upload data or generate sample data first."
    
    question = question.lower().strip()
    
    # Very basic NLP to identify question type and respond accordingly
    response = ""
    
    # Check for basic questions about the data
    if "highest" in question or "top" in question or "best" in question:
        if "product" in question and "sales" in question:
            top_product = df.groupby("product id")["sales quantity"].sum().sort_values(ascending=False).index[0]
            total_sales = df.groupby("product id")["sales quantity"].sum().sort_values(ascending=False).iloc[0]
            response = f"The product with highest sales is {top_product} with {total_sales} units sold."
            
        elif "store" in question:
            top_store = df.groupby("store id")["sales quantity"].sum().sort_values(ascending=False).index[0]
            store_sales = df.groupby("store id")["sales quantity"].sum().sort_values(ascending=False).iloc[0]
            response = f"The top performing store is {top_store} with {store_sales} units sold."
            
        elif "segment" in question or "customer" in question:
            if "customer segments" in df.columns:
                top_segment = df.groupby("customer segments")["sales quantity"].sum().sort_values(ascending=False).index[0]
                segment_sales = df.groupby("customer segments")["sales quantity"].sum().sort_values(ascending=False).iloc[0]
                response = f"The highest performing customer segment is {top_segment} with {segment_sales} units sold."
            
        elif "promotion" in question or "promo" in question:
            if "promotions" in df.columns:
                promo_performance = df.groupby("promotions")["sales quantity"].mean().sort_values(ascending=False)
                top_promo = promo_performance.index[0]
                promo_avg = promo_performance.iloc[0]
                response = f"The most effective promotion is '{top_promo}' with an average of {promo_avg:.2f} units sold per instance."
    
    elif "average" in question or "mean" in question:
        if "price" in question:
            avg_price = df["price"].mean()
            response = f"The average price across all products is ${avg_price:.2f}."
            
        elif "sales" in question:
            avg_sales = df["sales quantity"].mean()
            response = f"The average sales quantity per record is {avg_sales:.2f} units."
    
    elif "trend" in question or "over time" in question:
        if "sales" in question:
            trends = df.groupby(pd.Grouper(key="date", freq="W"))["sales quantity"].sum()
            direction = "increasing" if trends.iloc[-1] > trends.iloc[0] else "decreasing"
            pct_change = abs((trends.iloc[-1] - trends.iloc[0]) / trends.iloc[0] * 100)
            response = f"Sales are {direction} over time. There has been a {pct_change:.1f}% {direction} from the first week to the latest week."
    
    elif "forecast" in question or "predict" in question or "next week" in question:
        # Simple forecast - just average of last 2 weeks + 5%
        last_date = df["date"].max()
        two_weeks_ago = last_date - pd.Timedelta(days=14)
        recent_sales = df[df["date"] >= two_weeks_ago]["sales quantity"].sum()
        days = (df["date"].max() - df["date"].min()).days
        if days > 0:
            daily_avg = recent_sales / 14
            forecast = daily_avg * 7 * 1.05  # Simple 5% growth forecast
            response = f"Based on recent trends, I forecast approximately {int(forecast)} units will be sold next week."
        else:
            response = "Not enough historical data to make a forecast."
    
    elif "compare" in question:
        if "store" in question:
            store_sales = df.groupby("store id")["sales quantity"].sum()
            best_store = store_sales.idxmax()
            worst_store = store_sales.idxmin()
            response = f"Store {best_store} has the highest sales with {store_sales[best_store]} units, while store {worst_store} has the lowest with {store_sales[worst_store]} units."
            
        elif "product" in question:
            product_sales = df.groupby("product id")["sales quantity"].sum()
            best_product = product_sales.idxmax()
            worst_product = product_sales.idxmin()
            response = f"Product {best_product} is the best seller with {product_sales[best_product]} units, while product {worst_product} is the worst seller with {product_sales[worst_product]} units."
    
    elif "total" in question or "sum" in question:
        if "sales" in question:
            total_sales = df["sales quantity"].sum()
            response = f"Total sales across all products and stores is {total_sales} units."
            
        elif "revenue" in question or "income" in question:
            if "price" in df.columns:
                df["revenue"] = df["price"] * df["sales quantity"]
                total_revenue = df["revenue"].sum()
                response = f"Total revenue is ${total_revenue:,.2f}."
    
    # If no specific pattern is recognized, provide general stats
    if not response:
        # Count how many products, stores, and total sales
        num_products = df["product id"].nunique()
        total_sales = df["sales quantity"].sum()
        date_range = f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}"
        
        response = f"Your dataset contains {len(df)} records with {num_products} unique products. " + \
                   f"Total sales: {total_sales} units. Date range: {date_range}. " + \
                   "You can ask about highest selling products, sales trends, comparing stores, or forecasts."
    
    return response

# ---- Chat Interface ----
st.markdown("---")
st.subheader("🤖 Ask About Your Data")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Add predefined questions for quick selection
predefined_questions = [
    "Which product had the highest sales?",
    "What's the sales trend over time?",
    "Forecast next week's sales",
    "Compare store performance",
    "What was the total revenue?",
    "Which promotion was most effective?"
]
selected_question = st.selectbox("Choose a question:", [""] + predefined_questions)

# Text input for custom questions
chat_input = st.text_input("Or type your question about the data:")

# Use selected question if provided
if selected_question and selected_question != "":
    chat_input = selected_question

# Store the current dataframe in session state to use with chatbot
if "current_df" not in st.session_state:
    st.session_state.current_df = None

if st.button("Ask"):
    if chat_input.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Analyzing data..."):
            reply = local_chatbot(chat_input, st.session_state.current_df)
            st.session_state.chat_history.append(("You", chat_input))
            st.session_state.chat_history.append(("Agent", reply))

# Display chat history
for sender, msg in st.session_state.chat_history:
    if sender == "You":
        st.markdown(f"**🧑‍💼 You:** {msg}")
    else:
        st.markdown(f"**🤖 Agent:** {msg}")

# Clear chat button
if st.button("Clear Chat History") and st.session_state.chat_history:
    st.session_state.chat_history = []
    st.experimental_rerun()

# ----------------------------
# Load & Run
# ----------------------------

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        df = load_and_clean_data(df)
        st.session_state.current_df = df  # Store for chatbot use
        visualize(df)
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
elif use_model:
    output = run_forecast_model()
    json_data = extract_valid_json(output)
    if json_data:
        try:
            df = pd.DataFrame(json_data) if isinstance(json_data, list) else pd.DataFrame.from_dict(json_data)
            df = load_and_clean_data(df)
            st.session_state.current_df = df  # Store for chatbot use
            visualize(df)
        except Exception as e:
            st.error(f"❌ Failed to parse model output: {e}")
    else:
        st.error("❌ Could not extract valid JSON from model output.")
else:
    st.info("👈 Upload your retail dataset or check 'Run Local Forecast Instead' to generate sample data.")