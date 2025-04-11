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

st.set_page_config(page_title="📦 Inventory Monitoring Dashboard", layout="wide")
st.title("📦 Enhanced Inventory Monitoring Dashboard")

# ----------------------------
# CSV Upload
# ----------------------------

st.sidebar.header("📁 Upload Inventory CSV")
uploaded_file = st.sidebar.file_uploader("Upload inventory CSV", type=["csv"])

st.sidebar.markdown("---")
use_model = st.sidebar.checkbox("🧠 Run Local Inventory Model")

# ----------------------------
# Helper Functions
# ----------------------------

def run_inventory_model():
    # Generate sample inventory data instead of using Ollama
    sample_data = generate_sample_data()
    with open("inventory_output_raw.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(sample_data))
    return json.dumps(sample_data)

def generate_sample_data():
    # Generate some sample inventory data
    products = ["PR001", "PR002", "PR003", "PR004", "PR005", "PR006"]
    suppliers = ["Supplier A", "Supplier B", "Supplier C"]
    warehouses = ["Main Warehouse", "North Facility", "South Facility"]
    
    # Current date
    now = datetime.now()
    
    data = []
    for product in products:
        # Create realistic looking inventory patterns
        stock = random.randint(50, 500)
        capacity = random.randint(600, 1000)
        lead_time = random.randint(3, 30)
        reorder_point = random.randint(80, 150)
        stockout = round(random.uniform(0, 0.3), 2)  # As a percentage/frequency
        
        # Generate expiry dates (some products within 30 days, some further out)
        if random.random() < 0.7:  # 70% of products have expiry dates
            if random.random() < 0.3:  # 30% of those are expiring soon
                expiry = (now + timedelta(days=random.randint(5, 30))).strftime("%Y-%m-%d")
            else:
                expiry = (now + timedelta(days=random.randint(90, 365))).strftime("%Y-%m-%d")
        else:
            expiry = None
            
        data.append({
            "product id": product,
            "product name": f"Product {product[2:]}",
            "supplier": random.choice(suppliers),
            "warehouse": random.choice(warehouses),
            "stock levels": stock,
            "warehouse capacity": capacity,
            "supplier lead time (days)": lead_time,
            "reorder point": reorder_point,
            "stockout frequency": stockout,
            "expiry date": expiry,
            "last restocked": (now - timedelta(days=random.randint(1, 45))).strftime("%Y-%m-%d"),
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
    if "expiry date" in df.columns:
        df["expiry date"] = pd.to_datetime(df["expiry date"], errors='coerce')
    if "last restocked" in df.columns:
        df["last restocked"] = pd.to_datetime(df["last restocked"], errors='coerce')
    return df

# ----------------------------
# Visualizations
# ----------------------------

def visualize(df):
    st.subheader("📋 Full Inventory Table")
    st.dataframe(df)

    # Bar: Stock Levels by Product
    st.subheader("📦 Stock Levels by Product")
    stock_chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("product id:N", sort='-y'),
        y="stock levels:Q",
        color="product id:N"
    )
    st.altair_chart(stock_chart, use_container_width=True)

    # Line: Supplier Lead Time
    st.subheader("🚚 Supplier Lead Time by Product")
    lead_time_chart = alt.Chart(df).mark_line(point=True).encode(
        x="product id:N",
        y="supplier lead time (days):Q",
        tooltip=["product id:N", "supplier lead time (days):Q"],
        color="product id:N"
    ).interactive()
    st.altair_chart(lead_time_chart, use_container_width=True)

    # Bar: Stockout Frequency
    st.subheader("❗ Stockout Frequency by Product")
    stockout_chart = alt.Chart(df).mark_bar().encode(
        x="product id:N",
        y="stockout frequency:Q",
        color="product id:N"
    )
    st.altair_chart(stockout_chart, use_container_width=True)
    
    # --- Expiry Date Overview ---
    if "expiry date" in df.columns:
        st.subheader("🧪 Expiry Timeline")
        valid_expiry = df.dropna(subset=["expiry date"])
        if not valid_expiry.empty:
            expiry_chart = alt.Chart(valid_expiry).mark_bar().encode(
                x=alt.X("expiry date:T", title="Expiry Date"),
                y=alt.Y("stock levels:Q", title="Stock Levels"),
                color="product id:N",
                tooltip=["product id:N", "expiry date:T", "stock levels:Q"]
            )
            st.altair_chart(expiry_chart, use_container_width=True)
        else:
            st.warning("No valid expiry dates found in dataset.")

    # Reorder Point Analysis
    if "reorder point" in df.columns and "stock levels" in df.columns:
        st.subheader("🔄 Stock Levels vs. Reorder Points")
        df["needs reordering"] = df["stock levels"] <= df["reorder point"]
        reorder_chart = alt.Chart(df).mark_bar().encode(
            x="product id:N",
            y="stock levels:Q",
            color=alt.condition(
                alt.datum.needs_reordering == True,
                alt.value('red'),  # The true branch
                alt.value('green')  # The false branch
            ),
            tooltip=["product id:N", "stock levels:Q", "reorder point:Q"]
        )
        # Add a rule mark for the reorder point
        reorder_rule = alt.Chart(df).mark_rule(color='blue').encode(
            x="product id:N",
            y="reorder point:Q",
            tooltip=["product id:N", "reorder point:Q"]
        )
        st.altair_chart(reorder_chart + reorder_rule, use_container_width=True)

    # Warehouse Capacity Usage
    st.subheader("🏢 Warehouse Capacity vs. Stock")
    df["capacity usage %"] = (df["stock levels"] / df["warehouse capacity"]) * 100
    cap_chart = alt.Chart(df).mark_bar().encode(
        x="product id:N",
        y="capacity usage %:Q",
        color="product id:N"
    )
    st.altair_chart(cap_chart, use_container_width=True)

    # Download Button
    st.subheader("📥 Download Cleaned CSV")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", data=csv, file_name="inventory_cleaned.csv", mime="text/csv")

# ----------------------------
# Local Simple Chatbot for Inventory
# ----------------------------

def inventory_chatbot(question, df):
    """Simple rule-based chatbot that analyzes inventory data"""
    if df is None:
        return "Please upload inventory data or generate sample data first."
    
    question = question.lower().strip()
    response = ""
    
    # Get current date for expiry calculations
    now = datetime.now()
    
    # Handle various inventory-related questions
    if "low stock" in question or "below reorder" in question:
        if "reorder point" in df.columns:
            low_stock = df[df["stock levels"] <= df["reorder point"]]
            if not low_stock.empty:
                products = low_stock["product id"].tolist()
                levels = low_stock["stock levels"].tolist()
                info = [f"{p} ({l} units)" for p, l in zip(products, levels)]
                response = f"These products are below reorder point: {', '.join(info)}"
            else:
                response = "No products are currently below their reorder points."
        else:
            response = "Reorder point data is not available in this dataset."
            
    elif "expiring soon" in question or "expiry" in question:
        if "expiry date" in df.columns:
            # Find products expiring in the next 30 days
            df_with_expiry = df.dropna(subset=["expiry date"])
            if not df_with_expiry.empty:
                thirty_days = now + timedelta(days=30)
                expiring_soon = df_with_expiry[df_with_expiry["expiry date"] <= thirty_days]
                
                if not expiring_soon.empty:
                    products = []
                    for _, row in expiring_soon.iterrows():
                        days_left = (row["expiry date"] - now).days
                        products.append(f"{row['product id']} ({days_left} days left)")
                    response = f"Products expiring within 30 days: {', '.join(products)}"
                else:
                    response = "No products are expiring within the next 30 days."
            else:
                response = "No expiry date information is available in this dataset."
        else:
            response = "Expiry date information is not available in this dataset."
            
    elif "highest stock" in question or "most stock" in question:
        highest_stock = df.loc[df["stock levels"].idxmax()]
        response = f"The product with the highest stock level is {highest_stock['product id']} with {highest_stock['stock levels']} units."
    
    elif "lowest stock" in question or "least stock" in question:
        lowest_stock = df.loc[df["stock levels"].idxmin()]
        response = f"The product with the lowest stock level is {lowest_stock['product id']} with {lowest_stock['stock levels']} units."
        
    elif "stockout" in question or "out of stock" in question:
        if "stockout frequency" in df.columns:
            highest_stockout = df.loc[df["stockout frequency"].idxmax()]
            response = f"The product with the highest stockout frequency is {highest_stockout['product id']} with a frequency of {highest_stockout['stockout frequency']:.2f}."
        else:
            response = "Stockout frequency information is not available in this dataset."
    
    elif "lead time" in question:
        if "supplier lead time (days)" in df.columns:
            avg_lead = df["supplier lead time (days)"].mean()
            max_lead = df.loc[df["supplier lead time (days)"].idxmax()]
            response = f"The average lead time is {avg_lead:.1f} days. The product with the longest lead time is {max_lead['product id']} with {max_lead['supplier lead time (days)']} days."
        else:
            response = "Lead time information is not available in this dataset."
            
    elif "capacity" in question or "warehouse" in question:
        if "warehouse capacity" in df.columns and "stock levels" in df.columns:
            df["usage_pct"] = (df["stock levels"] / df["warehouse capacity"]) * 100
            avg_usage = df["usage_pct"].mean()
            max_usage = df.loc[df["usage_pct"].idxmax()]
            response = f"Average warehouse capacity usage is {avg_usage:.1f}%. The product using the most capacity is {max_usage['product id']} at {max_usage['usage_pct']:.1f}% of its allocated capacity."
        else:
            response = "Warehouse capacity information is not available in this dataset."
    
    elif "total stock" in question or "overall inventory" in question:
        total = df["stock levels"].sum()
        response = f"Total inventory across all products is {total} units."
        
    elif "last restocked" in question or "recent restock" in question:
        if "last restocked" in df.columns:
            df_with_dates = df.dropna(subset=["last restocked"])
            if not df_with_dates.empty:
                most_recent = df_with_dates.loc[df_with_dates["last restocked"].idxmax()]
                days_ago = (now - most_recent["last restocked"]).days
                response = f"The most recently restocked product is {most_recent['product id']}, restocked {days_ago} days ago."
            else:
                response = "No restock date information is available in this dataset."
        else:
            response = "Restock date information is not available in this dataset."
            
    elif "recommend" in question or "suggestion" in question or "what should" in question:
        # Make recommendations based on current inventory status
        recommendations = []
        
        # Check for products below reorder point
        if "reorder point" in df.columns:
            reorder_needed = df[df["stock levels"] <= df["reorder point"]]
            if not reorder_needed.empty:
                products = reorder_needed["product id"].tolist()
                recommendations.append(f"Place orders for: {', '.join(products)}")
        
        # Check for expiring products
        if "expiry date" in df.columns:
            df_with_expiry = df.dropna(subset=["expiry date"])
            if not df_with_expiry.empty:
                thirty_days = now + timedelta(days=30)
                expiring_soon = df_with_expiry[df_with_expiry["expiry date"] <= thirty_days]
                if not expiring_soon.empty:
                    products = expiring_soon["product id"].tolist()
                    recommendations.append(f"Monitor or discount soon-expiring products: {', '.join(products)}")
        
        # Check for high stockout frequency products
        if "stockout frequency" in df.columns:
            high_stockout = df[df["stockout frequency"] > 0.1]  # More than 10% stockout rate
            if not high_stockout.empty:
                products = high_stockout["product id"].tolist()
                recommendations.append(f"Increase safety stock for high-stockout products: {', '.join(products)}")
        
        # Check for warehouse capacity issues
        if "warehouse capacity" in df.columns and "stock levels" in df.columns:
            df["usage_pct"] = (df["stock levels"] / df["warehouse capacity"]) * 100
            high_capacity = df[df["usage_pct"] > 80]  # Using more than 80% of capacity
            if not high_capacity.empty:
                products = high_capacity["product id"].tolist()
                recommendations.append(f"Review warehouse allocation for high-usage products: {', '.join(products)}")
        
        if recommendations:
            response = "Recommendations based on current inventory status:\n• " + "\n• ".join(recommendations)
        else:
            response = "Based on the available data, no specific inventory actions are needed at this time."
    
    # If no specific pattern is recognized, provide general inventory stats
    if not response:
        total_products = len(df)
        total_stock = df["stock levels"].sum()
        avg_stock = df["stock levels"].mean()
        
        if "reorder point" in df.columns:
            below_reorder = df[df["stock levels"] <= df["reorder point"]]["product id"].count()
            reorder_info = f"Products below reorder point: {below_reorder}."
        else:
            reorder_info = ""
            
        response = f"Your inventory contains {total_products} products with a total of {total_stock} units (avg: {avg_stock:.1f} per product). {reorder_info} You can ask about low stock, expiring products, stockouts, lead times, or get inventory recommendations."
    
    return response

# ---- Chat Interface ----
st.markdown("---")
st.subheader("🤖 Ask About Your Inventory")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Add predefined questions for quick selection
predefined_questions = [
    "Which products are below reorder point?",
    "Which products are expiring soon?",
    "What product has the highest stockout frequency?",
    "What are the warehouse capacity usage levels?",
    "What is the longest supplier lead time?",
    "What inventory actions do you recommend?",
    "What's the total inventory count?"
]
selected_question = st.selectbox("Choose a question:", [""] + predefined_questions)

# Text input for custom questions
chat_input = st.text_input("Or type your question about inventory:")

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
        with st.spinner("Analyzing inventory data..."):
            reply = inventory_chatbot(chat_input, st.session_state.current_df)
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
    output = run_inventory_model()
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
    st.info("👈 Upload your inventory dataset or check 'Run Local Inventory Model' to generate sample data.")