import streamlit as st
import pandas as pd
import altair as alt
import subprocess
import json
import re
import random
from datetime import datetime, timedelta

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="💰 Pricing Optimization Dashboard", layout="wide")
st.title("💰 Smart Pricing Optimization Dashboard")

# ----------------------------
# CSV Upload or Sample Data
# ----------------------------
st.sidebar.header("📁 Upload Pricing CSV")
uploaded_file = st.sidebar.file_uploader("Upload pricing_optimization.csv", type=["csv"])

st.sidebar.markdown("---")
use_sample_data = st.sidebar.checkbox("🧠 Generate Sample Pricing Data")

# ----------------------------
# Sample Data Generation
# ----------------------------
def generate_sample_data():
    products = ["P1001", "P1002", "P1003", "P1004", "P1005", "P1006", "P1007"]
    stores = ["Store A", "Store B", "Store C"]
    categories = ["Electronics", "Clothing", "Home Goods", "Food", "Beauty"]
    
    now = datetime.now()
    
    data = []
    for product in products:
        # Create realistic pricing patterns
        price = round(random.uniform(10.0, 200.0), 2)
        comp_price = price * random.uniform(0.85, 1.15)  # Competitor price within ±15%
        margin = round(random.uniform(0.20, 0.45), 2)  # 20-45% margin
        elasticity = round(random.uniform(0.5, 2.5), 2)  # Price elasticity
        sales_volume = random.randint(100, 5000)
        discount = round(random.uniform(0, 0.3), 2)  # 0-30% discount
        return_rate = round(random.uniform(0.01, 0.15), 2)  # 1-15% return rate
        storage_cost = round(random.uniform(1.0, 15.0), 2)
        
        # Last price change date
        price_change_date = (now - timedelta(days=random.randint(1, 90))).strftime("%Y-%m-%d")
        
        data.append({
            "product id": product,
            "product name": f"Product {product[1:]}",
            "category": random.choice(categories),
            "store id": random.choice(stores),
            "price": price,
            "competitor prices": round(comp_price, 2),
            "profit margin": margin,
            "elasticity index": elasticity,
            "sales volume": sales_volume,
            "discounts": discount,
            "return rate (%)": return_rate * 100,
            "storage cost": storage_cost,
            "last price change": price_change_date,
            "recommended price": round(price * random.uniform(0.9, 1.1), 2)  # ±10% of current price
        })
    
    return data

def run_pricing_optimizer():
    # Generate sample pricing data
    sample_data = generate_sample_data()
    with open("pricing_output_raw.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(sample_data))
    return json.dumps(sample_data)

def extract_valid_json(text):
    match = re.search(r'({.*?}|\[.*?\])', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            st.error("⚠️ Found JSON-like text, but it's invalid.")
            return None
    return None

# ----------------------------
# Data Cleaning
# ----------------------------
def load_and_clean_data(df):
    df.columns = df.columns.str.strip().str.lower()
    
    # Convert date columns if present
    if "last price change" in df.columns:
        df["last price change"] = pd.to_datetime(df["last price change"], errors='coerce')
    
    # Ensure numeric columns are numeric
    numeric_cols = ["price", "competitor prices", "profit margin", "elasticity index", 
                   "sales volume", "discounts", "return rate (%)", "storage cost",
                   "recommended price"]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

# ----------------------------
# Visualizations
# ----------------------------
def visualize(df):
    # Main KPIs
    st.subheader("📊 Pricing Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_margin = df["profit margin"].mean() if "profit margin" in df.columns else 0
        st.metric("Average Margin", f"{avg_margin:.1%}")
    
    with col2:
        avg_price = df["price"].mean() if "price" in df.columns else 0
        st.metric("Average Price", f"${avg_price:.2f}")
    
    with col3:
        price_vs_comp = ((df["price"] / df["competitor prices"]) - 1).mean() if all(col in df.columns for col in ["price", "competitor prices"]) else 0
        st.metric("Price vs Competition", f"{price_vs_comp:.1%}")
    
    with col4:
        total_sales = df["sales volume"].sum() if "sales volume" in df.columns else 0
        st.metric("Total Sales Volume", f"{total_sales:,}")
    
    # Full data table
    st.subheader("📋 Full Pricing Table")
    st.dataframe(df)

    # ---- Price vs Competitor ----
    if all(col in df.columns for col in ["price", "competitor prices"]):
        st.subheader("📈 Price vs Competitor Price")
        
        # Create price comparison ratio
        df["price_ratio"] = df["price"] / df["competitor prices"]
        
        price_chart = alt.Chart(df).mark_circle(size=80).encode(
            x=alt.X("competitor prices:Q", title="Competitor Price ($)"),
            y=alt.X("price:Q", title="Our Price ($)"),
            color=alt.Color("price_ratio:Q", 
                           scale=alt.Scale(scheme='redblue', domain=[0.8, 1.2]),
                           title="Price Ratio"),
            tooltip=["product id:N", "product name:N", "price:Q", "competitor prices:Q", 
                    alt.Tooltip("price_ratio:Q", title="Price Ratio", format='.2f')]
        ).interactive()
        
        # Reference line for equal prices
        reference_line = alt.Chart(pd.DataFrame({'x': [0, 1000], 'y': [0, 1000]})).mark_line(
            color='gray', strokeDash=[3, 3]
        ).encode(x='x:Q', y='y:Q')
        
        st.altair_chart((price_chart + reference_line).properties(height=400), use_container_width=True)

    # ---- Discounts vs Sales Volume ----
    if all(col in df.columns for col in ["discounts", "sales volume"]):
        st.subheader("🎯 Discounts vs Sales Volume")
        
        # Convert discount to percentage for display
        df_display = df.copy()
        df_display["discount_pct"] = df_display["discounts"] * 100
        
        disc_chart = alt.Chart(df_display).mark_circle(size=80).encode(
            x=alt.X("discount_pct:Q", title="Discount (%)"),
            y=alt.Y("sales volume:Q", title="Sales Volume"),
            color="category:N" if "category" in df.columns else "product id:N",
            size=alt.Size("price:Q", title="Price ($)"),
            tooltip=["product id:N", "product name:N", 
                    alt.Tooltip("discount_pct:Q", title="Discount %", format='.1f'), 
                    "sales volume:Q", "price:Q"]
        ).interactive()
        
        st.altair_chart(disc_chart.properties(height=400), use_container_width=True)

    # ---- Price Elasticity Analysis ----
    if "elasticity index" in df.columns:
        st.subheader("📉 Price Elasticity by Product")
        
        # Sort by elasticity for better visualization
        df_sorted = df.sort_values("elasticity index")
        
        elast_chart = alt.Chart(df_sorted).mark_bar().encode(
            x=alt.X("product id:N", sort=None, title="Product"),
            y=alt.Y("elasticity index:Q", title="Elasticity Index"),
            color=alt.Color("elasticity index:Q", 
                           scale=alt.Scale(scheme='redyellowgreen', domain=[2.5, 0.5]),
                           title="Elasticity"),
            tooltip=["product id:N", "product name:N", "elasticity index:Q"]
        )
        
        # Add a reference line for neutral elasticity (1.0)
        rule = alt.Chart(pd.DataFrame({'y': [1.0]})).mark_rule(color='red').encode(y='y:Q')
        
        st.altair_chart((elast_chart + rule).properties(height=400), use_container_width=True)
        
        # Interpretation guide
        with st.expander("Understanding Price Elasticity"):
            st.markdown("""
            **Price Elasticity Interpretation:**
            - **Below 1.0**: Less elastic. Price changes have minimal impact on demand.
            - **Equal to 1.0**: Unit elastic. Percent change in demand equals percent change in price.
            - **Above 1.0**: More elastic. Small price changes significantly impact demand.
            
            Products with high elasticity (green bars) are more price-sensitive, while products with low elasticity (red bars) can sustain price increases with less impact on demand.
            """)

    # ---- Profit Margin Analysis ----
    if "profit margin" in df.columns:
        st.subheader("💵 Profit Margin Analysis")
        
        # Convert margin to percentage for display
        df_display = df.copy()
        df_display["margin_pct"] = df_display["profit margin"] * 100
        
        margin_chart = alt.Chart(df_display).mark_bar().encode(
            x=alt.X("product id:N", sort='-y', title="Product"),
            y=alt.Y("margin_pct:Q", title="Profit Margin (%)"),
            color=alt.Color("margin_pct:Q", scale=alt.Scale(scheme='viridis'), title="Margin %"),
            tooltip=["product id:N", "product name:N", 
                     alt.Tooltip("margin_pct:Q", title="Margin %", format='.1f'),
                     "price:Q"]
        )
        
        # Add average line
        avg_margin = df_display["margin_pct"].mean()
        avg_rule = alt.Chart(pd.DataFrame({'y': [avg_margin]})).mark_rule(
            color='red', strokeDash=[3, 3]
        ).encode(y='y:Q')
        
        st.altair_chart((margin_chart + avg_rule).properties(height=400), use_container_width=True)

    # ---- Return Rate by Product ----
    if "return rate (%)" in df.columns:
        st.subheader("📦 Return Rate by Product")
        ret_chart = alt.Chart(df).mark_bar().encode(
            x=alt.X("product id:N", sort='-y', title="Product"),
            y=alt.Y("return rate (%):Q", title="Return Rate (%)"),
            color=alt.Color("return rate (%):Q", scale=alt.Scale(scheme='redblue', domain=[15, 0]), title="Return %"),
            tooltip=["product id:N", "product name:N", "return rate (%):Q", "price:Q"]
        )
        
        # Add average line
        avg_return = df["return rate (%)"].mean()
        avg_rule = alt.Chart(pd.DataFrame({'y': [avg_return]})).mark_rule(
            color='black', strokeDash=[3, 3]
        ).encode(y='y:Q')
        
        st.altair_chart((ret_chart + avg_rule).properties(height=400), use_container_width=True)

    # ---- Price Recommendations ----
    if all(col in df.columns for col in ["price", "recommended price"]):
        st.subheader("💡 Price Recommendations")
        
        # Calculate the price difference percentage
        df["price_diff_pct"] = ((df["recommended price"] / df["price"]) - 1) * 100
        
        # Create a chart that shows current price vs recommended price
        price_recom_data = pd.melt(
            df, 
            id_vars=["product id", "product name"], 
            value_vars=["price", "recommended price"],
            var_name="price_type", 
            value_name="price_value"
        )
        
        recom_chart = alt.Chart(price_recom_data).mark_bar().encode(
            x=alt.X("product id:N", title="Product"),
            y=alt.Y("price_value:Q", title="Price ($)"),
            color=alt.Color("price_type:N", 
                           scale=alt.Scale(domain=["price", "recommended price"],
                                          range=["#5276A7", "#57A44C"]),
                           title="Price Type"),
            tooltip=["product id:N", "product name:N", "price_type:N", "price_value:Q"]
        ).properties(height=400)
        
        st.altair_chart(recom_chart, use_container_width=True)
        
        # Create a table showing the recommendations
        st.subheader("Price Change Recommendations")
        
        price_changes = df[["product id", "product name", "price", "recommended price", "price_diff_pct"]].copy()
        price_changes["price_diff_pct"] = price_changes["price_diff_pct"].round(2)
        
        def highlight_price_changes(s):
            return ['background-color: #d6efc7' if val > 0 else 
                    'background-color: #f7c8c8' if val < 0 else 
                    '' for val in s]
        
        st.dataframe(price_changes.style.apply(
            highlight_price_changes, subset=['price_diff_pct']
        ))

    # ---- Download Button ----
    st.subheader("📥 Download Pricing Data")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", data=csv, file_name="pricing_optimized.csv", mime="text/csv")

# ----------------------------
# Chat Agent for Pricing
# ----------------------------
def pricing_chatbot(question, df):
    """Rule-based pricing analysis chatbot"""
    if df is None:
        return "Please upload pricing data or generate sample data first."
    
    question = question.lower().strip()
    response = ""
    
    # Handle various pricing-related questions
    if "highest margin" in question or "best margin" in question:
        if "profit margin" in df.columns:
            highest_margin = df.loc[df["profit margin"].idxmax()]
            response = f"The product with the highest profit margin is {highest_margin['product id']} ({highest_margin['product name']}) with a margin of {highest_margin['profit margin']:.1%}."
        else:
            response = "Profit margin data is not available in this dataset."
            
    elif "lowest margin" in question:
        if "profit margin" in df.columns:
            lowest_margin = df.loc[df["profit margin"].idxmin()]
            response = f"The product with the lowest profit margin is {lowest_margin['product id']} ({lowest_margin['product name']}) with a margin of {lowest_margin['profit margin']:.1%}."
        else:
            response = "Profit margin data is not available in this dataset."
    
    elif "high price" in question or "highest price" in question or "most expensive" in question:
        highest_price = df.loc[df["price"].idxmax()]
        response = f"The highest priced product is {highest_price['product id']} ({highest_price['product name']}) at ${highest_price['price']:.2f}."
    
    elif "low price" in question or "lowest price" in question or "cheapest" in question:
        lowest_price = df.loc[df["price"].idxmin()]
        response = f"The lowest priced product is {lowest_price['product id']} ({lowest_price['product name']}) at ${lowest_price['price']:.2f}."
        
    elif "elasticity" in question:
        if "elasticity index" in df.columns:
            highest_elasticity = df.loc[df["elasticity index"].idxmax()]
            lowest_elasticity = df.loc[df["elasticity index"].idxmin()]
            avg_elasticity = df["elasticity index"].mean()
            
            response = f"Average elasticity is {avg_elasticity:.2f}. The most elastic product is {highest_elasticity['product id']} with {highest_elasticity['elasticity index']:.2f}, and the least elastic is {lowest_elasticity['product id']} with {lowest_elasticity['elasticity index']:.2f}."
            
            # Add interpretation
            if highest_elasticity["elasticity index"] > 1:
                response += f" {highest_elasticity['product id']} is highly sensitive to price changes."
            if lowest_elasticity["elasticity index"] < 1:
                response += f" {lowest_elasticity['product id']} can likely sustain price increases with minimal impact on demand."
        else:
            response = "Elasticity data is not available in this dataset."
    
    elif "competitor" in question or "competition" in question:
        if all(col in df.columns for col in ["price", "competitor prices"]):
            df["price_diff"] = df["price"] - df["competitor prices"]
            df["price_diff_pct"] = (df["price"] / df["competitor prices"] - 1) * 100
            
            avg_diff = df["price_diff_pct"].mean()
            
            # Find products with significant price differences
            much_higher = df[df["price_diff_pct"] > 10]
            much_lower = df[df["price_diff_pct"] < -10]
            
            response = f"On average, our prices are {avg_diff:.1f}% compared to competitors."
            
            if not much_higher.empty:
                products = [f"{p} ({d:.1f}%)" for p, d in zip(much_higher["product id"], much_higher["price_diff_pct"])]
                response += f" Products significantly higher than competitors: {', '.join(products)}."
                
            if not much_lower.empty:
                products = [f"{p} ({d:.1f}%)" for p, d in zip(much_lower["product id"], much_lower["price_diff_pct"])]
                response += f" Products significantly lower than competitors: {', '.join(products)}."
        else:
            response = "Competitor price data is not available in this dataset."
            
    elif "return" in question:
        if "return rate (%)" in df.columns:
            highest_return = df.loc[df["return rate (%)"].idxmax()]
            avg_return = df["return rate (%)"].mean()
            
            response = f"Average return rate is {avg_return:.1f}%. The product with the highest return rate is {highest_return['product id']} ({highest_return['product name']}) with {highest_return['return rate (%)']:.1f}%."
            
            # Check for correlation with price
            if "price" in df.columns:
                corr = df["price"].corr(df["return rate (%)"])
                if abs(corr) > 0.5:
                    direction = "positive" if corr > 0 else "negative"
                    response += f" There appears to be a {direction} correlation ({corr:.2f}) between price and return rate."
        else:
            response = "Return rate data is not available in this dataset."
            
    elif "sales volume" in question or "best seller" in question:
        if "sales volume" in df.columns:
            best_seller = df.loc[df["sales volume"].idxmax()]
            worst_seller = df.loc[df["sales volume"].idxmin()]
            
            response = f"The best-selling product is {best_seller['product id']} ({best_seller['product name']}) with {best_seller['sales volume']} units. The poorest-selling product is {worst_seller['product id']} with {worst_seller['sales volume']} units."
        else:
            response = "Sales volume data is not available in this dataset."
            
    elif ("recommend" in question or "suggestion" in question or "what should" in question or 
          "price change" in question or "adjust price" in question):
        
        recommendations = []
        
        # Check for price recommendations
        if all(col in df.columns for col in ["price", "recommended price"]):
            df["price_diff_pct"] = ((df["recommended price"] / df["price"]) - 1) * 100
            
            increase_recs = df[df["price_diff_pct"] > 5]  # >5% price increase recommended
            decrease_recs = df[df["price_diff_pct"] < -5]  # >5% price decrease recommended
            
            if not increase_recs.empty:
                products = [f"{p} (+{d:.1f}%)" for p, d in zip(increase_recs["product id"], increase_recs["price_diff_pct"])]
                recommendations.append(f"Consider increasing prices for: {', '.join(products)}")
                
            if not decrease_recs.empty:
                products = [f"{p} ({d:.1f}%)" for p, d in zip(decrease_recs["product id"], decrease_recs["price_diff_pct"])]
                recommendations.append(f"Consider decreasing prices for: {', '.join(products)}")
        
        # Check for competitive positioning
        if all(col in df.columns for col in ["price", "competitor prices"]):
            df["price_diff_pct"] = ((df["price"] / df["competitor prices"]) - 1) * 100
            
            much_higher = df[df["price_diff_pct"] > 15]  # >15% higher than competitors
            
            if not much_higher.empty and "elasticity index" in df.columns:
                elastic_and_expensive = much_higher[much_higher["elasticity index"] > 1.5]
                if not elastic_and_expensive.empty:
                    products = elastic_and_expensive["product id"].tolist()
                    recommendations.append(f"Review pricing of elastic products that are much higher than competitors: {', '.join(products)}")
        
        # Check for elasticity-based recommendations
        if "elasticity index" in df.columns and "profit margin" in df.columns:
            # Elastic products with high margins could potentially increase sales with lower prices
            elastic_high_margin = df[(df["elasticity index"] > 1.5) & (df["profit margin"] > 0.3)]
            if not elastic_high_margin.empty:
                products = elastic_high_margin["product id"].tolist()
                recommendations.append(f"Consider strategic price reductions for elastic, high-margin products: {', '.join(products)}")
            
            # Inelastic products with low margins could increase price without much volume impact
            inelastic_low_margin = df[(df["elasticity index"] < 0.8) & (df["profit margin"] < 0.25)]
            if not inelastic_low_margin.empty:
                products = inelastic_low_margin["product id"].tolist()
                recommendations.append(f"Consider price increases for inelastic, low-margin products: {', '.join(products)}")
        
        # Check for return rate-based recommendations
        if "return rate (%)" in df.columns and "price" in df.columns:
            high_return_high_price = df[(df["return rate (%)"] > 10) & (df["price"] > df["price"].mean())]
            if not high_return_high_price.empty:
                products = high_return_high_price["product id"].tolist()
                recommendations.append(f"Review pricing for high-priced items with high return rates: {', '.join(products)}")
        
        if recommendations:
            response = "Pricing recommendations based on analysis:\n• " + "\n• ".join(recommendations)
        else:
            response = "Based on the available data, no specific pricing actions are recommended at this time."
    
    # If no specific pattern is recognized, provide general pricing analysis
    if not response:
        avg_price = df["price"].mean()
        price_range = f"${df['price'].min():.2f} to ${df['price'].max():.2f}"
        
        if "profit margin" in df.columns:
            avg_margin = df["profit margin"].mean()
            margin_info = f"The average profit margin is {avg_margin:.1%}."
        else:
            margin_info = ""
            
        if "elasticity index" in df.columns:
            avg_elast = df["elasticity index"].mean()
            elasticity_info = f"The average price elasticity is {avg_elast:.2f}."
        else:
            elasticity_info = ""
            
        response = f"Your pricing data includes {len(df)} products with an average price of ${avg_price:.2f} (range: {price_range}). {margin_info} {elasticity_info} You can ask about margins, elasticity, competitive pricing, returns, sales volumes, or get pricing recommendations."
    
    return response

# ---- Chat Interface ----
st.markdown("---")
st.subheader("🤖 Ask Your Pricing Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Add predefined questions for quick selection
predefined_questions = [
    "What products have the highest profit margins?",
    "How do our prices compare to competitors?",
    "Which products are most elastic to price changes?",
    "What are our best-selling products?",
    "Which products have high return rates?",
    "What pricing changes do you recommend?",
    "Which products are priced highest relative to competitors?"
]
selected_question = st.selectbox("Choose a question:", [""] + predefined_questions)

# Text input for custom questions
chat_input = st.text_input("Or type your question about pricing:")

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
        with st.spinner("Analyzing pricing data..."):
            reply = pricing_chatbot(chat_input, st.session_state.current_df)
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
elif use_sample_data:
    output = run_pricing_optimizer()
    json_data = extract_valid_json(output)
    if json_data:
        try:
            df = pd.DataFrame(json_data) if isinstance(json_data, list) else pd.DataFrame.from_dict(json_data)
            df = load_and_clean_data(df)
            st.session_state.current_df = df  # Store for chatbot use
            visualize(df)
        except Exception as e:
            st.error(f"❌ Failed to parse sample data: {e}")
    else:
        st.error("❌ Could not generate valid sample data.")
else:
    st.info("👈 Upload your pricing dataset or check 'Generate Sample Pricing Data' to see a demonstration.")