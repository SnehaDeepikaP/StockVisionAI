import streamlit as st
import subprocess
import json
import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import os
import base64
from datetime import datetime, timedelta
import numpy as np
from io import StringIO
import warnings

# Ignore specific Streamlit warning
warnings.filterwarnings("ignore", message=".missing ScriptRunContext.")

# ------------------- Utility Functions -------------------
def extract_valid_json(text):
    # Look for JSON arrays with square brackets
    matches = re.findall(r'(\[.*?\])', text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
            
    # Also try looking for JSON objects with curly braces
    matches = re.findall(r'(\{.*?\})', text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    # Try to extract JSON with more flexible pattern
    try:
        # Find text that looks like JSON (between braces or brackets)
        json_pattern = re.compile(r'(?s)(?P<json>(\{.*\})|(\[.*\]))')
        match = json_pattern.search(text)
        if match:
            json_str = match.group('json')
            # Clean up the JSON string (remove markdown code blocks, etc.)
            json_str = re.sub(r'```(json)?', '', json_str)
            json_str = json_str.strip()
            return json.loads(json_str)
    except:
        pass
            
    return None

def create_sample_data(agent_type):
    """Create appropriate sample data based on agent type"""
    if agent_type == "forecasting":
        # Generate synthetic time series data
        dates = pd.date_range(start='2023-01-01', periods=90)
        base_sales = 1000
        trend = np.linspace(0, 200, 90)  # Upward trend
        seasonality = 100 * np.sin(np.linspace(0, 6*np.pi, 90))  # Seasonal pattern
        noise = np.random.normal(0, 50, 90)  # Random noise
        
        sales = base_sales + trend + seasonality + noise
        sales = sales.round(0).astype(int)
        
        df = pd.DataFrame({
            'date': dates.strftime('%Y-%m-%d'),
            'sales': sales
        })
        return df
        
    elif agent_type == "inventory":
        products = [f"Product {chr(65+i)}" for i in range(10)]
        stock_levels = np.random.randint(5, 100, 10)
        reorder_points = np.random.randint(20, 50, 10)
        lead_times = np.random.randint(3, 14, 10)
        
        df = pd.DataFrame({
            'product': products,
            'stock_level': stock_levels,
            'reorder_point': reorder_points,
            'lead_time': lead_times
        })
        return df
        
    elif agent_type == "pricing":
        products = [f"Product {chr(65+i)}" for i in range(10)]
        costs = np.random.randint(5, 50, 10)
        margins = np.random.uniform(0.3, 0.6, 10)
        current_prices = (costs * (1 + margins)).round(2)
        
        # Competitor prices vary around current price
        competitor_prices = (current_prices * np.random.uniform(0.9, 1.1, 10)).round(2)
        
        df = pd.DataFrame({
            'product': products,
            'cost': costs,
            'current_price': current_prices,
            'competitor_price': competitor_prices,
            'sales_volume': np.random.randint(50, 500, 10)
        })
        return df
        
    return None

def get_table_download_link(df, filename="data.csv", text="Download CSV"):
    """Generates a link allowing the data in a given pandas dataframe to be downloaded"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Create directory for outputs if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# ------------------- Streamlit Dashboard -------------------
st.set_page_config(page_title="StockVisionAI", layout="wide")

# Initialize session state for settings persistence
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
    
if 'advanced_mode' not in st.session_state:
    st.session_state.advanced_mode = False

# Custom CSS for better styling and dark mode
def get_css():
    dark_bg = "#0e1117" if st.session_state.dark_mode else "#ffffff"
    dark_text = "#ffffff" if st.session_state.dark_mode else "#0e1117"
    dark_sidebar = "#262730" if st.session_state.dark_mode else "#f0f2f6"
    button_bg = "#1e3a8a" if st.session_state.dark_mode else "#0ea5e9"
    button_hover = "#1e40af" if st.session_state.dark_mode else "#0284c7"
    box_bg = "#1a3d5c" if st.session_state.dark_mode else "#e6f3ff"
    
    return f"""
    <style>
    .app-header {{
        padding: 1.5rem;
        background: linear-gradient(90deg, {button_bg}, {button_hover});
        border-radius: 0.5rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}
    .app-title {{
        font-size: 3rem;
        font-weight: 800;
        color: white;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }}
    .app-subtitle {{
        font-size: 1.2rem;
        color: rgba(255,255,255,0.9);
        margin-top: 0.5rem;
    }}
    .main-header {{
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: {dark_text};
        text-align: center;
    }}
    .sub-header {{
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: {dark_text};
    }}
    .agent-header {{
        background-color: {dark_sidebar};
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        color: {dark_text};
    }}
    .status-box {{
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        background-color: {box_bg};
        color: {dark_text};
    }}
    .dashboard-card {{
        background-color: {dark_sidebar};
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
    }}
    .dashboard-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }}
    .card-title {{
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: {dark_text};
    }}
    .card-desc {{
        flex-grow: 1;
        color: {dark_text};
        opacity: 0.9;
    }}
    .card-button {{
        background-color: {button_bg};
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
        text-align: center;
        margin-top: 1rem;
        font-weight: 500;
        cursor: pointer;
        transition: background-color 0.3s ease;
        text-decoration: none;
    }}
    .card-button:hover {{
        background-color: {button_hover};
    }}
    .stApp {{
        background-color: {dark_bg};
        color: {dark_text};
    }}
    .css-1gvp5h0 {{  /* This targets the sidebar */
        background-color: {dark_sidebar};
    }}
    .css-18e3th9 {{  /* This targets the main content area */
        background-color: {dark_bg};
    }}
    .st-ee {{
        background-color: {dark_sidebar};
    }}
    .feature-icon {{
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }}
    .footer {{
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(128,128,128,0.2);
    }}
    .contact-section {{
        background-color: {dark_sidebar};
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }}
    .contact-title {{
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: {dark_text};
    }}
    .contact-item {{
        display: flex;
        align-items: center;
        margin-bottom: 0.5rem;
        color: {dark_text};
    }}
    .contact-icon {{
        margin-right: 0.5rem;
        font-size: 1.2rem;
    }}
    </style>
    """

st.markdown(get_css(), unsafe_allow_html=True)

# New custom header with app name and subtitle
st.markdown("""
<div class="app-header">
    <h1 class="app-title">🤖 StockVisionAI</h1>
    <p class="app-subtitle">Advanced Retail Intelligence Solutions</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    st.header("Navigation")
    
    # Replace radio button with separate navigation buttons
    st.write("Choose Dashboard:")
    
    # Create navigation buttons with custom styling
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Forecasting", key="forecast_nav", use_container_width=True):
            os.system("streamlit run forecast_updated.py")
            st.components.v1.html(js)
    
    with col2:
        if st.button("📦 Inventory", key="inventory_nav", use_container_width=True):
            os.system("streamlit run inventory_updated.py")
            st.components.v1.html(js)
    
    col3, _ = st.columns(2)
    
    with col3:
        if st.button("💰 Pricing", key="pricing_nav", use_container_width=True):
            os.system("streamlit run pricing_updated.py")
            st.components.v1.html(js)
    
    st.header("Model Settings")
    model = st.selectbox("Choose LLM", ["llama3.2", "mistral", "gemma", "phi3"], index=0)
    
    if st.session_state.advanced_mode:
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1, 
                              help="Higher values make output more random, lower values more deterministic")
        max_tokens = st.slider("Max Tokens", 500, 4000, 2000, 100, 
                             help="Maximum length of model response")
    else:
        temperature = 0.7
        max_tokens = 2000
    
    st.header("Interface Settings")
    dark_mode = st.checkbox("Dark Mode", value=st.session_state.dark_mode)
    if dark_mode != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_mode
        st.rerun()
        
    advanced_mode = st.checkbox("Advanced Mode", value=st.session_state.advanced_mode)
    if advanced_mode != st.session_state.advanced_mode:
        st.session_state.advanced_mode = advanced_mode
        st.rerun()
    
    st.header("Example Files")
    st.write("Generate sample data to test dashboards:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Forecasting Data", key="forecast_sample"):
            sample_df = create_sample_data("forecasting")
            if sample_df is not None:
                csv = sample_df.to_csv(index=False)
                st.session_state.forecast_csv = csv
                st.download_button(
                    "⬇️ Download",
                    data=csv,
                    file_name="forecast_sample.csv",
                    mime="text/csv"
                )
    
    with col2:
        if st.button("Inventory Data", key="inventory_sample"):
            sample_df = create_sample_data("inventory")
            if sample_df is not None:
                csv = sample_df.to_csv(index=False)
                st.session_state.inventory_csv = csv
                st.download_button(
                    "⬇️ Download",
                    data=csv,
                    file_name="inventory_sample.csv",
                    mime="text/csv"
                )
    
    col3, _ = st.columns(2)
    
    with col3:
        if st.button("Pricing Data", key="pricing_sample"):
            sample_df = create_sample_data("pricing")
            if sample_df is not None:
                csv = sample_df.to_csv(index=False)
                st.session_state.pricing_csv = csv
                st.download_button(
                    "⬇️ Download",
                    data=csv,
                    file_name="pricing_sample.csv",
                    mime="text/csv"
                )

# Main content area - Welcome screen
st.markdown('<p class="sub-header">Welcome to Retail Intelligence Hub</p>', unsafe_allow_html=True)

# Create dashboard cards with visually appealing design
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="dashboard-card">
        <div class="feature-icon">📊</div>
        <div class="card-title">Demand Forecasting</div>
        <div class="card-desc">
            Predict future sales based on historical data. Apply advanced time series forecasting with customizable parameters for seasonality and trend analysis.
            <ul>
                <li>Time series forecasting with confidence intervals</li>
                <li>Seasonal pattern detection</li>
                <li>Trend analysis and anomaly detection</li>
            </ul>
    """, unsafe_allow_html=True)
    if st.button("🚀 Launch Demand Forecasting Dashboard"):
        os.system("streamlit run forecast_updated2.py")

with col2:
    st.markdown("""
    <div class="dashboard-card">
        <div class="feature-icon">📦</div>
        <div class="card-title">Inventory Monitoring</div>
        <div class="card-desc">
            Optimize inventory levels and receive intelligent reordering recommendations. Identify critical stock items and prevent stockouts proactively.
            <ul>
                <li>Stock level classification (Critical, Low, Optimal, Excess)</li>
                <li>Reorder quantity calculations</li>
                <li>Days-until-stockout predictions</li>
            </ul>
    """, unsafe_allow_html=True)
    if st.button("🚀 Launch Inventory management Dashboard"):
        os.system("streamlit run inventory_updated2.py")

col3, _ = st.columns(2)

with col3:
    st.markdown("""
    <div class="dashboard-card">
        <div class="feature-icon">💰</div>
        <div class="card-title">Pricing Optimization</div>
        <div class="card-desc">
            Maximize revenue with intelligent pricing strategies. Account for costs, competition, elasticity, and market position to set optimal prices.
            <ul>
                <li>Price elasticity analysis</li>
                <li>Competitive pricing insights</li>
                <li>Revenue impact simulation</li>
            </ul>
    """, unsafe_allow_html=True)
    if st.button("🚀 Launch Pricing Oprimiization Dashboard"):
        os.system("streamlit run pricing_updated2.py")

# System overview section
st.markdown('<p class="sub-header">System Overview</p>', unsafe_allow_html=True)

# Create tabs for different sections of documentation
tab1, tab2, tab3 = st.tabs(["Getting Started", "Technical Documentation", "Troubleshooting"])

with tab1:
    st.subheader("Getting Started")
    st.write("""
    Welcome to the Retail Intelligence AI Dashboard! This tool helps retail businesses make data-driven decisions using advanced AI.
    
    **Quick Start Guide:**
    1. Select one of the dashboard options above based on your analysis needs
    2. Upload a CSV file with the relevant data (or use our sample data)
    3. Adjust the parameters to customize your analysis
    4. Run the analysis and explore the visualizations
    5. Download results or reports for your records
    """)
    
    st.info("💡 **Tip:** Start with the sample data to understand the required format for each analysis type.")
    
    # Features overview
    st.subheader("Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **AI-Powered Analytics**
        - Local LLM integration
        - Advanced pattern recognition
        - Natural language insights
        """)
    
    with col2:
        st.markdown("""
        **Interactive Visualizations**
        - Time series forecasts
        - Inventory status dashboards
        - Pricing comparisons
        """)
    
    with col3:
        st.markdown("""
        **Data Management**
        - CSV import/export
        - Sample data generation
        - Report downloads
        """)

with tab2:
    st.subheader("Technical Documentation")
    
    # Data format specs
    st.write("### Data Requirements")
    
    data_format_tabs = st.tabs(["Forecasting Data", "Inventory Data", "Pricing Data"])
    
    with data_format_tabs[0]:
        st.write("**Forecasting Data Format:** Time series data with date and sales columns")
        st.code("""
date,sales
2023-01-01,1056
2023-01-02,1124
2023-01-03,982
...
        """)
        st.write("**Optional columns:** promotions, seasonality_factor, external_factors")
    
    with data_format_tabs[1]:
        st.write("**Inventory Data Format:** Product inventory data with stock levels and reorder information")
        st.code("""
product,stock_level,reorder_point,lead_time
Product A,45,30,7
Product B,12,25,5
Product C,78,40,10
...
        """)
        st.write("**Optional columns:** stockout_frequency, warehouse_capacity, supplier_reliability")
    
    with data_format_tabs[2]:
        st.write("**Pricing Data Format:** Product pricing data with costs and competitive information")
        st.code("""
product,cost,current_price,competitor_price,sales_volume
Product A,35,59.99,64.99,324
Product B,22,39.99,37.99,512
Product C,47,89.99,92.99,176
...
        """)
        st.write("**Optional columns:** elasticity_index, discount_history, customer_segment")
    
    # Model capabilities
    st.write("### Model Capabilities")
    st.write("""
    This dashboard leverages local language models through Ollama to perform advanced retail analytics:
    
    - **llama3.2**: Best balance of capability and speed
    - **mistral**: Excellent reasoning and pattern recognition
    - **gemma**: Good for concise analytics and rapid responses
    - **phi3**: Specialized for numerical analysis
    
    The models perform complex analytics by processing tabular data and generating insights in structured JSON format, which is then parsed into interactive visualizations.
    """)

with tab3:
    st.subheader("Troubleshooting")
    
    # Common issues and solutions
    st.write("### Common Issues")
    
    issues = [
        {
            "issue": "The model generates invalid JSON",
            "solution": "Try lowering the temperature parameter in Advanced Mode or switch to a different model. Lower temperatures produce more deterministic and structured outputs."
        },
        {
            "issue": "Analysis takes too long or times out",
            "solution": "Check your internet connection and ensure Ollama is running properly. You can also try a smaller dataset or adjust the max tokens parameter."
        },
        {
            "issue": "Visualizations don't appear",
            "solution": "Verify your data format matches the requirements or try using the sample data first. Ensure all required columns are present."
        },
        {
            "issue": "Browser redirects aren't working",
            "solution": "Try using a different browser or check if you have any script blocking extensions active. Refresh the page and try again."
        }
    ]
    
    for issue in issues:
        with st.expander(f"**Issue:** {issue['issue']}"):
            st.write(f"**Solution:** {issue['solution']}")
    
    # System requirements
    st.write("### System Requirements")
    st.write("""
    - **Ollama**: Must be installed and running locally
    - **Python 3.8+**: With required packages (streamlit, pandas, plotly, etc.)
    - **Memory**: 8GB RAM minimum, 16GB recommended for large datasets
    - **Storage**: 10GB free disk space for model files
    """)
    
    # Contact support
    st.write("### Need More Help?")
    st.write("Contact support at support@example.com or visit our documentation site.")


# Add footer with contact section
st.markdown("---")
st.markdown('<div class="footer">', unsafe_allow_html=True)

contact_col1, contact_col2 = st.columns(2)

with contact_col1:
    st.markdown('<div class="contact-section">', unsafe_allow_html=True)
    st.markdown('<div class="contact-title">📞 Contact Us</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="contact-item">
        <span class="contact-icon">✉️</span> Email-I: psnehadeepika2006@gmail.com
    </div>
    <div class="contact-item">
        <span class="contact-icon">✉️</span> Email-II: venkatamahalakshmigogineni@gmail.com
    </div>
    <div class="contact-item">
        <span class="contact-icon">🌐</span> Website: www.stockvisionai.com
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with contact_col2:
    st.markdown('<div class="contact-section">', unsafe_allow_html=True)
    st.markdown('<div class="contact-title">🔔 Stay Connected</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="contact-item">
        <span class="contact-icon">📋</span> Subscribe to our newsletter
    </div>
    <div class="contact-item">
        <span class="contact-icon">📱</span> Follow us on social media
    </div>
    <div class="contact-item">
        <span class="contact-icon">🎓</span> Check out our knowledge base
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
st.markdown("© 2025 Retail Intelligence AI | Powered by Ollama + LLMs | v2.4.1")

# Add JS to make card links work properly with Streamlit
st.markdown("""
<script>
document.addEventListener('DOMContentLoaded', function() {
    const cardButtons = document.querySelectorAll('.card-button');
    cardButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            window.parent.location.href = this.getAttribute('href');
        });
    });
});
</script>
""", unsafe_allow_html=True)