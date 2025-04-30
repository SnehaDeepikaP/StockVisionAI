# StockVision AI - Agentic Retail Optimization Platform

**StockVision AI** is an AI-powered agentic platform designed to optimize retail operations, including **demand forecasting**, **inventory monitoring**, and **pricing strategies**. Using a multi-agent architecture and powered by Ollama's LLaMA 3.2 model, StockVision AI enables retailers to make data-driven decisions by forecasting demand, adjusting stock levels, and optimizing prices based on historical trends and competition.

---

## Features

- 🧠 **Multi-Agent System**:
  - **Demand Forecasting Agent**: Predicts future product demand using historical sales data.
  - **Inventory Monitoring Agent**: Tracks stock levels, detects low stock or overstock situations, and suggests reorder actions.
  - **Pricing Optimization Agent**: Adjusts product pricing based on competitor analysis, elasticity, and sales volume.

- 📊 **Real-Time Dashboards**:
  - Visualize demand forecasts, stock levels, and pricing optimization strategies through interactive charts.

- 🧠 **LLM-Powered Reasoning**:
  - Uses **locally hosted LLaMA 3.2 via Ollama** for intelligent, real-time decision-making.

- 🔄 **Custom Dataset Upload**:
  - Upload CSVs for demand, inventory, and pricing data to tailor the predictions and optimizations.

- 📈 **Interactive Chat Interface**:
  - Chat with the system to ask about stock levels, forecasted demand, and optimized pricing.

- 🛠 **Responsible AI**:
  - Transparent decision-making, with clear reasoning paths and human-in-the-loop control.

- 🧠 **Flexible Integration**:
  - Compatible with external data sources or APIs for seamless integration into existing retail systems.

---

## Prerequisites

- **Python**: Version 3.10 or 3.11 (recommended for compatibility).
- **Ollama**: A local instance running the LLaMA 3.2 model for natural language processing.
- **Dependencies**: Listed in the Installation section.
- **Optional**: External data integration for demand, inventory, and pricing (or use sample datasets).

---
## Components
**Data Sources**

**demand_forecasting.csv:** Historical sales data for demand prediction
**inventory_monitoring.csv:** Current inventory levels and historical stock movements
**pricing_optimization.csv:** Price points, competitor data, and sales performance metrics

**Processing Modules**

**forecast_updated2.py:** Implements time series forecasting models
**inventory_updated2.py:** Handles inventory analytics and optimization
**pricing_updated2.py:** Contains pricing strategy algorithms
**retail_dashboard_m3.py:** Main user interface integrating all components
## Installation

**Clone the Repository:**
```bash
git clone https://github.com/SnehaDeepikaP/StockVisionAI.git
cd StockVisionAI
```
**Install dependencies:**
```bash
pip install -r requirements.txt
```
**Run the dashboard:**
```bash
streamlit run retail_dashboard_m3.py
```

## Testing
Test predefined queries (e.g., “Show me upcoming stockouts” or “What is the forecast for next month?”).

Upload sample datasets for demand, inventory, and pricing to see how the agents react.

Use the interactive chat to simulate real-time queries and check for accurate responses.

Test the integration with external APIs or data sources.

## Project StructureStockVisionAI/
```
│
├── Datasets/
│   ├── demand_forecasting.csv      # Historical sales data
│   ├── inventory_monitoring.csv    # Inventory levels and movements
│   └── pricing_optimization.csv    # Price points and performance metrics
│
├── forecast_updated2.py            # Demand forecasting module
├── inventory_updated2.py           # Inventory management module
├── pricing_updated2.py             # Pricing strategy algorithms
├── retail_dashboard_m3.py          # Main Streamlit dashboard
├── requirements.txt                # Project dependencies
├── LICENSE                         # License information
└── README.md                       # Project documentation
```
## Contributing
Contributions are welcome! Please follow these steps:

1.Fork the repository.

2.Create a new branch (git checkout -b feature/your-feature).

3.Make your changes and commit (git commit -m "Add your feature").

4.Push to the branch (git push origin feature/your-feature).

5.Open a pull request.

6.Please ensure your code follows PEP 8 style guidelines and includes appropriate tests.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
