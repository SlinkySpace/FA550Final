=======================================================
FA550 CAPSTONE PROJECT
Jordan Ela
BTC Futures Microstructure and Kalshi Contract Price Behavior Dashboard
=======================================================

PROJECT ACCESS

Streamlit app:
https://fa550capstonejela.streamlit.app/

GitHub repository:
https://github.com/SlinkySpace/FA550Final

PROJECT OVERVIEW

This project is an interactive Streamlit dashboard for studying BTC futures microstructure and short-dated Kalshi BTC binary contract price behavior. The dashboard visualizes how Kalshi YES/NO prices relate to BTC futures quote movement, threshold distance, time-to-close, spread, local volatility, and trading activity.

The project is not a trading strategy. It is a visual analytics dashboard focused on market behavior, contract sensitivity, and dashboard-based interpretation.

DATA SOURCES

The Kalshi side uses Kalshi BTC contract and trade data, including contract tickers, YES/NO prices, trade sizes, open and close times, and contract metadata.

The BTC futures side uses CME BTC futures data pulled through Databento, including trades, TBBO quote data, and OHLCV bars. The final dashboard uses BTC futures quote midpoint as the consistent underlying reference:

quote_mid = (best_bid + best_ask) / 2

The app uses precomputed weekly dashboard files so it can load one selected week at a time instead of processing the full high-frequency dataset interactively.

FOLDER STRUCTURE

ai_logs/
Representative AI conversation logs used during project development.

app/
Main Streamlit app used for the deployed dashboard.

project/
Copy of the final Streamlit app for final submission package compatibility.

code/
Supporting scripts used for inspection, data checking, preprocessing, and dashboard file validation.

data/
Cleaned and dashboard-ready data files. The main app data is stored in weekly folders under data/app/weeks/. See data/README_data.md.

docs/
User guide, screenshots, diagrams, and other documentation materials.

reports/
Final project documentation PDF and supporting report files.

HOW TO RUN THE DASHBOARD LOCALLY

1. Open PowerShell or a VS Code terminal.

2. Navigate to the project folder:

   cd "C:\Users\jorda\OneDrive\Desktop\FA550_BTC_Kalshi_Final"

3. Install dependencies:

   pip install -r requirements.txt

4. Run the Streamlit app:

   streamlit run .\app\app.py

Alternative submission-copy path:

   streamlit run .\project\app.py

DASHBOARD FEATURES

Overview:
Shows selected-week summary metrics, contract outcome distribution, BTC futures quote midpoint path, and decision-time distribution.

Contract Evolution:
Shows one selected Kalshi contract through time, including YES/NO prices, BTC futures quote midpoint, decision marker, close marker, and short-horizon YES price movement distributions.

Futures Microstructure:
Shows futures-side quote midpoint, spread, trade activity, and local 60-second volatility.

Decision Time:
Shows how many minutes before close contracts became effectively decided.

Threshold Risk:
Shows distance-from-threshold and moneyness-style views for understanding when contracts are near the decision boundary.

Sensitivity:
Shows the relationship between short-horizon BTC futures quote movement and short-horizon Kalshi YES price movement.

Data Preview:
Provides audit tables showing which weekly files and columns are being loaded.

DOCUMENTATION

Complete project documentation:
reports/Project_Documentation.pdf

User guide:
docs/User_Guide.md

Presentation video:
Presentation_Video_Link.txt

NOTES FOR GRADER

The dashboard intentionally loads one week at a time for performance. Earlier versions attempted broader date-range selection, but that made the app slower and easier to overload.

The final dashboard uses BTC futures quote midpoint as the single underlying reference because it is available consistently across the dashboard data range. Actual futures trade prices were tested but removed because the available actual-trade file did not cover the full dashboard range.

AI USAGE

AI tools were used to assist with debugging, project organization, dashboard design, performance troubleshooting, report revision, and interpretation of binary-contract moneyness. Representative AI logs are included in ai_logs/.
