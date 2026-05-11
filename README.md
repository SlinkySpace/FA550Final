# FA550 BTC Futures and Kalshi Contract Behavior Dashboard

This project is an interactive Streamlit dashboard for analyzing BTC futures microstructure and short-dated Kalshi BTC binary contract price behavior.

The dashboard focuses on price behavior, threshold distance, decision timing, sensitivity, and futures-side market conditions. It is not a trading strategy.

## Project Structure

- app/ - Streamlit dashboard
- scripts/ - data cleaning, inspection, and weekly build scripts
- data/app/weeks/ - precomputed weekly dashboard files used by the app
- docs/ - project documentation, user guide, and screenshots
- ai_logs/ - representative AI usage logs
- submission/ - final submission notes

## How to Run

From the project root:

pip install -r requirements.txt
streamlit run .\app\app.py

## Data Notes

The dashboard uses precomputed weekly files stored in data/app/weeks/.

The full raw BTC futures and Kalshi datasets are not stored in this GitHub folder because the app is designed to run from the weekly processed dashboard files.

The cleaning and weekly build scripts included in scripts/ document how the working data was prepared.

## Main Dashboard Topics

- BTC futures microstructure
- Kalshi BTC contract evolution
- Decision time before close
- Threshold distance and binary moneyness
- Contract sensitivity to futures moves
- Weekly contract behavior summaries

## AI Usage

Representative AI logs are included in ai_logs/. AI was used for debugging, visualization design, code organization, metric interpretation, and documentation support.
