# End-to-End Sales Forecasting & Demand Intelligence System

**Author:** Karanpal Singh Ranawat
**Project:** End-to-End Sales Forecasting & Demand Intelligence System

## Overview
An end-to-end retail sales forecasting and demand intelligence system built on the
Superstore Sales dataset (2015–2018). The system performs time series decomposition,
compares three forecasting approaches (SARIMA, Prophet, XGBoost), detects sales
anomalies using two independent methods, segments products by demand behavior via
K-Means clustering, and exposes everything through a live interactive Streamlit dashboard.

## Contents
- `analysis.ipynb`: Full analysis notebook (Tasks 1 to 6)
- `train.csv`: Superstore Sales dataset
- `videogamesales.csv`: Supplementary dataset (multi-source merge practice)
- `app.py`: Streamlit dashboard (Task 7)
- `requirements.txt`: Python dependencies
- `summary.docx`: Executive business report (Task 8)
- `charts/`: Exported chart images

## Live Dashboard
https://superstore-demand-intelligence.streamlit.app

## Setup
```bash
pip install -r requirements.txt
jupyter notebook analysis.ipynb
streamlit run app.py
```
