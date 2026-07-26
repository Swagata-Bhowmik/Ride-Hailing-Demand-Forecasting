"""Forecasting model families.

One module per family (Holt-Winters, SARIMA/SARIMAX, VAR/VARMAX, Prophet,
XGBoost, LSTM/GRU), all sharing the common `Forecaster` interface so the
Evaluation_Framework can score them uniformly.
"""
