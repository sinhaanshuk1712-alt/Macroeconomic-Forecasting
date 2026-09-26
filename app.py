import streamlit as st
import wbgapi as wb
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import plotly.express as px
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(
    page_title="Macroeconomic Time-Series Forecasting",
    page_icon="⚙️",
    layout="wide"
)

# Available Countries
COUNTRY_MAP = {
    "Japan": "JPN", "South Asia (Region)": "SAS", "India": "IND",
    "United States": "USA", "China": "CHN", "Germany": "DEU",
    "United Kingdom": "GBR", "South Korea": "KOR", 
    "Sub-Saharan Africa": "SSF", "World": "WLD"
}

# --- Data Fetching ---
@st.cache_data(show_spinner=False)
def load_world_bank_data(country_codes):
    time_range = range(1990, 2025)
    df_pop = wb.data.DataFrame('SP.POP.1564.TO.ZS', country_codes, time=time_range, numericTimeKeys=True)
    df_gdp = wb.data.DataFrame('NY.GDP.PCAP.KD', country_codes, time=time_range, numericTimeKeys=True)

    pop_tidy = df_pop.T.stack().reset_index()
    pop_tidy.columns = ['Year', 'Country_Code', 'Working_Age_Pct']
    gdp_tidy = df_gdp.T.stack().reset_index()
    gdp_tidy.columns = ['Year', 'Country_Code', 'GDP_Per_Capita']

    merged = pd.merge(pop_tidy, gdp_tidy, on=['Year', 'Country_Code'])
    merged['Year'] = pd.to_numeric(merged['Year'])
    merged['Log_GDP'] = np.log(merged['GDP_Per_Capita'])
    code_to_name = {v: k for k, v in COUNTRY_MAP.items()}
    merged['Country'] = merged['Country_Code'].map(code_to_name)
    merged.dropna(subset=['Working_Age_Pct', 'GDP_Per_Capita'], inplace=True)
    
    return merged

# --- Sidebar Controls ---
st.sidebar.title("System Parameters")
selected_country_names = st.sidebar.multiselect(
    "Target Nodes (Countries):",
    options=list(COUNTRY_MAP.keys()),
    default=["Japan", "India", "South Korea"]
)

if not selected_country_names:
    st.warning("Initialize at least one target node.")
    st.stop()

selected_codes = [COUNTRY_MAP[name] for name in selected_country_names]
forecast_horizon = st.sidebar.slider("Forecast Horizon (Years):", min_value=5, max_value=30, value=15, step=1)

with st.spinner("Compiling World Bank API data..."):
    df_all = load_world_bank_data(tuple(selected_codes))
    df_filtered = df_all.sort_values(['Country', 'Year'])

# --- Main Dashboard ---
st.title("Computational Demographic Forecasting")
st.markdown("#### Time-Series Analysis of Labor Interdependency & GDP Output")
st.write("This engine leverages live API data to model historical demographic shifts and utilizes ARIMA forecasting to predict structural macroeconomic variance.")
st.divider()

# --- Section 1: Predictive Modeling (ARIMA) ---
st.subheader("1. ARIMA Stochastic Forecasting Model")
st.write("Model mathematical specification: $\\phi(L)(1-L)^d y_t = \\theta(L)\\varepsilon_t$")

forecast_country = st.selectbox("Select Node for Predictive Modeling:", options=selected_country_names)
df_ts = df_filtered[df_filtered['Country'] == forecast_country].set_index('Year')

if len(df_ts) > 10:
    # Train ARIMA Model on Working Age Population
    ts_data = df_ts['Working_Age_Pct']
    model = ARIMA(ts_data, order=(1, 1, 1))
    model_fit = model.fit()
    
    # Forecast
    forecast = model_fit.get_forecast(steps=forecast_horizon)
    forecast_index = np.arange(df_ts.index[-1] + 1, df_ts.index[-1] + 1 + forecast_horizon)
    forecast_mean = forecast.predicted_mean
    conf_int = forecast.conf_int()

    # Plotly Graph with Confidence Intervals
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=df_ts.index, y=ts_data, mode='lines+markers', name='Historical Data', line=dict(color='blue')))
    fig_forecast.add_trace(go.Scatter(x=forecast_index, y=forecast_mean, mode='lines', name='ARIMA Forecast', line=dict(color='red', dash='dash')))
    fig_forecast.add_trace(go.Scatter(
        x=np.concatenate([forecast_index, forecast_index[::-1]]),
        y=np.concatenate([conf_int.iloc[:, 0], conf_int.iloc[:, 1][::-1]]),
        fill='toself', fillcolor='rgba(255, 0, 0, 0.2)', line=dict(color='rgba(255,255,255,0)'),
        showlegend=True, name='95% Confidence Interval'
    ))
    
    fig_forecast.update_layout(title=f"{forecast_country}: Working-Age Population Forecast to {forecast_index[-1]}", xaxis_title="Year", yaxis_title="Working-Age Population (%)")
    st.plotly_chart(fig_forecast, use_container_width=True)
    
    # Model Diagnostics Printout
    with st.expander("Expand for Computational Diagnostics & AIC/BIC Matrices"):
        st.code(model_fit.summary().as_text(), language="text")
else:
    st.error("Insufficient historical nodes for timeseries processing.")

# --- Section 2: OLS Regression Estimation ---
st.subheader("2. Ordinary Least Squares (OLS) Optimization")
st.write("Objective function modeling: $\\ln(\\text{GDP per Capita}_t) = \\beta_0 + \\beta_1 (\\text{Working-Age Pct}_t) + \\varepsilon_t$")

X = sm.add_constant(df_ts['Working_Age_Pct'])
y = df_ts['Log_GDP']
ols_model = sm.OLS(y, X).fit()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Optimal Coefficient (β₁)", f"{ols_model.params['Working_Age_Pct']:.4f}")
m2.metric("Variance (p-value)", f"{ols_model.pvalues['Working_Age_Pct']:.3e}")
m3.metric("R² Fit", f"{ols_model.rsquared:.3f}")
m4.metric("Adjusted R²", f"{ols_model.rsquared_adj:.3f}")
