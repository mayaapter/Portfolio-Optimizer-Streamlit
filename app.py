import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(page_title="Portfolio Optimizer | Maya Apter", layout="wide")
st.title("Portfolio Optimizer - Maya Apter")
st.markdown("### Developed by Maya Apter - Data & Business Student")

# --- Sidebar Menu ---
st.sidebar.header("Portfolio Settings")
input_tickers = st.sidebar.text_area("Enter tickers separated by commas", 
                                    "AAPL, NVDA, MSFT, META, GOOGL, AMZN, PLTR")
rf_rate = st.sidebar.slider("Risk-Free Rate (%)", 0.0, 10.0, 2.0, 0.01) / 100
max_weight_limit = st.sidebar.slider("Max Weight per Asset (%)", 5, 100, 20, 1) / 100
initial_investment = st.sidebar.number_input("Initial Investment ($)", min_value=1, value=10000, step=100)
start_date = st.sidebar.date_input("Start Date for Backtest", value=pd.to_datetime("2024-01-01"))

tickers = [t.strip().upper() for t in input_tickers.split(',')]

# --- Caching Functions for Performance ---

@st.cache_data
def get_data(tickers, start):
    """Downloads and cleans data - runs only if tickers or date change"""
    df = yf.download(tickers, start=start)['Close']
    # Flatten Multi-index if present (common in newer yfinance versions)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(1)
    df = df.ffill().dropna(axis=1, how='all').dropna()
    return df

@st.cache_data
def run_monte_carlo(returns, num_portfolios, rf, max_w):
    """Calculates Monte Carlo Simulation"""
    num_assets = len(returns.columns)
    results = np.zeros((3, num_portfolios))
    weights_record = []
    
    # Pre-calculate annualized mean returns and covariance matrix
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    
    for i in range(num_portfolios):
        valid = False
        while not valid:
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            # Apply user-defined constraint
            if np.max(weights) <= max_w:
                valid = True
        
        weights_record.append(weights)
        p_ret = np.sum(mean_returns * weights)
        p_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        results[0,i] = p_ret # Expected Return
        results[1,i] = p_std # Volatility
        results[2,i] = (p_ret - rf) / p_std # Sharpe Ratio
        
    return results, weights_record

# --- Main Logic Execution ---

if st.sidebar.button("Run Optimization"):
    # 1. Data Retrieval
    with st.status("Fetching data from Yahoo Finance...") as status:
        data = get_data(tickers, start_date)
        returns = data.pct_change().dropna()
        status.update(label="Running Monte Carlo Simulation...", state="running")
        
        # 2. Optimization- finding the best portfolio
        results, weights_record = run_monte_carlo(returns, 3000, rf_rate, max_weight_limit)
        status.update(label="Calculations complete!", state="complete")

    # Locate the portfolio with the highest Sharpe Ratio
    max_sharpe_idx = np.argmax(results[2])
    best_weights = weights_record[max_sharpe_idx]

    # --- Display Optimization Results ---
    st.subheader("Optimization Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Expected Yearly Return", f"{results[0,max_sharpe_idx]:.2%}")
    c2.metric("Volatility (Risk)", f"{results[1,max_sharpe_idx]:.2%}")
    c3.metric("Sharpe Ratio", f"{results[2,max_sharpe_idx]:.2f}")

    # --- Recommended Allocation ---
    st.divider()
    col_l, col_r = st.columns([1, 1.5])
    
    weights_df = pd.DataFrame({
        'Stock': data.columns,
        'Weight (%)': [round(w * 100, 2) for w in best_weights]
    }).sort_values(by='Weight (%)', ascending=False)

    with col_l:
        st.write("**Optimal Asset Allocation:**")
        st.dataframe(weights_df, hide_index=True, use_container_width=True)

    with col_r:
            # Bar Chart Visualization
            fig_bar, ax_bar = plt.subplots(figsize=(8, 5))
            
            # Using Seaborn for a nicer look
            sns.barplot(
                x='Weight (%)', 
                y='Stock', 
                data=weights_df, 
                palette='viridis', 
                ax=ax_bar
            )
            
            # Adding labels and title
            ax_bar.set_title("Optimal Portfolio Allocation", fontsize=14)
            ax_bar.set_xlabel("Allocation Percentage (%)")
            ax_bar.set_ylabel("Stock Ticker")
            
            # Adding the percentage values on the bars
            for i, v in enumerate(weights_df['Weight (%)']):
                ax_bar.text(v + 0.5, i, f"{v}%", color='black', va='center', fontweight='bold')
                
            st.pyplot(fig_bar)

    # --- Backtesting Section (Reality Check) ---
    st.divider()
    st.subheader(f"Performance of ${initial_investment:,} Invested on {start_date}")
    
    # Calculate portfolio growth over time based on optimal weights
    portfolio_daily_rets = returns.dot(best_weights)
    cumulative_val = (1 + portfolio_daily_rets).cumprod() * initial_investment
    
    # Comparison with S&P 500 (Benchmark)
    spy_data = yf.download("SPY", start=start_date)['Close']
    if isinstance(spy_data, pd.DataFrame): 
        spy_data = spy_data.iloc[:, 0]
    
    # Syncing SPY returns with portfolio timeframe
    spy_rets = spy_data.pct_change().dropna().reindex(portfolio_daily_rets.index).fillna(0)
    cumulative_spy = (1 + spy_rets).cumprod() * initial_investment
    
    # Display final performance metrics
    final_p = cumulative_val.iloc[-1]
    final_s = cumulative_spy.iloc[-1]
    
    m1, m2 = st.columns(2)
    m1.metric("Optimal Portfolio Final Value", f"${final_p:,.2f}", f"{((final_p/initial_investment)-1):.2%}")
    m2.metric("S&P 500 Final Value", f"${final_s:,.2f}", f"{((final_s/initial_investment)-1):.2%}")
    
    # Historical Performance Line Chart
    comparison_df = pd.DataFrame({
        "Your Optimized Portfolio": cumulative_val,
        "S&P 500 (SPY)": cumulative_spy
    })
    st.line_chart(comparison_df)

    # --- Correlation Analysis ---
    st.divider()
    st.subheader("Correlation Matrix")
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sns.heatmap(returns.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
    st.pyplot(fig_corr)

else:
    st.info("Click 'Run Optimization' in the sidebar to begin the analysis.")