import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import gamma
import yfinance as yf
import pandas as pd
from datetime import timedelta
from scipy.interpolate import griddata
import plotly.io as pio
# Set the default renderer to display plots in your browser
pio.renderers.default = 'browser'
import plotly.graph_objects as go

st.title("Physics Informed Neural Network to Model and Calculate Implied Volatility Surface with Market Data")

# %%
"""
Created by sylvestercleveland | SCLEVEL3@ALUMNI.JH.EDU

"""

st.sidebar.header("Orders of Caputo Derivative with GL Scheme")
st.sidebar.write('Adjust the parameters for the Black-Scholes model.')
alpha = st.sidebar.number_input(
    'Caputo w/ GL scheme order alpha (e.g., 0.3 from 0.1 to 1.0)',
    value=0.7,
    step=0.1,
    format="%.2f"
)

risk_free_rate = st.sidebar.number_input(
    'Risk-Free Interest Rate (e.g., 0.050 for 0.5%)',
    value=0.050,
    format="%.3f"
)

dividend_yield = st.sidebar.number_input(
    'Dividend Yield (e.g., 0.013 for 1.3%)',
    value=0.000,
    format="%.3f"
)

st.sidebar.header('Ticker Symbol')
ticker_symbol = st.sidebar.text_input(
    'Enter Ticker Symbol',
    value='AAPL',
    max_chars=10
).upper()

# alpha_1 = st.sidebar.slider("𝗗𝛂", min_value=0.0, max_value=1.0, value=0.10, step=0.10, format='%.3f')
# alpha_2 = st.sidebar.slider("2nd 𝛂", min_value=0.0, max_value=1.0, value=0.30, step=0.10, format='%.3f')
# alpha_3 = st.sidebar.slider("3rd 𝛂", min_value=0.0, max_value=1.0, value=0.70, step=0.10, format='%.3f')
# alpha_4 = st.sidebar.slider("4th 𝛂", min_value=0.0, max_value=1.0, value=0.90, step=0.10, format='%.3f')

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters
r = risk_free_rate #0.05     # Risk-free rate
sigma_true = 0.35  # True (constant) volatility
alpha = alpha #0.7   # Fractional order
T = 1.0       # Maturity
K = 10     # Strike
S_max = 20   # Max stock price

S = torch.linspace(1e-5, S_max, 50)
t = torch.linspace(1e-5, T, 50)
S_grid, t_grid = torch.meshgrid(S, t, indexing='ij')
S_grid2, t_grid2 = torch.meshgrid(S.squeeze(), t.squeeze(), indexing='ij')  # rowwise, colwise
# S_grid[:, 0], S_grid2[:, 1], t_grid[0, :], t_grid2[0, :]

# Synthetic European Call Price (Black-Scholes)
def bs_call_price(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S * norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

# Neural Net Approximation
class PINN(nn.Module):
    def __init__(self, hidden=20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x)

# Generate Training Grid
# def generate_grid(N_S=50, N_t=50):
#     S = torch.linspace(1e-5, S_max, N_S)
#     t = torch.linspace(1e-5, T, N_t)
#     S_grid, t_grid = torch.meshgrid(S, t, indexing='ij')
#     X = torch.stack([S_grid.flatten(), t_grid.flatten()], dim=1).to(device)
#     return S_grid, t_grid, X

# Caputo Approximation using GL scheme
def caputo_GL(f, dt, alpha):
    N = f.ravel().shape[0]
    coeffs = torch.tensor([(-1)**k * gamma(alpha + 1)/(gamma(k + 1)*gamma(alpha - k + 1)) for k in range(N)]), device=device)
    conv = torch.nn.functional.conv1d(f.view(1,1,-1), coeffs.flip(0).float().view(1,1,-1), padding=0)
    return conv.view(-1) / dt**alpha

# Boundary & Initial Conditions
def boundary_loss(model, S_grid, t_grid):
    S0 = S_grid[:, 0]
    t0 = t_grid[0, :]
    St = torch.stack([S0, torch.zeros_like(S0)], dim=1).to(device)
    V0 = model(St).squeeze()
    payoff = torch.maximum(S0 - K, torch.tensor(0.0, device=device))
    return torch.mean((V0 - payoff)**2)

def boundary_condition(x, t, K=10, r=0.01):
  # May be can try all zeros?
  return torch.maximum(x - K, torch.tensor(0.0, device=device)) #torch.where(x == 0, K * torch.exp(-r * t), torch.zeros_like(t)) #K * torch.exp(-r * t), torch.zeros_like(t))

# Residual loss from PDE
def pde_loss(model, X, alpha, vol=0.2):
    # S = X[:, 0].view(-1,1).requires_grad_(True)
    # t = X[:, 1].view(-1,1).requires_grad_(True)
    S = X.view(-1,1).requires_grad_(True)
    t = X.view(-1,1).requires_grad_(True)
    V = model(torch.cat([S, t], dim=1))

    V_S = torch.autograd.grad(V, S, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    V_SS = torch.autograd.grad(V_S, S, grad_outputs=torch.ones_like(V), create_graph=True)[0]

    # Fractional time derivative approximation (Caputo GL)
    dt = t[1] - t[0]
    V_time_series = V.view(len(torch.unique(S[:, 0])), len(torch.unique(t[:, 0])))  # reshape to [S, t]
    V_t_frac = torch.stack([
        caputo_GL(V_time_series.ravel()[i], dt, alpha)
        for i in range(V_time_series.ravel().shape[0])
    ]).view(-1,1)
    
    pde_residual = V_t_frac - (r * V - r * S * V_S) * t ** (1-alpha)/gamma(2-alpha) + gamma(1+alpha)/2 * vol**2 * S**2 * V_SS
    # pde_residual = V_t_frac + 0.5 * sigma_true**2 * S**2 * V_SS + r * S * V_S - r * V
    return torch.mean(pde_residual**2)

# Main Training Loop
model = PINN() #.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# S_grid, t_grid, X = generate_grid()
N = 50 #200 #50
x = torch.linspace(0, 20, N).view(-1, 1)
t = torch.linspace(0, 1, N).view(-1, 1) # Global temporal definition.
X, tau = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='xy')
X = X.reshape(-1,1)   # <-- collocation spatial points.
tau = tau.reshape(-1,1)         # <-- collocation temporal points.
N = int(np.sqrt(len(X)))

loss_history = []
with st.spinner('Training PINN of implied volatility surface...'):
    for epoch in range(1000):
        optimizer.zero_grad()
        loss_pde = pde_loss(model, X, alpha)    # <-- The X is Size 50. May need Size 2500.
        u_pred_left = model(torch.stack([torch.full_like(t.ravel(), x.min().item()), t.ravel()], dim=1)).to(device))  # u(0, t)
        u_pred_right = model(torch.stack([torch.full_like(t.ravel(), x.max().item()), t.ravel()], dim=1)).to(device))  # u(1, t)
        loss_bc = torch.mean((u_pred_left - boundary_condition(torch.full_like(t.ravel(), x.min().item()), t.ravel())) ** 2) + \
                  torch.mean((u_pred_right - boundary_condition(torch.full_like(t.ravel(), x.min().item()), t.ravel()))**2)
        # loss_bc = boundary_loss(model, S_grid, t_grid)
        loss = loss_pde + loss_bc
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        if epoch % 50 == 0:
            st.write(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# Create the training data -------------------------------------------------
ticker = yf.Ticker(ticker_symbol)
expirations = ticker.options

try:
    expirations = ticker.options
except Exception as e:
    st.error(f'Error fetching options for {ticker_symbol}: {e}')
    st.stop()

today = pd.Timestamp('today').normalize()
exp_dates = [pd.Timestamp(exp) for exp in expirations if pd.Timestamp(exp) > today + timedelta(days=7)]
option_data = []

with st.spinner('Calculating implied volatility surface...'):
    for exp_date in exp_dates:
        opt_chain = ticker.option_chain(exp_date.strftime('%Y-%m-%d'))
        calls = opt_chain.calls
        calls = calls[(calls['bid'] > 0) & (calls['ask'] > 0)]
        for index, row in calls.iterrows():
            strike = row['strike']
            bid = row['bid']
            ask = row['ask']
            mid_price = (bid + ask) / 2
            # option_data generation
            option_data.append({
                'expirationDate': exp_date,
                'strike': strike,
                'bid': bid,
                'ask': ask,
                'mid': mid_price
            })

options_df = pd.DataFrame(option_data)
spot_history = ticker.history(period='5d')
spot_price = spot_history['Close'].iloc[-1]

options_df['daysToExpiration'] = (options_df['expirationDate'] - today).dt.days
options_df['timeToExpiration'] = options_df['daysToExpiration'] / 365

options_df = options_df[
    (options_df['strike'] >= spot_price * (80 / 100)) &
    (options_df['strike'] <= spot_price * (120 / 100))
]

options_df.reset_index(drop=True, inplace=True)
options_df.sort_values('strike', inplace=True)
options_df['moneyness'] = options_df['strike'] / spot_price

options_df.sort_values('timeToExpiration', inplace=True)

# X = torch.randn(324)
# Y = torch.randn(324)
X = torch.randn(361)
Y = torch.randn(361)
# X = torch.randn(1936)
# Y = torch.randn(1936)
N = options_df['timeToExpiration'].values.shape[0]
# X = torch.zeros(N)
# Y = torch.zeros(N)

X[:options_df['timeToExpiration'].values.shape[0]] = \
        torch.tensor([i.item() for i in options_df['timeToExpiration'].values])
Y[:options_df['moneyness'].values.shape[0]] = \
        torch.tensor([i.item() for i in options_df['moneyness'].values])
X = X.view(-1, 1)
Y = Y.view(-1, 1)

# Evaluate Model for Surface -----------------------------------------------
N = options_df['timeToExpiration'].values.shape[0]
S_test = torch.linspace(X.min(), X.max(), N)
t_test = torch.linspace(Y.min(), Y.max(), N)
S_eval, t_eval = torch.meshgrid(X.ravel(), Y.ravel(), indexing='ij')
X_eval = torch.stack([S_eval.flatten(), t_eval.flatten()], dim=1) #.to(device)
x_eval = torch.stack([X.flatten(), Y.flatten()], dim=1) #.to(device)

V_pred = model(X_eval).view(X.shape[0], X.shape[0]).detach() #.detach().cpu().numpy()#.view(19,19).detach().cpu().numpy()

Z_pred = griddata((S_eval.ravel(), t_eval.ravel()), V_pred.ravel(), (S_eval, t_eval), method='linear')
# Z_pred = griddata((X.ravel(), Y.ravel()), V_pred.ravel(), (S_eval, t_eval), method='linear')
Z_pred = np.ma.array(Z_pred, mask=np.isnan(Z_pred))

x = torch.linspace(0, 1, V_pred.shape[0])
y = torch.linspace(0, 20, V_pred.shape[0])

fig = go.Figure(data=[go.Surface(#z=V_pred.reshape(X.shape[0], Y.shape[0]),#V_pred, #.reshape(N,N),#z=u_pred_pt2.reshape(N, N),
    x=S_eval, y=t_eval, z=V_pred,#.reshape(M, M) * 10, #u_pred_pt2.reshape(M, M) * 10,
    colorscale='Viridis',
    showscale=True,
    opacity=0.75
)])
fig.update_layout(
    title=f'Implied Volatility Surface for {ticker} Options',
    scene=dict(
        xaxis_title='Time to Expiration (years)',
        yaxis_title="Moneyness (S/K)",
        zaxis_title='Implied Volatility (%)'
        ),
        autosize=False,
        width=900,
        height=800,
        margin=dict(l=65, r=50, b=65, t=90)
)
st.plotly_chart(fig)
