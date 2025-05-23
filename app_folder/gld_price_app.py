import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load and cache data
@st.cache_data
def load_data():
    return pd.read_csv("gld_price_data.csv")

df = load_data()

# Feature and target selection
X = df[['SPX', 'USO', 'SLV', 'EUR/USD']]
y = df['GLD']

# Train/test split (shared across models)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
@st.cache_resource
def train_models():
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    gb_model.fit(X_train, y_train)
    return rf_model, gb_model

rf_model, gb_model = train_models()

# App layout
st.title("Gold Price Prediction & Model Analysis")

# Radio button menu
option = st.radio(
    "Select an Option",
    ["Random Forest Analysis", "Gradient Boosting Analysis", "Predict Using Input"]
)

# 1. Random Forest Analysis
if option == "Random Forest Analysis":
    st.header("Random Forest Regressor - Detailed Analysis Based on R²")

    # Custom-tuned model
    regressor = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=7,
        random_state=42
    )
    regressor.fit(X_train, y_train)

    # Predictions
    train_predictions = regressor.predict(X_train)
    test_predictions = regressor.predict(X_test)

    # Metrics
    train_r2 = metrics.r2_score(y_train, train_predictions)
    test_r2 = metrics.r2_score(y_test, test_predictions)
    test_mse = metrics.mean_squared_error(y_test, test_predictions)
    test_rmse = np.sqrt(test_mse)  # Manual RMSE calculation
    test_mae = metrics.mean_absolute_error(y_test, test_predictions)

    # Show metrics
    st.subheader("Training Metrics")
    st.write(f"**R² (Train)**: `{train_r2:.4f}`")

    st.subheader("Testing Metrics")
    st.markdown(f"""
    | Metric | Value |
    |--------|--------|
    | R² Score | `{test_r2:.4f}` |
    | RMSE     | `{test_rmse:.4f}` |
    | MSE      | `{test_mse:.4f}` |
    | MAE      | `{test_mae:.4f}` |
    """, unsafe_allow_html=True)

    # Plot actual vs predicted
    st.subheader("Actual vs Predicted GLD Price")

    fig = plt.figure(figsize=(10, 5))
    plt.plot(y_test.values, label='Actual Value', color='red')
    plt.plot(test_predictions, label='Predicted Value', color='blue')
    plt.title('Actual Price vs Predicted Price')
    plt.xlabel('Index')
    plt.ylabel('GLD Price')
    plt.legend()
    st.pyplot(fig)

    fig_rf = plt.figure()
    plt.scatter(y_test, test_predictions, alpha=0.5, color='cyan')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual GLD")
    plt.ylabel("Predicted GLD")
    plt.title("Random Forest Predictions")
    st.pyplot(fig_rf)


# 2. Gradient Boosting Analysis
elif option == "Gradient Boosting Analysis":
    st.header("Gradient Boosting Regressor - Detailed Analysis Based on R²")

    # Prepare features and target
    X = df.drop(['Date', 'GLD'], axis=1)
    Y = df['GLD']

    # Train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Train model with custom hyperparameters
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    import numpy as np

    gbr = GradientBoostingRegressor(
        n_estimators=50,
        learning_rate=0.07,
        max_depth=7,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    )
    gbr.fit(X_train, Y_train)

    # Training and test predictions
    train_predictions = gbr.predict(X_train)
    test_predictions = gbr.predict(X_test)

    # Metrics
    train_r2 = r2_score(Y_train, train_predictions)
    test_r2 = r2_score(Y_test, test_predictions)
    test_mse = mean_squared_error(Y_test, test_predictions)
    test_rmse = np.sqrt(test_mse)  # Manual RMSE
    test_mae = mean_absolute_error(Y_test, test_predictions)

    # Show metrics
    st.subheader("Training Metrics")
    st.write(f"**R² (Train)**: `{train_r2:.4f}`")

    st.subheader("Testing Metrics")
    st.markdown(f"""
    | Metric | Value |
    |--------|--------|
    | R² Score | `{test_r2:.4f}` |
    | RMSE     | `{test_rmse:.4f}` |
    | MSE      | `{test_mse:.4f}` |
    | MAE      | `{test_mae:.4f}` |
    """, unsafe_allow_html=True)

    # Plot actual vs predicted
    st.subheader("Actual vs Predicted GLD Price")

    fig = plt.figure(figsize=(10, 5))
    plt.plot(Y_test.values, label='Actual Value', color='red')
    plt.plot(test_predictions, label='Predicted Value', color='orange')
    plt.title('Actual Price vs Predicted Price (Gradient Boosting)')
    plt.xlabel('Index')
    plt.ylabel('GLD Price')
    plt.legend()
    st.pyplot(fig)

    fig_gb = plt.figure()
    plt.scatter(Y_test, test_predictions, alpha=0.5, color='green')
    plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--')
    plt.xlabel("Actual GLD")
    plt.ylabel("Predicted GLD")
    plt.title("Gradient Boosting Predictions")
    st.pyplot(fig_gb)

# 3. User Input Prediction
elif option == "Predict Using Input":
    st.header("Gold Price Prediction from User Input")

    # Select model
    model_choice = st.selectbox("Choose a Model", ["Random Forest", "Gradient Boosting"])

    # Input sliders
    spx = st.slider("S&P 500 Index (SPX)", min_value=0.0, max_value=2000.0, value=1400.0)
    uso = st.slider("US Oil Fund (USO)", min_value=0.0, max_value=120.0, value=75.0)
    slv = st.slider("Silver Price (SLV)", min_value=0.0, max_value=40.0, value=15.0)
    eur_usd = st.slider("EUR/USD Exchange Rate", min_value=0.5, max_value=2.0, value=1.45)

    # Input data for prediction
    input_df = pd.DataFrame([[spx, uso, slv, eur_usd]], columns=['SPX', 'USO', 'SLV', 'EUR/USD'])

    # Predict using pre-trained model
    if st.button("Predict"):
        model = rf_model if model_choice == "Random Forest" else gb_model
        prediction = model.predict(input_df)
        st.success(f"Predicted Gold Price using {model_choice}: ${prediction[0]:.2f}")
