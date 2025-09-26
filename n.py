import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import plotly.graph_objects as go

class HousePricePredictorApp:
    def __init__(self):
        self.load_and_prepare_data()
        self.train_models()

    def load_and_prepare_data(self):
        data = pd.read_csv('Housing.csv')
        self.features = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'furnishingstatus']
        self.target = 'price'

        for feature in self.features:
            if data[feature].dtype == 'object':
                if feature == 'furnishingstatus':
                    dummies = pd.get_dummies(data[feature], prefix=feature, drop_first=True)
                    data = pd.concat([data, dummies], axis=1)
                    self.features.remove(feature)
                    self.features.extend(dummies.columns)
                else:
                    data[feature] = (data[feature] == 'yes').astype(int)

        self.X = data[self.features]
        self.y = data[self.target]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

    def train_models(self):
        self.lr_model = LinearRegression()
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)

        self.lr_model.fit(self.X_train_scaled, self.y_train)
        self.rf_model.fit(self.X_train_scaled, self.y_train)
        self.gb_model.fit(self.X_train_scaled, self.y_train)

    def predict_price(self, input_data):
        input_df = pd.DataFrame([input_data])

        for col in self.X.columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[self.X.columns]
        input_scaled = self.scaler.transform(input_df)

        model_name = st.session_state.model
        if model_name == 'Linear Regression':
            prediction = self.lr_model.predict(input_scaled)[0]
        elif model_name == 'Random Forest':
            prediction = self.rf_model.predict(input_scaled)[0]
        elif model_name == 'Gradient Boosting':
            prediction = self.gb_model.predict(input_scaled)[0]

        return prediction

    def get_model_performance(self):
        lr_pred = self.lr_model.predict(self.X_test_scaled)
        rf_pred = self.rf_model.predict(self.X_test_scaled)
        gb_pred = self.gb_model.predict(self.X_test_scaled)

        performance_data = {
            'Linear Regression': self.get_metrics(self.y_test, lr_pred),
            'Random Forest': self.get_metrics(self.y_test, rf_pred),
            'Gradient Boosting': self.get_metrics(self.y_test, gb_pred)
        }

        return performance_data

    @staticmethod
    def get_metrics(y_true, y_pred):
        r2 = r2_score(y_true, y_pred)
        accuracy = 100 * (1 - np.mean(np.abs((y_true - y_pred) / y_true)))
        return {
            'Accuracy (%)': accuracy,
            'R-squared': r2
        }

def main():
    st.set_page_config(page_title="Advanced House Price Predictor", layout="wide")

    # Custom CSS for advanced styling with enhanced fonts and colors
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&family=Playfair+Display:wght@700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
        color: #ffffff;
    }

    h1, h2, h3 {
        font-family: 'Playfair Display', serif;
        color: #3498db;
    }

    .main {
        background-color: #1e1e1e;
    }

    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }

    .st-bw {
        background-color: #2c2c2c;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    </style>
    """, unsafe_allow_html=True)

    st.title("🏠 Advanced House Price Predictor")
    st.markdown("<h3 style='color: #b0b0b0;'>Estimate the price of your dream home with our advanced machine learning models.</h3>", unsafe_allow_html=True)

    app = HousePricePredictorApp()

    tab1, tab2 = st.tabs(["💰 Price Prediction", "📊 Model Performance"])

    with tab1:
        st.header("House Details")
        col1, col2, col3 = st.columns(3)

        with col1:
            area = st.slider("Area (sq ft)", min_value=1000, max_value=10000, value=5000, step=100)
            bedrooms = st.slider("Bedrooms", min_value=1, max_value=6, value=3)
            bathrooms = st.slider("Bathrooms", min_value=1, max_value=4, value=2)
            stories = st.slider("Stories", min_value=1, max_value=4, value=2)

        with col3:
            mainroad = st.checkbox("Main Road", key="mainroad")
            guestroom = st.checkbox("Guest Room", key="guestroom")
            basement = st.checkbox("Basement", key="basement")
            hotwaterheating = st.checkbox("Hot Water Heating", key="hotwaterheating")
            airconditioning = st.checkbox("Air Conditioning", key="airconditioning")
            parking = st.slider("Parking Spaces", min_value=0, max_value=3, value=1)

        st.markdown("<h3 style='color: #3498db;'>Furnishing Status</h3>", unsafe_allow_html=True)
        furnishing = st.selectbox("Select furnishing status", ["Unfurnished", "Semi-Furnished", "Furnished"], key="furnishing")

        # Display selected furnishing status
        st.markdown(f"<h4 style='color: #e74c3c;'>Selected Furnishing Status: {furnishing}</h4>", unsafe_allow_html=True)

        st.markdown("<h3 style='color: #3498db;'>Select Model</h3>", unsafe_allow_html=True)
        model = st.selectbox("Choose prediction model", ["Linear Regression", "Random Forest", "Gradient Boosting"], key="model")

        # Display selected model
        st.markdown(f"<h4 style='color: #e74c3c;'>Selected Model: {model}</h4>", unsafe_allow_html=True)

        if st.button("Predict Price", key="predict"):
            input_data = {
                'area': area,
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'stories': stories,
                'mainroad': int(mainroad),
                'guestroom': int(guestroom),
                'basement': int(basement),
                'hotwaterheating': int(hotwaterheating),
                'airconditioning': int(airconditioning),
                'parking': parking,
                'furnishingstatus_semi-furnished': int(furnishing == "Semi-Furnished"),
                'furnishingstatus_furnished': int(furnishing == "Furnished")
            }

            prediction = app.predict_price(input_data)

            st.markdown("<h3 style='color: #3498db;'>🏷 Estimated House Price</h3>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: center; color: #2ecc71; font-size: 48px;'>₹ {prediction:,.2f}</h1>", unsafe_allow_html=True)

            # Create a radar chart for house features
            features = ['Area', 'Bedrooms', 'Bathrooms', 'Stories', 'Parking']
            values = [area/1000, bedrooms, bathrooms, stories, parking]

            fig = go.Figure(data=go.Scatterpolar(
              r=values,
              theta=features,
              fill='toself',
              line=dict(color='#3498db')
            ))

            fig.update_layout(
              polar=dict(
                radialaxis=dict(
                  visible=True,
                  range=[0, max(values)],
                  color='white'
                ),
                angularaxis=dict(color='white')
              ),
              showlegend=False,
              paper_bgcolor='rgba(0,0,0,0)',
              plot_bgcolor='rgba(0,0,0,0)'
            )

            st.plotly_chart(fig)

    with tab2:
        st.header("Model Performance")
        performance_data = app.get_model_performance()
        metrics = ['Accuracy (%)', 'R-squared']
        models = ['Linear Regression', 'Random Forest', 'Gradient Boosting']

        for metric in metrics:
            values = [performance_data[model][metric] for model in models]
            fig = go.Figure(data=[go.Bar(x=models, y=values, text=values, textposition='auto', marker=dict(color=['#3498db', '#e74c3c', '#2ecc71']))])
            fig.update_layout(title=f"{metric} Comparison", xaxis_title="Model", yaxis_title=metric)
            st.plotly_chart(fig)

if __name__ == "__main__":
    main()
