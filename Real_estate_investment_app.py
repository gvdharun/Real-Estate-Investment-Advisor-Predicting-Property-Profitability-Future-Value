import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(page_title="🏠 Real Estate Investment Advisor", layout="wide")

@st.cache_resource
def load_models():
    """Load pickled models"""
    with open("models/best_xgb_model.pkl", "rb") as f:
        xgb_clf = joblib.load(f)
    with open("models/best_lin_reg_model.pkl", "rb") as f:
        lin_reg = joblib.load(f)
    return xgb_clf, lin_reg

@st.cache_data
def load_sample_data():
    """Load dataset for filtering and visuals"""
    df = pd.read_csv("data/india_housing_prices.csv")
    
    # Add engineered features to match training data
    df["Price_per_SqFt"] = df["Price_in_Lakhs"] * 1e5 / df["Size_in_SqFt"]
    df["School_Density_Score"] = df["Nearby_Schools"] / (df["Size_in_SqFt"] / 1000)
    df["Hospital_Density_Score"] = df["Nearby_Hospitals"] / (df["Size_in_SqFt"] / 1000)
    
    return df

# Load models and data
xgb_clf, lin_reg = load_models()
df_sample = load_sample_data()

# Exact feature columns from your training
FEATURE_COLS = [
    "State", "City", "Locality", "Property_Type",
    "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt",
    "Year_Built", "Age_of_Property",
    "Nearby_Schools", "Nearby_Hospitals",
    "Public_Transport_Accessibility", "Parking_Space",
    "Furnished_Status", "Floor_No", "Total_Floors",
    "Security", "Amenities", "Facing",
    "Owner_Type", "Availability_Status",
    "School_Density_Score", "Hospital_Density_Score"
]

st.title("🏠 **Real Estate Investment Advisor**")
st.markdown("---")

# Sidebar: Filters
st.sidebar.header("🔍 **Filter Properties**")
min_price = st.sidebar.slider("Min Price (Lakhs)", 0, 1000, 10)
max_price = st.sidebar.slider("Max Price (Lakhs)", 0, 5000, 500)
min_size = st.sidebar.slider("Min Size (sqft)", 200, 10000, 500)
max_size = st.sidebar.slider("Max Size (sqft)", 200, 20000, 5000)
bhk_filter = st.sidebar.multiselect("BHK", sorted(df_sample["BHK"].unique()), default=[2,3])

filtered_df = df_sample[
    (df_sample["Price_in_Lakhs"] >= min_price) &
    (df_sample["Price_in_Lakhs"] <= max_price) &
    (df_sample["Size_in_SqFt"] >= min_size) &
    (df_sample["Size_in_SqFt"] <= max_size) &
    (df_sample["BHK"].isin(bhk_filter))
]

# Main input form - MATCHES EXACT FEATURE_COLS
col1, col2 = st.columns(2)
with col1:
    st.header("📝 **Property Details**")
    state = st.selectbox("State", df_sample["State"].unique())
    city = st.selectbox("City", df_sample[df_sample["State"] == state]["City"].unique())
    locality = st.selectbox("Locality", df_sample[df_sample["City"] == city]["Locality"].unique())
    prop_type = st.selectbox("Property Type", df_sample["Property_Type"].unique())
    bhk = st.number_input("BHK", 1, 10, 3)
    size_sqft = st.number_input("Size (sqft)", 200, 20000, 1200)
    
with col2:
    price_lakhs = st.number_input("Current Price (Lakhs)", 5.0, 5000.0, 100.0)
    furnished = st.selectbox("Furnished Status", df_sample["Furnished_Status"].unique())
    floor_no = st.number_input("Floor No", 0, 50, 2)
    total_floors = st.number_input("Total Floors", 1, 100, 10)
    schools = st.number_input("Nearby Schools", 0, 20, 3)
    hospitals = st.number_input("Nearby Hospitals", 0, 20, 2)

# Prediction button
if st.button("🚀 **Get Investment Analysis**", type="primary"):
    # Create EXACT input matching FEATURE_COLS
    price_per_sqft = price_lakhs * 1e5 / size_sqft
    school_density = schools / (size_sqft / 1000)
    hospital_density = hospitals / (size_sqft / 1000)
    
    input_data = {
        "State": state,
        "City": city,
        "Locality": locality,
        "Property_Type": prop_type,
        "BHK": bhk,
        "Size_in_SqFt": size_sqft,
        "Price_in_Lakhs": price_lakhs,
        "Price_per_SqFt": price_per_sqft,
        "Year_Built": df_sample["Year_Built"].median(),
        "Age_of_Property": df_sample["Age_of_Property"].median(),
        "Nearby_Schools": schools,
        "Nearby_Hospitals": hospitals,
        "Public_Transport_Accessibility": df_sample["Public_Transport_Accessibility"].mode().iloc[0],
        "Parking_Space": df_sample["Parking_Space"].mode().iloc[0],
        "Furnished_Status": furnished,
        "Floor_No": floor_no,
        "Total_Floors": total_floors,
        "Security": df_sample["Security"].mode().iloc[0],
        "Amenities": df_sample["Amenities"].mode().iloc[0],
        "Facing": df_sample["Facing"].mode().iloc[0],
        "Owner_Type": df_sample["Owner_Type"].mode().iloc[0],
        "Availability_Status": df_sample["Availability_Status"].mode().iloc[0],
        "School_Density_Score": school_density,
        "Hospital_Density_Score": hospital_density
    }
    
    X_input = pd.DataFrame([input_data])[FEATURE_COLS]
    
    # Predict
    good_proba = xgb_clf.predict_proba(X_input)[:, 1][0]
    future_price = lin_reg.predict(X_input)[0]
    
    # Results
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("💰 **Price in 5 Years**", f"₹{future_price:,.0f}L")
    with col2:
        decision = "✅ GOOD INVESTMENT" if good_proba >= 0.5 else "❌ **NOT IDEAL**"
        st.metric("📈 **Decision**", decision)
    with col3:
        st.metric("🎯 **Confidence**", f"{good_proba:.1%}")
    
    st.progress(float(good_proba))

# Visual insights
st.header("📊 **Market Insights**")
col1, col2 = st.columns(2)

with col1:
    st.subheader("🏙️ **Avg Price by City**")
    city_prices = filtered_df.groupby("City")["Price_in_Lakhs"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 4))
    city_prices.plot(kind="barh", ax=ax, color="skyblue")
    plt.title("Average Price by City (Filtered)")
    plt.xlabel("Price (Lakhs)")
    st.pyplot(fig)

with col2:
    st.subheader("📏 **Price vs Size**")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(data=filtered_df.sample(min(1000, len(filtered_df))), 
                    x="Size_in_SqFt", y="Price_in_Lakhs", ax=ax, alpha=0.6)
    plt.title("Price vs Size Distribution")
    st.pyplot(fig)

# Feature importance
# ... (keep all previous code until the checkbox)

# ✅ COMPLETE FIXED FEATURE IMPORTANCE
if st.checkbox("⚖️ **Show Feature Importance**"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🟡 XGBoost Classifier")
        xgb_model = xgb_clf.named_steps["model"]
        feature_names_clf = xgb_clf.named_steps["preprocess"].get_feature_names_out(FEATURE_COLS)
        importance_clf = pd.Series(xgb_model.feature_importances_, index=feature_names_clf)
        
        # Clean up long OneHotEncoder names for display
        importance_clf.index = importance_clf.index.str[:50]  # Truncate long names
        
        top10_clf = importance_clf.nlargest(10)
        fig, ax = plt.subplots(figsize=(10, 5))
        top10_clf.sort_values().plot(kind="barh", ax=ax, color="gold")
        plt.title("Top 10 Features Driving Investment Decisions")
        plt.xlabel("Importance Score")
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.subheader("🔴 Linear Regression")
        linreg_model = lin_reg.named_steps["model"]
        feature_names_reg = lin_reg.named_steps["preprocess"].get_feature_names_out(FEATURE_COLS)
        coef_reg = pd.Series(linreg_model.coef_, index=feature_names_reg)
        
        # Clean up names
        coef_reg.index = coef_reg.index.str[:50]
        
        top10_reg = coef_reg.abs().nlargest(10)
        fig, ax = plt.subplots(figsize=(10, 5))
        top10_reg.sort_values().plot(kind="barh", ax=ax, color="coral")
        plt.title("Top 10 Features Impacting Price Prediction")
        plt.xlabel("Coefficient Magnitude (|β|)")
        plt.tight_layout()
        st.pyplot(fig)

st.markdown("---")
st.caption("🤖 Powered by your exact FEATURE_COLS | XGBoost + Linear Regression")
