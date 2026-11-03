import os
import pickle
import pandas as pd
import streamlit as st

#Image file to use.
IMAGE_FILE = next((f for f in ("house.jpg", "house.jpeg", "house.png") if os.path.exists(f)), None)

# use the photo as the browser-tab icon.
st.set_page_config(page_title="California House Price Predictor",
                   page_icon=IMAGE_FILE if IMAGE_FILE else None)

@st.cache_resource
def load_model():
    with open("best_knn_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

if IMAGE_FILE:
    try:
        st.image(IMAGE_FILE, width="stretch")
    except TypeError:  # older Streamlit versions
        st.image(IMAGE_FILE, use_container_width=True)

st.title("California House Price Predictor")
st.write("KNN regression pipeline (StandardScaler + tuned KNeighborsRegressor) "
         "trained on the 1990 California census data.")

st.sidebar.header("Block group features")
MedInc = st.sidebar.slider("MedInc (median income, $10,000s)", 0.5, 15.0, 3.9, 0.1)
HouseAge = st.sidebar.slider("HouseAge (median age, years)", 1, 52, 29)
AveRooms = st.sidebar.slider("AveRooms (avg rooms per household)", 1.0, 15.0, 5.4, 0.1)
AveBedrms = st.sidebar.slider("AveBedrms (avg bedrooms per household)", 0.5, 5.0, 1.1, 0.05)
Population = st.sidebar.number_input("Population", 3, 36000, 1200)
AveOccup = st.sidebar.slider("AveOccup (avg household members)", 1.0, 10.0, 3.0, 0.1)
Latitude = st.sidebar.slider("Latitude", 32.5, 42.0, 34.2, 0.01)
Longitude = st.sidebar.slider("Longitude", -124.4, -114.3, -118.5, 0.01)

input_df = pd.DataFrame([{
    "MedInc": MedInc, "HouseAge": HouseAge, "AveRooms": AveRooms,
    "AveBedrms": AveBedrms, "Population": Population, "AveOccup": AveOccup,
    "Latitude": Latitude, "Longitude": Longitude,
}])

st.subheader("Your input")
st.dataframe(input_df, hide_index=True)

if st.button("Predict price", type="primary"):
    pred = model.predict(input_df)[0]
    st.success(f"Predicted median house value: **${pred * 100000:,.0f}**")
    st.caption(f"Model output: {pred:.3f} (in $100,000s)")
    st.map(pd.DataFrame({"lat": [Latitude], "lon": [Longitude]}))
