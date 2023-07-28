import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import pickle

st.markdown(
        """
        <style>
        body {
            background-color: lightblue;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


data=pd.read_csv(r'C:\Users\sn767\coding\code\python\cleaned_data.csv')
pipe=pickle.load(open(r'C:\Users\sn767\coding\code\python\RidgeModel.pkl','rb'))


def predicted_price(area,bhk,bathrooms,selected_location):
       location=selected_location
       bhk=bhk
       bathrooms=bathrooms
       area=area


       input=pd.DataFrame([[location,area,bathrooms,bhk]],columns=['location','total_sqft','bath','bhk'])
       return pipe.predict(input)[0]
st.title("Banglore house price prediction")

st.write("Enter the details of the house to get the predicted price")
area = st.number_input("Area in square feet", min_value=100, max_value=10000, step=100)
bhk = st.number_input("Enter the BHK ", min_value=1, max_value=16, step=1)
bathrooms = st.number_input("Enter the bathrooms", min_value=1, max_value=16, step=1)
location = ['1st Block Jayanagar', '1st Phase JP Nagar',
       '2nd Phase Judicial Layout', '2nd Stage Nagarbhavi',
       '5th Block Hbr Layout', '5th Phase JP Nagar', '6th Phase JP Nagar',
       '7th Phase JP Nagar', '8th Phase JP Nagar', '9th Phase JP Nagar',
       'AECS Layout', 'Abbigere', 'Akshaya Nagar', 'Ambalipura',
       'Ambedkar Nagar', 'Amruthahalli', 'Anandapura', 'Ananth Nagar',
       'Anekal', 'Anjanapura', 'Ardendale', 'Arekere', 'Attibele',
       'BEML Layout', 'BTM 2nd Stage', 'BTM Layout', 'Babusapalaya',
       'Badavala Nagar', 'Balagere', 'Banashankari',
       'Banashankari Stage II', 'Banashankari Stage III',
       'Banashankari Stage V', 'Banashankari Stage VI', 'Banaswadi',
       'Banjara Layout', 'Bannerghatta', 'Bannerghatta Road',
       'Basavangudi', 'Basaveshwara Nagar', 'Battarahalli', 'Begur',
       'Begur Road', 'Bellandur', 'Benson Town', 'Bharathi Nagar',
       'Bhoganhalli', 'Billekahalli', 'Binny Pete', 'Bisuvanahalli',
       'Bommanahalli', 'Bommasandra', 'Bommasandra Industrial Area',
       'Bommenahalli', 'Brookefield', 'Budigere', 'CV Raman Nagar',
       'Chamrajpet', 'Chandapura', 'Channasandra', 'Chikka Tirupathi',
       'Chikkabanavar', 'Chikkalasandra', 'Choodasandra', 'Cooke Town',
       'Cox Town', 'Cunningham Road', 'Dasanapura', 'Dasarahalli',
       'Devanahalli', 'Devarachikkanahalli', 'Dodda Nekkundi',
       'Doddaballapur', 'Doddakallasandra', 'Doddathoguru', 'Domlur',
       'Dommasandra', 'EPIP Zone', 'Electronic City',
       'Electronic City Phase II', 'Electronics City Phase 1',
       'Frazer Town', 'GM Palaya', 'Garudachar Palya', 'Giri Nagar',
       'Gollarapalya Hosahalli', 'Gottigere', 'Green Glen Layout',
       'Gubbalala', 'Gunjur', 'HAL 2nd Stage', 'HBR Layout',
       'HRBR Layout', 'HSR Layout', 'Haralur Road', 'Harlur', 'Hebbal',
       'Hebbal Kempapura', 'Hegde Nagar', 'Hennur', 'Hennur Road',
       'Hoodi', 'Horamavu Agara', 'Horamavu Banaswadi', 'Hormavu',
       'Hosa Road', 'Hosakerehalli', 'Hoskote', 'Hosur Road', 'Hulimavu',
       'ISRO Layout', 'ITPL', 'Iblur Village', 'Indira Nagar', 'JP Nagar',
       'Jakkur', 'Jalahalli', 'Jalahalli East', 'Jigani',
       'Judicial Layout', 'KR Puram', 'Kadubeesanahalli', 'Kadugodi',
       'Kaggadasapura', 'Kaggalipura', 'Kaikondrahalli',
       'Kalena Agrahara', 'Kalyan nagar', 'Kambipura', 'Kammanahalli',
       'Kammasandra', 'Kanakapura', 'Kanakpura Road', 'Kannamangala',
       'Karuna Nagar', 'Kasavanhalli', 'Kasturi Nagar', 'Kathriguppe',
       'Kaval Byrasandra', 'Kenchenahalli', 'Kengeri',
       'Kengeri Satellite Town', 'Kereguddadahalli', 'Kodichikkanahalli',
       'Kodigehaali', 'Kodigehalli', 'Kodihalli', 'Kogilu', 'Konanakunte',
       'Koramangala', 'Kothannur', 'Kothanur', 'Kudlu', 'Kudlu Gate',
       'Kumaraswami Layout', 'Kundalahalli', 'LB Shastri Nagar',
       'Laggere', 'Lakshminarayana Pura', 'Lingadheeranahalli',
       'Magadi Road', 'Mahadevpura', 'Mahalakshmi Layout', 'Mallasandra',
       'Malleshpalya', 'Malleshwaram', 'Marathahalli', 'Margondanahalli',
       'Marsur', 'Mico Layout', 'Munnekollal', 'Murugeshpalya',
       'Mysore Road', 'NGR Layout', 'NRI Layout', 'Nagarbhavi',
       'Nagasandra', 'Nagavara', 'Nagavarapalya', 'Narayanapura',
       'Neeladri Nagar', 'Nehru Nagar', 'OMBR Layout', 'Old Airport Road',
       'Old Madras Road', 'Padmanabhanagar', 'Pai Layout', 'Panathur',
       'Parappana Agrahara', 'Pattandur Agrahara', 'Poorna Pragna Layout',
       'Prithvi Layout', 'R.T. Nagar', 'Rachenahalli',
       'Raja Rajeshwari Nagar', 'Rajaji Nagar', 'Rajiv Nagar',
       'Ramagondanahalli', 'Ramamurthy Nagar', 'Rayasandra',
       'Sahakara Nagar', 'Sanjay nagar', 'Sarakki Nagar', 'Sarjapur',
       'Sarjapur  Road', 'Sarjapura - Attibele Road',
       'Sector 2 HSR Layout', 'Sector 7 HSR Layout', 'Seegehalli',
       'Shampura', 'Shivaji Nagar', 'Singasandra', 'Somasundara Palya',
       'Sompura', 'Sonnenahalli', 'Subramanyapura', 'Sultan Palaya',
       'TC Palaya', 'Talaghattapura', 'Thanisandra', 'Thigalarapalya',
       'Thubarahalli', 'Thyagaraja Nagar', 'Tindlu', 'Tumkur Road',
       'Ulsoor', 'Uttarahalli', 'Varthur', 'Varthur Road', 'Vasanthapura',
       'Vidyaranyapura', 'Vijayanagar', 'Vishveshwarya Layout',
       'Vishwapriya Layout', 'Vittasandra', 'Whitefield',
       'Yelachenahalli', 'Yelahanka', 'Yelahanka New Town', 'Yelenahalli',
       'Yeshwanthpur', 'other']

selected_location = st.selectbox("Select a location", sorted(location))

if st.button("Predict"):
    features = {'area': area,'bhk': bhk,'bathrooms': bathrooms,'location': selected_location
        }
    price=predicted_price(area,bhk,bathrooms,selected_location)
    st.markdown(f"****Predicted Price  =   ₹ {np.round(price*100000,2)}****")
