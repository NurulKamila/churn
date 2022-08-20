from unicodedata import category
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
st.write ("""
# Prediksi Churn Pelanggan Perusahaan Telekomunikasi
Aplikasi berbasis Web untuk **memprediksi churn pelanggan perusahaan telekomunikasi** menggunakan pendekatan machine learning yaitu algoritma Logistic Regression . 
""")

img = Image.open('log.png')
img = img.resize((700,418))
st.image(img, use_column_width=False)

img2 = Image.open('log2.jpeg')
img2 = img2.resize((700,451))
st.image(img2, use_column_width=False)

st.sidebar.header('Parameter Inputan')

# Upload file csv untuk parameter inputan
upload_file = st.sidebar.file_uploader("Upload file CSV Anda", type=["csv"])
if upload_file is not None:
    inputan = pd.read_csv(upload_file)
else:
    def input_user():
        Married = st.sidebar.selectbox('Married', ('Yes', 'No'))
        Number_of_Dependents = st.sidebar.slider('Number_of_Dependents', 0, 5, 9)
        Tenure_in_Months = st.sidebar.slider('Tenure_in_Months', 1, 24, 80)
        Phone_Service = st.sidebar.selectbox('Phone_Service', ('Yes', 'No'))
        Multipl_Lines = st.sidebar.selectbox('Multipl_Lines', ('Yes', 'No'))
        Internet_Service = st.sidebar.selectbox('Internet_Service', ('Yes', 'No'))
        Internet_Type = st.sidebar.selectbox('Internet_Type', ('Cable', 'DSL', 'Fiber Optic'))
        Online_Security = st.sidebar.selectbox('online_Security', ('Yes', 'No'))
        Online_Backup = st.sidebar.selectbox('Online_Backup', ('Yes', 'No'))
        Streaming_TV = st.sidebar.selectbox('Streaming_TV', ('Yes', 'No'))
        Streaming_Movies = st.sidebar.selectbox('Streaming_Movies', ('Yes', 'No'))
        Streaming_Music = st.sidebar.selectbox('Streaming_Music', ('Yes', 'No'))
        Unlimited_Data = st.sidebar.selectbox('Unlimited_Data', ('Yes', 'No'))
        Contract = st.sidebar.selectbox('Contract', ('Month-to-month', 'One Year', 'Two Year'))
        Paperless_Billing = st.sidebar.selectbox('Paperless_Billing', ('Yes', 'No'))
        Payment_Method = st.sidebar.selectbox('Payment_Method', ('Bank Withdrawal', 'Credit Card', 'Mailed Check'))
        data = {
                'Married': Married,
                'Number_of_Dependents': Number_of_Dependents,
                'Tenure_in_Months': Tenure_in_Months,
                'Phone_Service': Phone_Service,
                'Multipl_Lines': Multipl_Lines,
                'Internet_Service': Internet_Service,
                'Internet_Type': Internet_Type,
                'Online_Security': Online_Security,
                'Online_Backup': Online_Backup,
                'Streaming_TV': Streaming_TV,
                'Streaming_Movies': Streaming_Movies,
                'Streaming_Music': Streaming_Music,
                'Unlimited_Data': Unlimited_Data,
                'Contract': Contract,
                'Paperless_Billing': Paperless_Billing,
                'Payment_Method': Payment_Method}
        fitur = pd.DataFrame(data, index=[0])
        return fitur
    inputan = input_user()

# Menggabungkan inputan dan dataset churn
churn_raw = pd.read_csv('churn_pelanggan_bersih.csv')

churn = churn_raw.drop(columns=['Customer_Status'])
data = pd.concat([inputan, churn], axis=0)
# data2 = pd.concat([datanew, churn], axis=0)
df = data.iloc[:, :-1]
# dt = data2.iloc[:, :-1]
    

# Encode untuk fitur ordinal
encoder = LabelEncoder()
# df['Gender'] = encoder.fit_transform(df['Gender'])
for column in df.columns:
    if df[column].dtype == np.number:
        continue
    # else:
        # churn[column].dtype== "category"
    df[column] = encoder.fit_transform(df[column])
# df = df[:1]
# # Encode untuk fitur ordinal
#     encoder = LabelEncoder()    
#     for column in dt.columns:
#         if dt[column].dtype == np.number:
#             continue
#         # else:
#             # churn[column].dtype== "category"
#         dt[column] = encoder.fit_transform(dt[column])
#         # df = df.drop(columns=['Unnamed:0'])
#     dt = dt[:1]
# Menampilkan parameter hasil inputan
st.subheader('Parameter Inputan')
st.write(data)
if upload_file is not None:
    st.write(df)
else:
    st.write('Menunggu file csv untuk diupload. Saat ini memakai sampel inputan (seperti tampilan dibawah).')
    st.write(df)

# Load model Logistic Regression
load_model = pickle.load(open('modelLog_churn.pkl', 'rb'))

# Terapkan modelnya
prediksi = load_model.predict(df)
prediksi_proba = load_model.predict_proba(df)


st.subheader('Keterangan Label Kelas')
jenis_pelanggan = np.array(['Churned','Stayed'])
st.write(jenis_pelanggan)

st.subheader('Hasil Prediksi Churn Pelanggan')
st.write(jenis_pelanggan[prediksi])

st.subheader('Probabilitas Hasil Prediksi Churn Pelanggan')
st.write(prediksi_proba)

