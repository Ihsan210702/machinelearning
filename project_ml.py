import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score

st.write("""<h2>Pengelompokkan Penduduk Usia Lanjut Berdasarkan Kecamatan di Kota Surabaya Menggunakan K-Medoids Clustering</h2>""",unsafe_allow_html=True)

with st.container():
    with st.sidebar:
        selected = option_menu(
        st.write("""<h3 style = "text-align: center;"><img src="https://www.analyticsvidhya.com/wp-content/uploads/2016/11/clustering.png" width="120" height="100"><br> 
        Kelompok 9 <p>Machine Learning</p></h3>""",unsafe_allow_html=True), 
        ["Home", "Data", "Prepocessing", "Clustering"], 
            icons=['house', 'file-earmark-font', 'bar-chart', 'gear', 'arrow-down-square', 'check2-square'], menu_icon="cast", default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "black", "font-size": "18px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px"},
                "nav-link-selected": {"background-color": "Cyan"},
            }
        )

    if selected == "Home":
        img = Image.open('peta_surabaya.png')
        st.image(img, use_column_width=False, width=500)

        st.write("""
        Badan Pusat Statistik (BPS) Kabupaten Bangkalan merilis hasil Sensus Penduduk Tahun 2020 per September 2020. Sesuai data yang dipublikasikan pada 26 Januari 2020, bahwa populasi penduduk Kabupaten Bangkalan sebanyak 1.060.377 jiwa.

Agip Yunaidi Solichin, Koordinator Fungsi Statistik Sosial BPS Bangkalan, mengatakan jumlah tersebut  meningkat sebanyak 153.616 jiwa (16,95%) dari jumlah penduduk 10 tahun yang lalu (2010) yang tercatat sebanyak 906.761 jiwa.
        """)

    elif selected == "Data":
        st.subheader("""Dataset Penduduk Lansia Surabaya""")
        df = pd.read_csv('https://raw.githubusercontent.com/Ihsan210702/dataset/main/datalansia_surabaya.csv')
        st.dataframe(df)
    elif selected == "Prepocessing":
        st.subheader("""Normalisasi Data""")
        st.write("""Rumus Normalisasi Data :""")
        st.image('https://i.stack.imgur.com/EuitP.png', use_column_width=False, width=250)
        #Read Dataset
        df = pd.read_csv('https://raw.githubusercontent.com/Ihsan210702/dataset/main/datalansia_surabaya.csv')
        st.markdown("""
        Dimana :
        - X = data yang akan dinormalisasi atau data asli
        - min = nilai minimum semua data asli
        - max = nilai maksimum semua data asli
        """)
        #Mendefinisikan Varible X dan Y
        X = df.drop(columns=['No', 'Kecamatan'])

        #NORMALISASI NILAI KAMAL
        scaler = MinMaxScaler()
        #scaler.fit(features)
        #scaler.transform(features)
        scaled = scaler.fit_transform(X)
        features_names = X.columns.copy()
        #features_names.remove('label')
        data = pd.DataFrame(scaled, columns=features_names)

        st.subheader('Hasil Normalisasi Data Kecamatan Wilayah Kota Surabaya')
        st.write(data)
    elif selected =="Clustering":
        with st.form("my_form"):
            st.subheader("Implementasi Clustering Penduduk Lansia Kota Surabaya")
            Clustering = st.slider('Jumlah Cluster', 2, 4)
            
            prediksi = st.form_submit_button("Submit")
            if prediksi:
                inputs = np.array([
                    Clustering
                ])
                df = pd.read_csv('https://raw.githubusercontent.com/Ihsan210702/dataset/main/datalansia_surabaya.csv')
                X = df.drop(columns=['No','Kecamatan'])
                y = df['Kecamatan'].values
                
                df_min = X.min()
                df_max = X.max()

                #NORMALISASI DATA KAMAL
                scaler = MinMaxScaler()
                #scaler.fit(features)
                #scaler.transform(features)
                scaled = scaler.fit_transform(X)
                features_names = X.columns.copy()
                #features_names.remove('label')
                data = pd.DataFrame(scaled, columns=features_names)

                kecamatan = pd.DataFrame({'Kecamatan': y})
                cluster_K = KMedoids (n_clusters=Clustering,metric="euclidean",init="random")
                fitted_K = cluster_K.fit_predict(data)

                cluster = pd.DataFrame({'Cluster': fitted_K})
                result = pd.concat([kecamatan, cluster], axis=1)
                result
                st.success(f'Silhouette Score: {silhouette_score(data, fitted_K)}')
        
