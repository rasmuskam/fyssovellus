import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from scipy.signal import find_peaks, butter, filtfilt

# Suodatusfunktio
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# Otsikko sovellukselle
st.title("Urheilusovellus")

# Tiedostojen lataus
accel_file = st.file_uploader("Lataa kiihtyvyysdata (Accelerometer.csv)", type=["csv"])
gps_file = st.file_uploader("Lataa GPS-data (Location.csv)", type=["csv"])
time_file = st.file_uploader("Lataa aikadata (time.csv)", type=["csv"])

if accel_file and gps_file and time_file:
    # Lue data
    accel_data = pd.read_csv(accel_file)
    gps_data = pd.read_csv(gps_file)
    time_data = pd.read_csv(time_file)

    st.write("Ensimmäiset rivit kiihtyvyysdatasta:")
    st.write(accel_data.head())

    st.write("Ensimmäiset rivit GPS-datasta:")
    st.write(gps_data.head())

    st.write("Ensimmäiset rivit aikadatasta:")
    st.write(time_data.head())

    # Valitaan analysoitava kiihtyvyyden komponentti
    component = st.selectbox("Valitse analysoitava kiihtyvyyden komponentti:", ['X (m/s^2)', 'Y (m/s^2)', 'Z (m/s^2)'])

    # Suodatetaan kiihtyvyysdata valitun komponentin mukaan
    fs = 50.0  # Näytteenottotaajuus
    cutoff = 3.0  # Korkeusraja (Hz)
    filtered_data = lowpass_filter(accel_data[component], cutoff, fs)

    # Etsitään askelia piikeistä
    peaks, _ = find_peaks(filtered_data, height=0.5, distance=20)
    step_count = len(peaks)
    st.write(f"Askelmäärä suodatetusta kiihtyvyysdatasta ({component}): {step_count}")

    # Fourier-analyysi
    freq = np.fft.fftfreq(len(filtered_data), d=1/fs)
    fft_result = np.fft.fft(filtered_data)
    psd = np.abs(fft_result)**2

    # Askellaulukko Fourier-analyysin perusteella
    step_count_fourier = (psd > 5.0).sum()  # Kynnysarvoa voi säätää
    st.write(f"Askelmäärä Fourier-analyysin perusteella: {step_count_fourier}")

    # GPS-datan analyysi
    if 'latitude' in gps_data.columns and 'longitude' in gps_data.columns and 'Velocity (m/s)' in gps_data.columns:
        average_velocity = gps_data['Velocity (m/s)'].mean()
        gps_data['distance'] = gps_data['Velocity (m/s)'] * gps_data['Time (s)'].diff().fillna(0)
        total_distance = gps_data['distance'].sum()

        st.write(f"Keskinopeus (GPS-datasta): {average_velocity:.2f} m/s")
        st.write(f"Kuljettu matka (GPS-datasta): {total_distance:.2f} m")

        # Askelpituuden laskeminen
        if step_count > 0:
            step_length = total_distance / step_count
            st.write(f"Askelpituus: {step_length:.2f} m")
        else:
            st.write("Askelmäärä on nolla, joten askelpituuden laskeminen ei onnistu.")

        # Visualisoidaan suodatettu kiihtyvyysdata
        st.line_chart(filtered_data, use_container_width=True)

        # Tehospektritiheys visualisointi
        st.subheader("Tehospektritiheys")
        plt.figure(figsize=(10, 6))
        plt.plot(freq, psd)
        plt.title("Tehospektritiheys")
        plt.xlabel("Taajuus [Hz]")
        plt.ylabel("Teho")
        plt.xlim(0, 10)  # Voit säätää rajoja tarpeen mukaan
        st.pyplot(plt)

        # Reittikartta
        st.subheader("Reittikartta")
        map_location = [gps_data['latitude'].mean(), gps_data['longitude'].mean()]
        my_map = folium.Map(location=map_location, zoom_start=15)
        coordinates = gps_data[['latitude', 'longitude']].dropna().values.tolist()
        if len(coordinates) > 1:
            folium.PolyLine(coordinates, color="blue", weight=2.5, opacity=1).add_to(my_map)

        st.components.v1.html(my_map._repr_html_(), height=500)
    else:
        st.error("GPS-datasta puuttuu tarvittavat sarakkeet (latitude, longitude, Velocity (m/s)). Tarkista CSV-tiedosto.")