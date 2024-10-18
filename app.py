import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from scipy.signal import find_peaks, butter, filtfilt

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

st.title("Urheilusovellus")

accel_file = st.file_uploader("Lataa kiihtyvyysdata (Accelerometer.csv)", type=["csv"])
gps_file = st.file_uploader("Lataa GPS-data (Location.csv)", type=["csv"])

if accel_file and gps_file:
    accel_data = pd.read_csv(accel_file)
    gps_data = pd.read_csv(gps_file)
    component = 'Y (m/s^2)'

    fs = 50.0
    cutoff = 3.0  
    filtered_data = lowpass_filter(accel_data[component], cutoff, fs)

    peaks, _ = find_peaks(filtered_data, height=0.5, distance=20)
    step_count = len(peaks)
    st.write(f"Askelmäärä suodatetusta kiihtyvyysdatasta ({component}): {step_count}")

    freq = np.fft.fftfreq(len(filtered_data), d=1/fs)
    fft_result = np.fft.fft(filtered_data)
    psd = np.abs(fft_result)**2
    fourier_peaks, _ = find_peaks(psd, height=0.5)
    fourier_step_count = len(fourier_peaks)
    st.write(f"Askelmäärä laskettuna Fourier-analyysin perusteella: {fourier_step_count}")

    if 'latitude' in gps_data.columns and 'longitude' in gps_data.columns and 'Velocity (m/s)' in gps_data.columns:
        average_velocity = gps_data['Velocity (m/s)'].mean()
        gps_data['distance'] = gps_data['Velocity (m/s)'] * gps_data['Time (s)'].diff().fillna(0)
        total_distance = gps_data['distance'].sum()

        st.write(f"Keskinopeus (GPS-datasta): {average_velocity:.2f} m/s")
        st.write(f"Kuljettu matka (GPS-datasta): {total_distance:.2f} m")

        if step_count > 0:
            step_length = total_distance / step_count
            st.write(f"Askelpituus: {step_length:.2f} m")
        else:
            st.write("Askelmäärä on nolla, joten askelpituuden laskeminen ei onnistu.")

        st.subheader("Suodatettu kiihtyvyysdata")
        st.line_chart(filtered_data, use_container_width=True)

        st.subheader("Tehospektritiheys")
        plt.figure(figsize=(10, 6))
        plt.plot(freq, psd)
        plt.title("Tehospektritiheys")
        plt.xlabel("Taajuus [Hz]")
        plt.ylabel("Teho")
        plt.xlim(0, 10)
        st.pyplot(plt)

        st.subheader("Reittikartta")
        map_location = [gps_data['latitude'].mean(), gps_data['longitude'].mean()]
        my_map = folium.Map(location=map_location, zoom_start=15)
        coordinates = gps_data[['latitude', 'longitude']].dropna().values.tolist()
        if len(coordinates) > 1:
            folium.PolyLine(coordinates, color="blue", weight=2.5, opacity=1).add_to(my_map)

        st.components.v1.html(my_map._repr_html_(), height=500)
    else:
        st.error("GPS-datasta puuttuu tarvittavat sarakkeet (latitude, longitude, Velocity (m/s)). Tarkista CSV-tiedosto.")