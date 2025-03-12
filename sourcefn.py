import numpy as np
import matplotlib.pyplot as plt

def init_data():
    global wind

    fd = open("csv/wind_sv.csv", "r")
    wind = fd.readlines()
    fd.close()
    wind = [float(x.strip()) for x in wind[1:]]  # rimuove l'header e converte in float

# Funzione per generare la disponibilità di energia
def available_energy_complex(t, ampiezza): #settando l'ampiezza si setta anche il massimo che è il doppio dell'ampiezza

    T1 = 24  # Periodo della sinusoide principale
    omega1 = 2 * np.pi / T1  # Frequenza angolare della sinusoide principale
    phi1 = 1  # Fase iniziale della sinusoide principale

    T2 = 168 # Periodo della modulazione dell'ampiezza
    omega2 = 2 * np.pi / T2  # Frequenza angolare della modulazione
    phi2 = 0  # Fase iniziale della modulazione

    noise_std = 0 #0.3  Deviazione standard del rumore
    # Modulazione dell'ampiezza nel tempo
    ampiezza_modulata = 1 + ampiezza * np.sin(omega2 * t + phi2)
    
    # Componente periodica principale con ampiezza modulata
    componente_periodica = ampiezza_modulata * np.sin(omega1 * t + phi1) + ampiezza
    
    # Aggiunta del rumore gaussiano
    rumore = np.random.normal(0, noise_std, 1).item()
    
    disponibilita = componente_periodica + rumore
    return disponibilita

# Funzione più semplice per generare la disponibilità di energia
def available_energy_simple(time, maximum):
    return ((np.sin(time)+1)/2)*maximum
    
def wind_data(time):
    return wind[time]
    
init_data()