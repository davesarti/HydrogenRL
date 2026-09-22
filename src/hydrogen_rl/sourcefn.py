import numpy as np
import matplotlib.pyplot as plt
from .paths import DATA_DIR

NOISE_STD = 0  # Possible to add Gaussian noise to the function

def init_data():
    global wind

    fd = open(DATA_DIR / "wind_sv.csv", "r")
    wind = fd.readlines()
    fd.close()
    wind = [float(x.strip()) for x in wind[1:]]  # remove header and convert to float

# Function to generate wind speed
def function_complex(t, ampiezza=10):  # by setting the amplitude, you also set the maximum which is twice the amplitude

    t = t/20  # Function dilation

    T1 = 24  # Period of the main sinusoid
    omega1 = 2 * np.pi / T1  # Angular frequency of the main sinusoid
    phi1 = 1  # Initial phase of the main sinusoid

    T2 = 168  # Period of the amplitude modulation
    omega2 = 2 * np.pi / T2  # Angular frequency of the modulation
    phi2 = 0  # Initial phase of the modulation

    ampiezza_modulata = 1 + ampiezza * np.sin(omega2 * t + phi2)
    
    # Main periodic component with modulated amplitude
    componente_periodica = ampiezza_modulata * np.sin(omega1 * t + phi1) + ampiezza
    
    # Addition of Gaussian noise
    rumore = np.random.normal(0, NOISE_STD, 1).item()
    
    disponibilita = componente_periodica + rumore
    return disponibilita

# Simpler function for testing
def function_simple(time, maximum):
    return ((np.sin(time)+1)/2)*maximum
    
def wind_data(time):
    return wind[time]
    
init_data()