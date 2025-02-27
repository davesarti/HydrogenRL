import numpy as np
import matplotlib.pyplot as plt

def init_data():
    global power, power_sv, wind

    fd = open("csv/power.csv", "r")
    power = fd.readlines()
    fd.close()
    power = [float(x) for x in power[1:]]

    fd = open("csv/wind.csv", "r")
    wind = fd.readlines()
    fd.close()
    wind = [float(x) for x in wind[1:]]

    try:
        fd = open("csv/power_sv.csv", "r")

    except FileNotFoundError:
        window = 50
        power_sv = np.convolve(power, np.ones(window)/window, mode='valid')

        fd=open("csv/power_sv.csv", "w")
        fd.write("LV ActivePower (kW) Smoothed Values\n")
        for p in power_sv:
            fd.write(str(p)+"\n")
        print("File power_sv.csv creato")
        fd.close()

        fd = open("power_sv.csv", "r")
    
    power_sv = fd.readlines()
    fd.close()
    power_sv = [float(x) for x in power_sv[1:]]

# Funzione per generare la disponibilità di energia
def available_energy_complex(t, ampiezza): #settando l'ampiezza si setta anche il massimo che è il doppio dell'ampiezza
    # Parametri della funzione
    T1 = 24  # Periodo della sinusoide principale (ad esempio, 24 ore)
    omega1 = 2 * np.pi / T1  # Frequenza angolare della sinusoide principale
    phi1 = 1  # Fase iniziale della sinusoide principale

    T2 = 168  # Periodo della modulazione dell'ampiezza (ad esempio, 7 giorni = 168 ore)
    omega2 = 2 * np.pi / T2  # Frequenza angolare della modulazione
    phi2 = 0  # Fase iniziale della modulazione

    noise_std = 10 # Deviazione standard del rumore
    # Modulazione dell'ampiezza nel tempo
    ampiezza_modulata = 1 + ampiezza * np.sin(omega2 * t + phi2)
    
    # Componente periodica principale con ampiezza modulata
    componente_periodica = ampiezza_modulata * np.sin(omega1 * t + phi1) + ampiezza
    
    # Aggiunta del rumore gaussiano
    rumore = np.random.normal(0, noise_std, 1).item()
    
    disponibilita = componente_periodica + rumore
    
    return disponibilita

def available_energy_simple(time, maximum):
    return ((np.sin(time)+1)/2)*maximum

def available_energy_data(time): #sostituire power_sv con power se si vogliono utilizzare dati non smoothed
    try:
        return power_sv[time]
    except NameError:
        init_data()
        return power_sv[time]
    
def wind_data(time):
    try:
        return wind[time]
    except NameError:
        init_data()
        return wind[time]
    
def power_max():
    try:
        return max(power_sv)
    except NameError:
        init_data()
        return max(power_sv)

def main():

    init_data()

    plot_power_sv = power_sv[1:10000]
    t = np.arange(0, len(plot_power_sv))
    plot_power = power[:len(plot_power_sv)]

    plt.figure(figsize=(12, 6))
    plt.plot(t, plot_power, label='Disponibilità di energia', color='blue')
    plt.plot(t, plot_power_sv, label='Tendenza disponibilità di energia', color='red')
    plt.xlabel('Tempo (ore)')
    plt.ylabel('Disponibilità di energia')
    plt.title('Disponibilità di energia elettrica con creste variabili')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()