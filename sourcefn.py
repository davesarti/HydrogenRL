import numpy as np
import matplotlib.pyplot as plt

# Funzione per generare la disponibilità di energia
def available_energy_complex(t, ampiezza): #settando l'ampiezza si setta anche il massimo che è il doppio dell'ampiezza
    # Parametri della funzione
    T1 = 24  # Periodo della sinusoide principale (ad esempio, 24 ore)
    omega1 = 2 * np.pi / T1  # Frequenza angolare della sinusoide principale
    phi1 = 1  # Fase iniziale della sinusoide principale

    T2 = 168  # Periodo della modulazione dell'ampiezza (ad esempio, 7 giorni = 168 ore)
    omega2 = 2 * np.pi / T2  # Frequenza angolare della modulazione
    phi2 = 0  # Fase iniziale della modulazione

    noise_std = 20 # Deviazione standard del rumore
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

def main():
    t = np.linspace(0, 336, 1000)

    disponibilita = available_energy_complex(t, 250)

    plt.figure(figsize=(12, 6))
    plt.plot(t, disponibilita, label='Disponibilità di Energia', color='blue')
    plt.xlabel('Tempo (ore)')
    plt.ylabel('Disponibilità di Energia')
    plt.title('Disponibilità di energia periodica con creste variabili')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()