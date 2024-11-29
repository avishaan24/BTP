import numpy as np
import matplotlib.pyplot as plt
import csv
import random
import pandas as pd
from datetime import datetime, timedelta


def estimatePredictedEnergyUKF():
    n = 2  
    m = 2
    # Initial state vector
    x_0 = np.zeros((n, 1))
    x_0[0, 0] = 0.1  # Initial energy/power
    x_0[1, 0] = 0.1  # Initial rate of change of energy

    mngm = MNGM2(1000, csv_file)
    mngm.generate_data()
    ukf = UKF(n, m)
    # Generated data
    dataX = mngm.x
    dataY = mngm.y
    size_n = dataX.shape[0]
    ukf.resetUKF(0.1, 0.1, x_0)
    timeUpdateInput = np.zeros((n, 1))
    measurementUpdateInput = np.zeros((m, 1))

    predicted_energy = [] 
    # Estimation loop
    for i in range(size_n):
        timeUpdateInput = i
        measurementUpdateInput = dataY[i, :]
        # Time update and measurement correction
        ukf.timeUpdate(timeUpdateInput)
        ukf.measurementUpdate(measurementUpdateInput)
        # Append predicted energy (x[0]) to the list
        predicted_energy.append(ukf.x_aposteriori[0])

    predicted_energy = np.array(predicted_energy)
    return predicted_energy

predicted_energy = estimatePredictedEnergyUKF()

num_data_points = 1000
start_time = datetime.now()
data = []

for i in range(num_data_points):
    local_time = (start_time + timedelta(minutes=i)).strftime("%m/%d/%y %H:%M")
    SINR = random.uniform(5, 30)  # Example SINR values in dB
    d = random.uniform(1, 100)    # Distance in meters
    E = random.uniform(0.1, 10)   # Residual energy in Joules
    Sreq = random.randint(1, 5)   # Minimum security level requirement
    Sth = random.randint(1, 10)   # Maximum allowable security level
    t = random.randint(1, 5)      # Duration of each time interval in seconds

    data.append([local_time, SINR, d, E, Sreq, Sth, t])

# Append predictions to the data
for i, E_pred in enumerate(predicted_energy):
    data[i].append(E_pred)

header = ["local_time", "SINRr", "d", "Er", "Sreq", "Sth", "t", "Eh(i)"]

# Write data to a CSV file
output_file = "ukf_energy_data.csv"

with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(data)

print(f"Data saved to {output_file}")
df = pd.DataFrame(data, columns=header)
print(df.head())
