import numpy as np
import pandas as pd

csv_file = "./data/HA4_47.65_-122.45_2006_DPV_9MW_60_Min.csv"

class MNGM2:
    # class used for generation of multivariate nonstationary growth model
    def __init__(self, n, csv_file):

        self.n = n
        self.x = np.zeros((n, 2))
        self.y = np.zeros((n, 2))

        self.xi = np.zeros(2)
        self.yi = np.zeros(2)

        # Read data from CSV file
        self.data = pd.read_csv(csv_file)
        self.power = self.data['Power(MW)'].values  # Extract Power data from CSV

        self.x[0, :] = self.initial_state(self.power[0])

        self.u = np.random.normal(0., 1.0, (n, 2))
        self.v = np.random.normal(0., 1.0, (n, 2))

        self.F = np.zeros((2, 2))  # jacobian matrix of the state f model
        self.H = np.zeros((2, 2))  # jacobian matrix of the output h model

    def initial_state(self, power_value):
        # Initialize state based on the first power value
        x0 = np.zeros(2)
        x0[0] = power_value * 0.5  
        x0[1] = power_value * 0.3
        return x0

    def generate_data(self):
        for i in range(1, self.n):
            self.x[i, :] = self.state(i, self.x[i-1, :]) + self.u[i, :]
            self.y[i, :] = self.output(self.x[i, :]) + self.v[i, :]

    def state(self, i, xp):
        # Modify the state based on the power data (assuming it affects the system)
        power_value = self.power[i % len(self.power)]  # Wrap around if needed

        self.xi[0] = 0.5 * xp[0] - 0.1 * xp[1] + 0.7 * (xp[0] / (1 + xp[0] ** 2)) + 2.2 * np.cos(1.2 * (i - 1)) + 1.5 + 0.1 * power_value
        self.xi[1] = 0.8 * xp[1] - 0.2 * xp[0] + 0.9 * (xp[1] / (1 + xp[1] ** 2)) + 2.4 * np.cos(2.2 * (i - 1)) - 1.5 + 0.05 * power_value

        return self.xi

    def output(self, xi):
        self.yi[0] = (xi[0] ** 2) / 18.0 - (xi[1] ** 2) / 24.0
        self.yi[1] = (xi[0] ** 2) / 17.0 + (xi[1] ** 2) / 2.0

        return self.yi

    def f_jacob(self, xi):
        self.F[0, 0] = 0.5 + 0.7 * (1 - xi[0] ** 2) / ((1 + xi[0] ** 2) ** 2)
        self.F[0, 1] = -0.1
        self.F[1, 0] = -0.2
        self.F[1, 1] = 0.8 + 0.9 * (1 - xi[1] ** 2) / ((1 + xi[1] ** 2) ** 2)

        return self.F

    def h_jacob(self, xi):
        self.H[0, 0] = 2.0 * xi[0] / 18.0
        self.H[0, 1] = -2.0 * xi[1] / 24.0
        self.H[1, 0] = 2.0 * xi[0] / 17.0
        self.H[1, 1] = 2.0 * xi[1] / 2.0

        return self.H
