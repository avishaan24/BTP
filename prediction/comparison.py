import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def estimateState():
    n = 2  
    m = 2 

    # initial x value
    x_0 = np.zeros((n, 1))
    x_0[0, 0] = 1.1
    x_0[1, 0] = 2.1
    mngm = MNGM2(500, csv_file)
    mngm.generate_data()

    ukf = UKF(n, m)
    ekf = EKF(n, m)
    dataX = mngm.x
    dataY = mngm.y
    size_n = dataX.shape[0]

    ukf.resetUKF(3.0, 1.0, x_0)
    ekf.resetEKF(3.0, 1.0, x_0)

    err_ukf = 0
    err_ekf = 0
    est_state_ukf = np.zeros((size_n, n))
    est_y_ukf = np.zeros((size_n, m))
    est_P_ukf = np.zeros((size_n, n * 2))
    est_Py_ukf = np.zeros((size_n, n * 2))

    est_state_ekf = np.zeros((size_n, n))
    est_y_ekf = np.zeros((size_n, m))
    est_P_ekf = np.zeros((size_n, n * 2))
    est_Py_ekf = np.zeros((size_n, n * 2))

    for i in range(size_n):
        timeUpdateInput = i
        measurementUpdateInput = dataY[i, :]
        # recursively go through time update and measurement correction
        ekf.timeUpdate(timeUpdateInput)
        ekf.measurementUpdate(measurementUpdateInput)

        ukf.timeUpdate(timeUpdateInput)
        ukf.measurementUpdate(measurementUpdateInput)

        err_ukf = err_ukf + np.sum((ukf.x_aposteriori - dataX[i, :]) ** 2)
        err_ekf = err_ekf + np.sum((ekf.x_aposteriori - dataX[i, :]) ** 2)

        est_state_ukf[i, :] = ukf.x_aposteriori
        est_P_ukf[i, :] = ukf.P_aposteriori.flatten()
        est_Py_ukf[i, :] = ukf.P_y.flatten()
        est_y_ukf[i, :] = ukf.y

        est_state_ekf[i, :] = ekf.x_aposteriori
        est_P_ekf[i, :] = ekf.P_aposteriori.flatten()
        est_Py_ekf[i, :] = ekf.P_y.flatten()
        est_y_ekf[i, :] = ekf.y

    print("UKF Total Error:", err_ukf)
    print("EKF Total Error:", err_ekf)
    # Plotting: x1 Original vs EKF vs UKF
    plt.figure(figsize=(12, 6))
    # Plot for x1
    plt.subplot(1, 2, 1)
    plt.plot(dataX[:, 0], 'g', label='x1 Original')
    plt.plot(est_state_ukf[:, 0], 'r--', label='x1 UKF')
    plt.plot(est_state_ekf[:, 0], 'b--', label='x1 EKF')
    plt.title("x1 Original vs EKF vs UKF")
    plt.xlabel("Time Steps")
    plt.ylabel("x1 Value")
    plt.legend(loc='upper right')

    # Plot for x2
    plt.subplot(1, 2, 2)
    plt.plot(dataX[:, 1], 'g', label='x2 Original')
    plt.plot(est_state_ukf[:, 1], 'r--', label='x2 UKF')
    plt.plot(est_state_ekf[:, 1], 'b--', label='x2 EKF')
    plt.title("x2 Original vs EKF vs UKF")
    plt.xlabel("Time Steps")
    plt.ylabel("x2 Value")
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

    plot_confidence_ellipse_cov_all(dataX, dataY,
                                     est_state_ekf, est_y_ekf, est_P_ekf, est_Py_ekf,
                                     est_state_ukf, est_y_ukf, est_P_ukf, est_Py_ukf)


def plot_confidence_ellipse_cov_all(dataX, dataY, est_state_ekf, est_y_ekf, P_ekf, Py_ekf, est_state_ukf, est_y_ukf, P_ukf, Py_ukf):
    n = dataX.shape[0]

    sig = np.zeros((n, 4))  # x0 x1 y0 y1
    sig_mean = np.zeros(4)  # mean(x0) mean(x1) mean(y0) mean(y1)

    est_mean_ekf = np.mean(est_state_ekf, axis=0)
    est_y_mean_ekf = np.mean(est_y_ekf, axis=0)
    est_mean_ukf = np.mean(est_state_ukf, axis=0)
    est_y_mean_ukf = np.mean(est_y_ukf, axis=0)

    for i in range(2):
        sig[:, i] = dataX[:, i]
        sig[:, i + 2] = dataY[:, i]

        sig_mean[i] = np.mean(sig[:, i])
        sig_mean[i + 2] = np.mean(sig[:, i + 2])

    # cov of EKF P
    cov1 = np.zeros((2, 2))
    cov1[0, 0] = np.mean(P_ekf[:, 0])
    cov1[0, 1] = np.mean(P_ekf[:, 1])
    cov1[1, 0] = np.mean(P_ekf[:, 2])
    cov1[1, 1] = np.mean(P_ekf[:, 3])
    lambda_p_ekf, v = np.linalg.eig(cov1)
    lambda_p_ekf = np.sqrt(lambda_p_ekf)
    vvv = v[:, 0][::-1]
    aaa = np.arctan2(vvv[0], vvv[1])
    angle_p_ekf = np.arctan2(*v[:, 0][::-1])

    # cov of UKF P
    cov1 = np.zeros((2, 2))
    cov1[0, 0] = np.mean(P_ukf[:, 0])
    cov1[0, 1] = np.mean(P_ukf[:, 1])
    cov1[1, 0] = np.mean(P_ukf[:, 2])
    cov1[1, 1] = np.mean(P_ukf[:, 3])
    lambda_p_ukf, v = np.linalg.eig(cov1)
    lambda_p_ukf = np.sqrt(lambda_p_ukf)
    vvv = v[:, 0][::-1]
    aaa = np.arctan2(vvv[0], vvv[1])
    angle_p_ukf = np.arctan2(*v[:, 0][::-1])

    # signal cov
    angle = np.zeros(2)
    lambda_ = np.zeros(4)
    # first ellipse:
    cov1 = np.cov(sig[:, 0], sig[:, 1])
    lambda_1, v = np.linalg.eig(cov1)
    lambda_1 = np.sqrt(lambda_1)
    lambda_[0:2] = lambda_1
    angle[0] = np.arctan2(*v[:, 0][::-1])

    # second ellipse:
    cov2 = np.cov(sig[:, 2], sig[:, 3])
    lambda_2, v = np.linalg.eig(cov2)
    lambda_2 = np.sqrt(lambda_2)
    lambda_[2:4] = lambda_2
    angle[1] = np.arctan2(*v[:, 0][::-1])

    fig, axs = plt.subplots(1, 1, figsize=(6, 3)) 
    ax = axs  

    ax.scatter(sig[:, 0], sig[:, 1], s=0.9)
    ax.axvline(c='grey', lw=1)
    ax.axhline(c='grey', lw=1)

    # Plot for EKF state covariance
    ell_p_ekf = Ellipse(xy=(est_mean_ekf[0], est_mean_ekf[1]), width=lambda_p_ekf[0] * 6, height=lambda_p_ekf[1] * 6,
                        angle=np.rad2deg(angle_p_ekf), edgecolor='blue', label='EKF Covariance')
    ell_p_ekf.set_facecolor('none')
    ax.add_artist(ell_p_ekf)
    ax.scatter(est_mean_ekf[0], est_mean_ekf[1], c='blue', s=3)

    # Plot for UKF state covariance
    ell_p_ukf = Ellipse(xy=(est_mean_ukf[0], est_mean_ukf[1]), width=lambda_p_ukf[0] * 6, height=lambda_p_ukf[1] * 6,
                        angle=np.rad2deg(angle_p_ukf), edgecolor='green', label='UKF Covariance')
    ell_p_ukf.set_facecolor('none')
    ax.add_artist(ell_p_ukf)
    ax.scatter(est_mean_ukf[0], est_mean_ukf[1], c='green', s=3)

    # Plot the signal covariance
    ell = Ellipse(xy=(sig_mean[0], sig_mean[1]), width=lambda_[0] * 6, height=lambda_[1] * 6,
                  angle=np.rad2deg(angle[0]), edgecolor='firebrick', label='Real Signal Covariance')
    ell.set_facecolor('none')
    ax.add_artist(ell)
    ax.scatter(sig_mean[0], sig_mean[1], c='red', s=3)

    ax.set_title('Covariance State Variables')
    ax.legend()  
    plt.show()
