import cv2
import numpy as np
import scipy.stats

# Set default values
DIM_DEF = 4
F_DEF = np.array([
            [1, 0, 0.2, 0],
            [0, 1, 0, 0.2],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
            ])
H_DEF = np.eye(DIM_DEF)
Q_DEF = np.array([
            [0.001, 0, 0, 0],
            [0, 0.001, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
            ])
R_DEF = 0.1 * np.eye(DIM_DEF)
NOISE_MEAN_DEF = [0] * DIM_DEF
NOISE_COV_DEF = [1] * DIM_DEF

class TrackingSystem():
    """A system in 2D with 4D state vector changing in time.
    The state vector is [x, y, vx, vy], where (x, y) are coordinates, (vx, vy) are velocities in 2D
    
    Args:
        dim -- (default: 4)
        F -- (default: 4)
        H -- (default: I_4)
        R -- (default: 0.1 * I_4)
        noise_model -- (mean_vector, variance_vector / covariance_matrix)
                       parameters of multivariate normal random variable
                       default: ([0,0,0,0], [1,1,1,1])
    """
    def __init__(self, dim=DIM_DEF, F=F_DEF, H=H_DEF, Q=Q_DEF, R=R_DEF,
                 noise_model=(NOISE_MEAN_DEF, NOISE_COV_DEF)):
        self.dim = dim
        self.fps = 60
        self.pixel_size = 1
        # Real position (ground-truth, gt) vector
        self.gt = [] # all
        self.x_gt = np.zeros(self.dim) # last
        # Noisy position measurement (input, inp) vector
        self.inp = [] # all
        self.x_inp = np.zeros(self.dim) # last
        # Kalman filter (prediction, pred) state vector
        self.pred = [] # all
        self.x_pred = np.zeros(self.dim) # last
        # Matrices
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.P = np.zeros((self.dim, self.dim))
        # Noise model
        self.noise_rv = scipy.stats.multivariate_normal(*noise_model)
        print(self.noise_rv.rvs().shape)

    def run(self):
        while True:
            self.get_ground_truth()
            self.synthesize_measurement()
            self.KF_iteration()
            # self.x_inp = m[:2]
            break

    def get_ground_truth(self):
        coords_gt = np.zeros(2)
        v_gt = coords_gt - self.x_gt[:2]
        self.x_gt = np.concatenate(coords_gt, v_gt)

    def synthesize_measurement(self):
        coords_inp = self.x_gt[:2] + self.noise_rv.rvs()
        v_inp = coords_inp - self.x_pred[:2]
        self.x_inp = np.concatenate(coords_inp, v_inp)

    def KF_iteration(self):
        self.gt.append(self.x_gt)
        self.inp.append(self.x_inp)

        # Prediction
        x_hat = self.F @ self.x_pred
        P_hat = self.F @ self.P @ self.F.T + self.R

        # Correction
        K = P_hat @ self.H.T @ np.linalg.inv(self.H @ P_hat @ self.H.T + self.Q)
        self.x_pred = x_hat + K @ (self.z - self.H @ x_hat)
        self.P = (np.eye(self.dim) - K @ self.H) @ P_hat

        self.pred.append(self.x_pred)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument("--dirname", type=str, default="Jump", help="path to image and template (default: Jump)")
    # parser.add_argument("--adaptive", action='store_true', default=False, help="CAMShift")
    args = parser.parse_args()

    tracking_system = TrackingSystem()
    # tracking_system.run()