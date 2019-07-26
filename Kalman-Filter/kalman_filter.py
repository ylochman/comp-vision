import cv2
import numpy as np
import scipy.stats
import threading

# Default values
SIZE = (512, 512) # window size
PRED_SCALE = 5 # multiplier for better prediction visualization
DIM_MEAS_DEF = 2 # 2 coordinates (x,y)
DIM_STATE_DEF = 4 # 2 coordinates (x,y) + 2 velocities (vx, vy)
NOISE_MEAN_DEF = [0] * DIM_MEAS_DEF # mean vector for the noise
NOISE_COV_DEF = [100] * DIM_MEAS_DEF # variance vector for the noise
# State transition matrix
F_DEF = np.array([
            [1, 0, 0.2, 0],
            [0, 1, 0, 0.2],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
            ])
# Sensor function
H_DEF = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
            ])
# Sensor noise covariance matrix
Q_DEF = 0.1 * np.eye(DIM_STATE_DEF)
# Action uncertainty
R_DEF = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0.005, 0],
            [0, 0, 0, 0.005]
            ])

class KalmanFilter(object):
    """Implementation of Kalman filter that operates in a 2D system with a 4D state vector.
    The state vector is by default [x, y, vx, vy],
    where (x, y) are coordinates and (vx, vy) are velocities in 2D.
    For another state vector a function `getdata()` should be rewritten.
    
    Args:
        dim -- state vector dimensionality
        F -- state transition matrix
        H -- sensor function
        Q -- sensor noise, covariance matrix
        R -- action uncertainty
    """
    def __init__(self, dim=DIM_STATE_DEF, F=F_DEF, H=H_DEF, Q=Q_DEF, R=R_DEF):
        self.dim = dim
        
        # Real (ground-truth) vector (this is unknow in the wild therefore optional)
        self.gt = [] # for accumulation
        self.x_gt = np.zeros(self.dim, dtype='int') # current
        
        # Noisy measurement (input) vector
        self.inp = [] # for accumulation
        self.x_inp = np.zeros(self.dim, dtype='int') # current

        # Kalman filter (prediction) state: mean vector x_pred, and covariance P
        self.pred = [] # for accumulation
        self.x_pred = np.zeros(self.dim) # current vector
        self.P = np.zeros((self.dim, self.dim)) # current covariance (not accumulated)
        
        # System Matrices
        self.F = F # state transition matrix
        self.H = H # "sensor" function
        self.Q = Q # "sensor" noise, covariance matrix
        self.R = R # action uncertainty

    def getdata(self, coords_inp, coords_gt):
        # Get input data
        v_inp = coords_inp - self.x_pred[:2]
        self.x_inp = np.concatenate((coords_inp, v_inp))
        self.inp.append(self.x_inp)

        # Get ground truth data
        v_gt = coords_gt - self.x_gt[:2]
        self.x_gt = np.concatenate((coords_gt, v_gt))
        self.gt.append(self.x_gt)

    def iteration(self):
        # Prediction
        x_hat = self.F @ self.x_pred
        P_hat = self.F @ self.P @ self.F.T + self.R

        # Correction
        K = P_hat @ self.H.T @ np.linalg.inv(self.H @ P_hat @ self.H.T + self.Q)
        self.x_pred = x_hat + K @ (self.x_inp - self.H @ x_hat)
        self.P = (np.eye(self.dim) - K @ self.H) @ P_hat
        self.pred.append(self.x_pred)

class MouseTracker(object):
    """Implementation of the mouse tracking system that interacts with user,
    synthesizes noisy measurements of the mouse position
    and aplies Kalman filter for smoothing.

    Args:
        canvsize -- (h, w) tuple, window sizes
        winname -- str, window title
        noise_model -- (mean_vector, variance_vector or covariance_matrix),
                       parameters of multivariate normal random variable
        fps -- int, frequency rate
        show_prediction -- bool, show Kalman prediction vector 
    """
    def __init__(self, canvsize=SIZE,
                 winname="Kalman Filter for Mouse Traking",
                 noise_model=(NOISE_MEAN_DEF, NOISE_COV_DEF),
                 fps=60, show_prediction=True):
        self.fps = fps
        self.winname = winname
        self.canvsize = canvsize
        self.show_prediction = show_prediction
        self.KF = KalmanFilter()
        self.coords_gt = np.zeros(2, dtype='int') 
        self.bg = np.zeros((*self.canvsize, 3), np.uint8) # black background
        self.canvas = self.bg.copy()
        if self.show_prediction:
            self.canvas_pred = self.bg.copy()
        self.frames = []
        # Noise model
        self.noise_rv = scipy.stats.multivariate_normal(*noise_model)

    def frame(self):
        # Synthesize measurement (add noise)
        coords_inp = self.coords_gt + self.noise_rv.rvs()

        # Run Kalman Filter
        self.KF.getdata(coords_inp, self.coords_gt)
        self.KF.iteration()

        # Draw data
        # Ground-truth
        x_prev = self.KF.gt[-1] if len(self.KF.gt) == 1 else self.KF.gt[-2]
        cv2.line(self.canvas, tuple(x_prev[:2]), tuple(self.KF.gt[-1][:2]), color=(47,43,37), thickness=1)

        # Input
        cv2.circle(self.canvas, tuple(self.KF.inp[-1].astype(int)[:2]), 2, (190,173,149), -1)
        
        # Output
        x_prev = self.KF.pred[-1] if len(self.KF.pred) == 1 else self.KF.pred[-2]
        cv2.line(self.canvas, tuple(x_prev.astype(int)[:2]), tuple(self.KF.pred[-1].astype(int)[:2]),
                 color=(111,130,243), thickness=2)
        
        # Prediction
        if self.show_prediction:
            self.canvas_pred = self.canvas.copy()
            x1 = self.KF.pred[-1].astype(int)[:2]
            x2 = x1 + self.KF.pred[-1].astype(int)[2:] * PRED_SCALE
            cv2.line(self.canvas_pred, tuple(x1), tuple(x2), color=(231,150,243), thickness=2)
        
        self.th = threading.Timer(1/self.fps, self.frame)
        self.th.start() 

    def run(self):
        print("Started tracking")
        cv2.namedWindow(self.winname)
        cv2.moveWindow(self.winname, 0, 0) 

        def _mouseMoved(event, x, y, flags, param):
            if event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_MBUTTONDOWN):
                self.coords_gt = np.array([x, y], dtype='int')

        cv2.setMouseCallback(self.winname, _mouseMoved, None)
        self.frame()
        while True:
            final = self.canvas if not self.show_prediction else self.canvas_pred
            cv2.imshow(self.winname, final)
            self.frames.append(final)
            if cv2.waitKey(1) & 0xFF == ord("c"):
                self.canvas = self.bg.copy()
                cv2.imshow(self.winname, self.canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        self.quit()

    def quit(self):
        self.th.cancel()
        cv2.destroyAllWindows()
        print("Ended tracking")

def save_video(filepath, frames, framesize=None, fps=40):
    if framesize is None:
        framesize = frames[0].shape[1::-1]
    writer = cv2.VideoWriter(filepath, cv2.VideoWriter_fourcc(*"MJPG"), fps, framesize)
    for frame in frames:
        writer.write(frame)
    writer.release()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--save-path', type=str, default="res/mouse-tracking.avi", help='video file saving path')
    args = parser.parse_args()
    
    mouse_tracker = MouseTracker()
    mouse_tracker.run()
    save_video(args.save_path, mouse_tracker.frames)

    cv2.destroyAllWindows()