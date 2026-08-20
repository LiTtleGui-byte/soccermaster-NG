# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg


"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in pitch coordinates.
    The 4-dimensional state space (x, y, vx, vy) contains the pitch position and velocity of the target.
    """

    def __init__(self):
        # Define constant velocity model
        self._motion_mat = np.eye(4, 4)
        self._motion_mat[0, 2] = 1.0  # x += vx
        self._motion_mat[1, 3] = 1.0  # y += vy

        # Define measurement matrix (projects from 4D state space to 2D measurement space)
        self._update_mat = np.zeros((2, 4))
        self._update_mat[0, 0] = 1.0  # x
        self._update_mat[1, 1] = 1.0  # y

        # Position and velocity std in meters and meters/second
        self._std_weight_position = 0.75  # 0.75 meters position std
        self._std_weight_velocity = 3.0   # 3.0 m/s velocity std

        self._std_pos = 1.0   # 1 meter std for position uncertainty growth
        self._std_vel = 0.5   # 0.5 m/s std for velocity uncertainty growth

    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Position coordinates (x, y) in pitch coordinate system (meters)

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (4 dimensional) and covariance matrix (4x4
            dimensional) of the new track.
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            self._std_weight_position,  # x position std (meters)
            self._std_weight_position,  # y position std (meters)
            self._std_weight_velocity,  # x velocity std (m/s)
            self._std_weight_velocity   # y velocity std (m/s)
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 4 dimensional mean vector of the object state at the previous time step.
        covariance : ndarray
            The 4x4 dimensional covariance matrix of the object state at the previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted state.
        """
        # Predict next state (project state forward in time)
        predicted_mean = np.dot(self._motion_mat, mean)

        # Add process noise (uncertainty growth)
        process_noise = np.diag([
            self._std_pos ** 2,     # position noise (meters^2)
            self._std_pos ** 2,     # position noise (meters^2)
            self._std_vel ** 2,     # velocity noise (m^2/s^2)
            self._std_vel ** 2      # velocity noise (m^2/s^2)
        ])

        predicted_covariance = (
            np.dot(np.dot(self._motion_mat, covariance), self._motion_mat.T) 
            + process_noise
        )

        return predicted_mean, predicted_covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (4 dimensional array).
        covariance : ndarray
            The state's covariance matrix (4x4 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.
        """
        std = [
            self._std_weight_position,
            self._std_weight_position]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (4 dimensional).
        covariance : ndarray
            The state's covariance matrix (4x4 dimensional).
        measurement : ndarray
            The 2 dimensional measurement vector (x, y), representing the
            position coordinates on the pitch.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, self._update_mat, covariance))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                       only_position=False):
        """Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 2 degrees of
        freedom.

        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (4 dimensional).
        covariance : ndarray
            Covariance of the state distribution (4x4 dimensional).
        measurements : ndarray
            An Nx2 dimensional matrix of N measurements, each in
            format (x, y) representing pitch coordinates.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the position
            only (which is already the case for pitch coordinates).

        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False,
            overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha
