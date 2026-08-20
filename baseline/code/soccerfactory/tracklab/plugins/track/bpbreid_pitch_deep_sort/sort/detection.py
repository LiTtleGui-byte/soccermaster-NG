# vim: expandtab:ts=4:sw=4
import numpy as np


class Detection(object):
    """
    This class represents a detection of a player on the pitch.

    Parameters
    ----------
    xy : array_like
        Position in format `(x, y)` representing pitch coordinates.
    confidence : float
        Detector confidence score.
    feature : array_like
        A feature vector that describes the object contained in this image.

    Attributes
    ----------
    xy : ndarray
        Position in format `(x, y)` representing pitch coordinates.
    confidence : ndarray
        Detector confidence score.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.
    """

    def __init__(self, id, xy, confidence, label, feature, mask=None):
        self.id = id
        self.xy = np.asarray(xy, dtype=np.float32)
        self.confidence = float(confidence)
        self.cls = int(label)
        self.feature = np.asarray(feature, dtype=np.float32)
        self.mask = mask

    def get_xy(self):
        """Return position coordinates."""
        return self.xy.copy()
