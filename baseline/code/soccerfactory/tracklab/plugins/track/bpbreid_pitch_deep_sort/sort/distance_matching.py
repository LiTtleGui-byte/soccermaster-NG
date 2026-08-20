# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import linear_assignment


def euclidean_distance(position, candidates):
    """Compute Euclidean distance between a position and multiple candidates.

    Parameters
    ----------
    position : ndarray
        A position in format `(x, y)`.
    candidates : ndarray
        A matrix of candidate positions (one per row) in the same format
        as `position`.

    Returns
    -------
    ndarray
        The Euclidean distances between the `position` and each candidate.
        A lower score means the candidate is closer to the position.
    """
    return np.linalg.norm(candidates - position, axis=1)


def distance_cost(tracks, detections, track_indices=None,
                detection_indices=None, distance_matching_max_time_since_update=1):
    """A Euclidean distance metric.

    Parameters
    ----------
    tracks : List[deep_sort.track.Track]
        A list of tracks.
    detections : List[deep_sort.detection.Detection]
        A list of detections.
    track_indices : Optional[List[int]]
        A list of indices to tracks that should be matched. Defaults to
        all `tracks`.
    detection_indices : Optional[List[int]]
        A list of indices to detections that should be matched. Defaults
        to all `detections`.
    max_time_since_update : int
        Maximum allowed time since last update before a track is considered lost.
        Defaults to 1.

    Returns
    -------
    ndarray
        Returns a cost matrix of shape
        len(track_indices), len(detection_indices) where entry (i, j) is
        the normalized distance between tracks[track_indices[i]] and 
        detections[detection_indices[j]].
    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):
        if tracks[track_idx].time_since_update > distance_matching_max_time_since_update: # 这个参数限制了基于距离的匹配，只能发生在没有lose track的track上，这样好吗？比如一些短暂的消失，也许用这个还能救回来
            cost_matrix[row, :] = linear_assignment.INFTY_COST
            continue

        # Get track's predicted position (first two elements of the state vector)
        pos = tracks[track_idx].mean[:2]
        candidates = np.asarray([detections[i].xy for i in detection_indices])
        distances = euclidean_distance(pos, candidates)
        
        cost_matrix[row, :] = distances

    return cost_matrix