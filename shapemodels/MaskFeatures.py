import cv2
import numpy as np

from shapemodels.Config import config


class MaskFeatures:
    """
    Class to generate and manage offsets as features in the hand RDF problem.
    """

    def __init__(self, ephsilon, mean, cov, distribution='gaussian_length', factor=None):
        np.random.seed(config.OFFSETS_SEED)
        self.n_features = config.N_FEATURES
        self.ephsilon = ephsilon
        factor = config.features_factor if factor is None else factor

        if distribution == 'gaussian':
            angles_u = np.random.uniform(0, 2 * np.pi, config.N_FEATURES)
            angles_v = np.random.uniform(0, 2 * np.pi, config.N_FEATURES)

            self._offsets_u = np.random.multivariate_normal(mean, cov, config.N_FEATURES)
            self._offsets_v = np.random.multivariate_normal(mean, cov, config.N_FEATURES)

            self._offsets_u[:, 1] = self._offsets_u[:, 1] * np.cos(angles_u) + self._offsets_u[:, 0] * np.sin(angles_u)
            self._offsets_u[:, 0] = self._offsets_u[:, 0] * np.cos(angles_u) - self._offsets_u[:, 1] * np.sin(angles_u)

            self._offsets_v[:, 1] = self._offsets_v[:, 1] * np.cos(angles_v) + self._offsets_v[:, 0] * np.sin(angles_v)
            self._offsets_v[:, 0] = self._offsets_v[:, 0] * np.cos(angles_v) - self._offsets_v[:, 1] * np.sin(angles_v)

        elif distribution == 'gaussian_length':
            self._offsets_u = np.random.uniform(-1, 1, (self.n_features, 2))
            self._offsets_v = np.random.uniform(-1, 1, (self.n_features, 2))

            self._offsets_u *= np.random.multivariate_normal(mean, cov, config.N_FEATURES)
            self._offsets_v *= np.random.multivariate_normal(mean, cov, config.N_FEATURES)

        else:
            self._offsets_u, self._offsets_v = self.generate_offsets()
            self._offsets_u[:, 0] *= self.ephsilon[0]
            self._offsets_u[:, 1] *= self.ephsilon[1]

            self._offsets_v[:, 0] *= self.ephsilon[0]
            self._offsets_v[:, 1] *= self.ephsilon[1]

        self._offsets_u *= factor
        self._offsets_v *= factor

        self._offsets_u = self._offsets_u[:, :, np.newaxis].astype(np.int32)
        self._offsets_v = self._offsets_v[:, :, np.newaxis].astype(np.int32)

        self.max_offset = int(np.max([np.max(np.abs(self._offsets_u)), np.max(np.abs(self._offsets_v))])) + 1

    def get_image_features(self, image, positions):
        """
        Given a depth image and a set of (x, y) positions,
        get the features of the pixels in the image specified by the positions.

        :param image: uint16 ndarray The depth image to use
        :param positions: List|ndarray of the (x, y) positions to compute the features.

        :return: Tuple list, ndarray
        A list with the positions of the features computed.
        An ndarray of shape (len(positions), len(offsets)) with the features computed in each position.

        """
        im = cv2.copyMakeBorder(image, self.max_offset, self.max_offset, self.max_offset, self.max_offset,
                                cv2.BORDER_CONSTANT, 0)
        pos = positions + self.max_offset

        offsets_u = (pos + self._offsets_u).T
        offsets_v = (pos + self._offsets_v).T

        offsets_u_y, offsets_u_x = offsets_u[:, 0, :], offsets_u[:, 1, :]
        offsets_v_y, offsets_v_x = offsets_v[:, 0, :], offsets_v[:, 1, :]

        features = im[(offsets_u_y, offsets_u_x)] - im[(offsets_v_y, offsets_v_x)]

        return positions, features.astype(np.float64)

    @staticmethod
    def generate_offsets() -> tuple:
        """
        Generate the feature offsets following the specified distribution in the parameter

        :return: the offsets generated
        """

        offsets_u = config.features_factor * np.random.uniform(-config.OFFSET_MAX, config.OFFSET_MAX,
                                                               (config.N_FEATURES, 2)).astype(np.float64)
        offsets_v = config.features_factor * np.random.uniform(-config.OFFSET_MAX, config.OFFSET_MAX,
                                                               (config.N_FEATURES, 2)).astype(np.float64)

        # a half of candidates either u or v putted to 0
        offsets_u[list(np.arange(0, len(offsets_u), 4))] = [0., 0.]
        offsets_v[list(np.arange(1, len(offsets_v), 4))] = [0., 0.]

        return offsets_u, offsets_v
