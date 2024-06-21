import numpy as np
from skimage.filters import gaussian
from skimage.color import rgb2lab, lab2rgb
from sklearn import cluster
from typing import Dict, List, Tuple


class SkinColorPredictor:
    def __init__(self):
        pass

    def get_hue(
        self, a_values: np.ndarray, b_values: np.ndarray, eps: float = 1e-8
    ) -> np.ndarray:
        """Compute hue angle"""
        return np.degrees(np.arctan(b_values / (a_values + eps)))

    def mode_hist(self, x: np.ndarray, bins: str = "sturges") -> float:
        """Compute a histogram and return the mode"""
        hist, bins = np.histogram(x, bins=bins)
        mode = bins[hist.argmax()]
        return mode

    def clustering(
        self, x: np.ndarray, n_clusters: int = 5, random_state: int = 2021
    ) -> Tuple[np.ndarray, cluster.KMeans]:
        model = cluster.KMeans(n_clusters, random_state=random_state)
        model.fit(x)
        return model.labels_, model

    def get_scalar_values(
        self,
        skin_smoothed_lab: np.ndarray,
        labels: np.ndarray,
        topk: int = 3,
        bins: str = "sturges",
    ) -> Dict[str, np.ndarray]:
        # gather values of interest
        hue_angle = self.get_hue(skin_smoothed_lab[:, 1], skin_smoothed_lab[:, 2])
        skin_smoothed = lab2rgb(skin_smoothed_lab)

        # concatenate data to be clustered (L, h, and RGB for visualization)
        data_to_cluster = np.vstack(
            [
                skin_smoothed_lab[:, 0],
                hue_angle,
                skin_smoothed[:, 0],
                skin_smoothed[:, 1],
                skin_smoothed[:, 2],
            ]
        ).T

        # Extract skin pixels for each mask (by clusters)
        n_clusters = len(np.unique(labels))
        masked_skin = [data_to_cluster[labels == i, :] for i in range(n_clusters)]
        n_pixels = np.asarray([np.sum(labels == i) for i in range(n_clusters)])

        # get scalar values per cluster
        keys = ["lum", "hue", "red", "green", "blue"]
        res = {}
        for i, key in enumerate(keys):
            res[key] = np.array(
                [self.mode_hist(part[:, i], bins=bins) for part in masked_skin]
            )

        # only keep top3 in luminance and average results
        idx = np.argsort(res["lum"])[::-1][:topk]
        total = np.sum(n_pixels[idx])
        res_topk = {}
        for key in keys:
            res_topk[key] = np.average(res[key][idx], weights=n_pixels[idx])
            res_topk[key + "_std"] = np.sqrt(
                np.average((res[key][idx] - res_topk[key]) ** 2, weights=n_pixels[idx])
            )

        return res_topk

    def get_skin_values(
        self, img: np.ndarray, mask: np.ndarray, n_clusters: int = 5
    ) -> Dict[str, float]:
        # smoothing
        img_smoothed = gaussian(img, sigma=(1, 1), truncate=4, channel_axis=2)

        # get skin pixels (shape will be Mx3) and go to Lab
        skin_smoothed = img_smoothed[mask == 1]
        skin_smoothed_lab = rgb2lab(skin_smoothed)

        res = {}

        # L and hue
        hue_angle = self.get_hue(skin_smoothed_lab[:, 1], skin_smoothed_lab[:, 2])
        data_to_cluster = np.vstack([skin_smoothed_lab[:, 0], hue_angle]).T
        labels, model = self.clustering(data_to_cluster, n_clusters=n_clusters)
        tmp = self.get_scalar_values(skin_smoothed_lab, labels)
        res["lum"] = tmp["lum"]
        res["hue"] = tmp["hue"]
        res["lum_std"] = tmp["lum_std"]
        res["hue_std"] = tmp["hue_std"]

        # also extract RGB for visualization purposes
        res["red"] = tmp["red"]
        res["green"] = tmp["green"]
        res["blue"] = tmp["blue"]
        res["red_std"] = tmp["red_std"]
        res["green_std"] = tmp["green_std"]
        res["blue_std"] = tmp["blue_std"]

        return res

    def predict(
        self, image: np.ndarray, mask: np.ndarray
    ) -> Dict[str, Union[np.ndarray, float]]:
        # Convert image and mask to numpy arrays
        img_original = np.asarray(image)
        mask = np.asarray(mask)

        # Ensure mask is binary [0, 1]
        mask = (mask == 1).astype(int)

        # Get skin values
        skin_values = self.get_skin_values(img_original, mask)

        # Extract a_values and b_values
        skin_smoothed = img_original[mask == 1]
        skin_smoothed_lab = rgb2lab(skin_smoothed)
        a_values = skin_smoothed_lab[:, 1]
        b_values = skin_smoothed_lab[:, 2]

        # Prepare the result dictionary
        attrs = ["lum", "hue"]
        res = {attr: skin_values[attr] for attr in attrs}
        res.update({attr + "_std": skin_values[attr + "_std"] for attr in attrs})
        res["a_values"] = a_values
        res["b_values"] = b_values

        return res
