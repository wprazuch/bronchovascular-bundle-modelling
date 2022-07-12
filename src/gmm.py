# poetry run python -m covrads.segmentation.bronchi.threshold_bronchi --input_nrrd_path INPUT_PATH --number_of_gmms NO --output_path OUT_PATH
# input_nrrd_path, number_of_gmms, output_path
import os

import pydicom
import nrrd

import numpy as np
from scipy import linalg
import scipy.stats as stats
import cv2
import pandas as pd

import matplotlib.pyplot as plt

from sklearn import mixture
from skimage.measure import label

import fire

opj = os.path.join
ld = os.listdir

# Cell
def plot_fitted_gmm(data, gmm, output_path):

    weights = gmm.weights_
    means = gmm.means_
    covars = gmm.covariances_

    weights = list(weights)
    means = list(means.squeeze())
    covars = list(covars.squeeze())
    stds = list(np.sqrt(covars))

    gmm_list = [
        {"mean": mean, "weight": weight, "std": std, "covar": covar}
        for mean, std, weight, covar in zip(means, stds, weights, covars)
    ]

    gmm_list = sorted(gmm_list, key=lambda x: x["mean"])

    plt.hist(
        data,
        bins=100,
        histtype="bar",
        density=True,
        ec="#dcdedc",
        color="#dcdedc",
        alpha=0.5,
    )

    colors = [
        "#ACD5FE",
        "#759EC5",
        "#FFDD61",
        "yellow",
        "black",
        "brown",
        "orange",
        "purple",
        "pink",
    ]

    # plt.figure(figsize=(12, 10))
    f_axis = data.copy().ravel()
    f_axis.sort()
    for i in range(len(weights)):
        plt.plot(
            f_axis,
            gmm_list[i]["weight"]
            * stats.norm.pdf(
                f_axis, gmm_list[i]["mean"], np.sqrt(gmm_list[i]["covar"])
            ).ravel(),
            c=colors[i],
        )

    plt.xlabel("Intensywność HU", fontsize=10)
    plt.ylabel("Zliczenia", fontsize=10)

    plt.rcParams["agg.path.chunksize"] = 10000

    plot_output_path = os.path.join(output_path, "gmm_fitted_plot.png")
    plt.savefig(plot_output_path)
    plt.close()


# Cell
def get_gmm_metadata(gmm):

    weights = gmm.weights_
    means = gmm.means_
    covars = gmm.covariances_

    weights = list(weights)
    means = list(means.squeeze())
    covars = list(covars.squeeze())
    stds = list(np.sqrt(covars))

    gmm_list = [
        {"mean": mean, "weight": weight, "std": std}
        for mean, std, weight in zip(means, stds, weights)
    ]

    gmm_list = sorted(gmm_list, key=lambda x: x["mean"])

    return gmm_list


# Cell
def solve(m1, m2, std1, std2):
    a = 1 / (2 * std1 ** 2) - 1 / (2 * std2 ** 2)
    b = m2 / (std2 ** 2) - m1 / (std1 ** 2)
    c = (
        m1 ** 2 / (2 * std1 ** 2)
        - m2 ** 2 / (2 * std2 ** 2)
        - np.log(std2 / std1)
    )
    return np.roots([a, b, c])


# Cell
def get_thresholds(gmm_list, max_value):

    thresholds = []

    for i in range(len(gmm_list) - 1):
        current_gauss_dict = gmm_list[i]
        next_gauss_dict = gmm_list[i + 1]

        threshold_candidates = solve(
            current_gauss_dict["mean"],
            next_gauss_dict["mean"],
            current_gauss_dict["std"],
            next_gauss_dict["std"],
        )

        if max(threshold_candidates) < max_value:
            threshold = max(threshold_candidates)
        else:
            threshold = min(threshold_candidates)

        thresholds.append(threshold)

    return thresholds


# Cell
def create_thresholded_volumes(
    thresholds, nrrd_volume_seg, nrrd_header, output_path
):

    for i in range(len(thresholds) - 1):

        lower_threshold = thresholds[i]
        upper_threshold = thresholds[i + 1]

        thresholded = nrrd_volume_seg.copy()
        idx_inside_threshold = np.where(
            (lower_threshold <= thresholded) & (thresholded < upper_threshold)
        )
        idx_outside_threshold = np.where(
            (thresholded < lower_threshold) | (upper_threshold <= thresholded)
        )

        thresholded[idx_inside_threshold] = 1
        thresholded[idx_outside_threshold] = 0

        thresholded[nrrd_volume_seg == nrrd_volume_seg.min()] = 0

        full_output_path = os.path.join(
            output_path, f"gmm_threshold_{i+1}.nrrd"
        )

        nrrd.write(full_output_path, thresholded, nrrd_header)


# Cell
def run_thresholding(input_nrrd_path, number_of_gmms, output_path):

    if not os.path.exists(output_path):
        print("Creating new directory...")
        os.makedirs(output_path)

    print("Preparing data...")
    nrrd_volume_seg, nrrd_header = nrrd.read(input_nrrd_path)

    # nrrd_volume_seg = np.swapaxes(nrrd_volume_seg, 0, 2)

    nrrd_volume_seg_sq = nrrd_volume_seg.copy()

    X = nrrd_volume_seg_sq.copy()
    X = X.flatten()

    # Remove Background Intensities Outside Patient
    background_val = X.min()
    background_idx = np.where(X == background_val)
    X = np.delete(X, background_idx)

    background_idx = np.where(X > 500)
    X = np.delete(X, background_idx)
    X = X[:, np.newaxis]
    # print("Generating Distplot...")
    # Generate distplot
    # sns_plot = sns.distplot(X)
    # dist_plot_path = os.path.join(output_path, 'dist_plot.png')

    # Model gmm
    print("Running Gaussian modelling...")
    gmm = mixture.GaussianMixture(n_components=number_of_gmms)
    gmm.fit(X)

    print("Plotting fitted GMM...")
    plot_fitted_gmm(X, gmm, output_path)

    print("Sortings GMMs")
    gmm_list = get_gmm_metadata(gmm)
    thresholds = get_thresholds(gmm_list, X.max())
    thresholds.insert(0, nrrd_volume_seg.min() - 1)
    thresholds.append(nrrd_volume_seg.max() + 1)

    thresholds_df = pd.DataFrame(
        data=np.array(thresholds), columns=["threshold"]
    )
    thresholds_df.to_csv(
        os.path.join(output_path, "thresholds.csv"), index=False
    )

    print("Generating thresholded volumes...")
    create_thresholded_volumes(
        thresholds, nrrd_volume_seg, nrrd_header, output_path
    )


def segment_bronchi(input_nrrd_path, output_path):

    run_thresholding(
        input_nrrd_path, number_of_gmms=3, output_path=output_path
    )

    gmm_volume, hdr = nrrd.read(
        os.path.join(output_path, "gmm_threshold_3.nrrd")
    )

    # label regions by region growing
    labelled_gmm_volume = label(gmm_volume)
    uniq, counts = np.unique(labelled_gmm_volume, return_counts=True)

    # Pair ids with their counts in the volume
    labels = list(zip(uniq[1:], counts[1:]))
    labels = sorted(labels, key=lambda x: x[1])

    # Get two largest volumes - left bronchi and right bronchi
    try:
        id_bronchi_1, id_bronchi_2 = labels[-1][0], labels[-2][0]

    except:
        nrrd.write(
            os.path.join(output_path, BRONCHI_TREE_MASK_FILENAME),
            gmm_volume,
            hdr,
        )
        return 0

    # Generate mask of those bronchis
    bronchi = labelled_gmm_volume.copy()

    bronchi[(bronchi != id_bronchi_1) & (bronchi != id_bronchi_2)] = 0
    bronchi[(bronchi == id_bronchi_1) | (bronchi == id_bronchi_2)] = 1

    bronchi = bronchi.astype("int16")

    # Save mask
    nrrd.write(
        os.path.join(output_path, BRONCHI_TREE_MASK_FILENAME), bronchi, hdr
    )


if __name__ == "__main__":
    fire.Fire()
