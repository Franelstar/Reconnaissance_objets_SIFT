from scipy import spatial
from tqdm import tqdm
import cv2 as cv
import numpy as np


# Calculation of local descriptors for each image
def description(t_images, t_image_gray):
    sift = cv.xfeatures2d.SIFT_create()
    t_desc = [0] * t_images.shape[0]
    t_kp = [0] * t_images.shape[0]
    t_list_kp = [0] * t_images.shape[0]

    for i, img in enumerate(t_image_gray):
        kp, des = sift.detectAndCompute(img, None)
        t_desc[i] = des
        t_kp[i] = cv.drawKeypoints(img, kp, t_images[i], flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        t_list_kp[i] = kp

    return np.array(t_desc), np.array(t_kp), t_list_kp


# Collect 2 vector arrays
# Returns the vectors of the 2nd array corresponding to the 1st array, and
# Returns the score = number of matches / number of descriptor of the 2nd table
def matching(vectors_ent, vectors_test, threshold):
    key_ent_to_test = []

    if vectors_test is not None:
        result = [None] * vectors_test.shape[0]
        correspondence = 0

        tree = spatial.KDTree(vectors_ent)

        # distance_ecl is ordered by the shortest distance
        distance_ecl, index = tree.query(vectors_test, p=2, k=2)

        for k, (dist, ind) in enumerate(zip(distance_ecl, index)):

            # We calculate the ratio between the shortest distance and the second shortest distance
            ratio = dist[0] / dist[1]
            if ratio < threshold:
                # Cross pairing
                tree_2 = spatial.KDTree(vectors_test)

                distance_ecl_2, index_2 = tree_2.query(vectors_ent[ind[0]], p=2, k=2)

                # If the ration is below the threshold
                # So there is correspondence on both sides
                ratio_2 = distance_ecl_2[0] / distance_ecl_2[1]
                if ratio_2 < threshold:
                    result[k] = vectors_ent[ind[0]]
                    key_ent_to_test.append(cv.DMatch(ind[0], index_2[0], distance_ecl_2[0]))
                    correspondence += 1
                else:
                    result[k] = None
            else:
                result[k] = None

        return correspondence / vectors_ent.shape[0], result, key_ent_to_test
    else:
        return 0, [], key_ent_to_test


# Calculation of the correspondence score between the descriptors of an image and the descriptors
# of training images
# Returns an array (of the same size desc_img_ent) corresponding to the different scores
def score_match(desc_img_ent, desc_img_test, threshold):
    result = []
    d_match = []

    with tqdm(total=len(desc_img_ent), desc="Chargement", bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
        for i, img in enumerate(desc_img_ent):
            dm = []
            if isinstance(img, type(None)):
                score = 0
            else:
                score, _, dm = matching(img, desc_img_test, threshold)
            result.append(score)
            d_match.append(dm)
            pbar.update(1)

    return result, d_match


# Finding indexes of the K highest score value
def get_k_max_index(tab, d_match, k, labels_train):
    scores = [0] * k
    labels = [0] * k
    dm = [0] * k
    index_s = [0] * k

    for i in range(k):
        m = max(tab)
        index = tab.index(m)
        scores[i] = round(m, 2)
        labels[i] = labels_train[index]
        dm[i] = d_match[index]
        index_s[i] = index
        tab[index] = -1

    return np.array(list(zip(labels, scores, dm, index_s)))


# Prediction from the descriptors of an image
def predict_desc(desc, descriptors_train, label_train, threshold, k):
    sc, d_match = score_match(descriptors_train, desc, threshold)
    p = get_k_max_index(sc, d_match, k, label_train)
    (objects, qte) = np.unique([i[0] for i in p], return_counts=True)
    arg = list(zip(objects, qte))
    arg = sorted(arg, key=lambda item: item[1], reverse=True)

    # We search for the predicted object with the highest score
    obj = arg[0][0]
    p = sorted(p, key=lambda item: item[1], reverse=True)
    # We return the first found
    for find in p:
        if obj == find[0]:
            return find
