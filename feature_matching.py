# feature_matching.py
# Shi Zhang
# This file is for feature detection and matching

# import statements
import cv2
import os
import csv
import matplotlib.pyplot as plt
import numpy as np

# Initialize SIFT with custom parameters
nfeatures = 5000  # Increase for more features
contrastThreshold = 0.1  # Decrease to retain more features with lower contrast
edgeThreshold = 100  # Decrease to retain more features that are edge-like
sigma = 1.6  # Typically left at default

sift = cv2.SIFT_create(nfeatures=nfeatures, nOctaveLayers = 6, contrastThreshold=contrastThreshold,
                       edgeThreshold=edgeThreshold, sigma=sigma)

def draw_title(img, title, font_scale=1, font=cv2.FONT_HERSHEY_SIMPLEX, y_offset=30):
    # Adds a title to an image at a specified position.
    # The title is centrally aligned with a specified font scale and color.
    text_size = cv2.getTextSize(title, font, font_scale, 1)[0]
    text_x = (img.shape[1] - text_size[0]) // 2
    cv2.putText(img, title, (text_x, y_offset),
                font, font_scale, (0, 255, 0), 2)


def detect_and_match_features(image1, image2, sift, pair_name, save_path):
    # Detects and matches features between two images using SIFT and FLANN-based matcher.
    # It filters good matches based on Lowe's ratio test and draws top matches for visualization.
    # The resulting image with matches is saved to the specified path.
    # Returns the good matches and keypoints for both images.

    # Detect and compute keypoints and descriptors with SIFT
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    # Check if SIFT found descriptors
    if descriptors1 is None or descriptors2 is None:
        print(f"Not enough features in one of the images. Skipping this pair.")
        return [], [], []

    print(
        f"Found {len(keypoints1)} and {len(keypoints2)} keypoints in the images respectively.")

    # Convert descriptors to float32 for FLANN
    if descriptors1.dtype != np.float32:
        descriptors1 = descriptors1.astype(np.float32)
    if descriptors2.dtype != np.float32:
        descriptors2 = descriptors2.astype(np.float32)

    # FLANN-based matcher parameters
    FLANN_INDEX_KDTREE = 1  # KD-Tree algorithm
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=9)
    search_params = dict(checks=150)

    # Matching descriptor vectors using FLANN matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    print(f"Found {len(matches)} raw matches.")

    # Print out distances of raw matches
    for i, (m, n) in enumerate(matches):
        print(
            f"Match {i}: Distance 1 - {m.distance}, Distance 2 - {n.distance}")
        if i == 10:  # Just print the first 10 to check the range
            break

    # Store all good matches as per Lowe's ratio test.
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    print(f"{len(good_matches)} matches passed Lowe's ratio test.")

    # Apply RANSAC to filter out outliers
    if len(good_matches) > 4:  # Minimum number of matches to find the homography
        ptsA = np.float32(
            [keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        ptsB = np.float32(
            [keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        matrix, mask = cv2.findHomography(
            ptsA, ptsB, cv2.RANSAC, 10.0)
        if mask is not None:
            matchesMask = mask.ravel().tolist()
            good_matches = [gm for gm, mask in zip(
                good_matches, matchesMask) if mask]
            print(f"{sum(matchesMask)} matches survived the RANSAC filter.")
        else:
            print("RANSAC homography could not be computed. All matches discarded.")
            matchesMask = None
    else:
        print("Not enough good matches to apply RANSAC.")
        matchesMask = None

    # Draw top matches
    img_matches = cv2.drawMatches(image1, keypoints1, image2, keypoints2,
                                  good_matches[:1000], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Adding title to the image
    draw_title(img_matches, 'Feature Matches')

    # Resize image if necessary (similar to adjusting figure size)
    desired_width = 1200  # Example width, adjust as needed
    scale_ratio = desired_width / img_matches.shape[1]
    img_matches = cv2.resize(img_matches, None, fx=scale_ratio, fy=scale_ratio)

    # Save the image
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    output_file = os.path.join(save_path, f"matches_{pair_name}.png")
    cv2.imwrite(output_file, img_matches)

    print(f"Saved matching visualization to {output_file}")

    return good_matches, keypoints1, keypoints2


def load_images_from_folder(folder):
    # Loads all grayscale images from a specified folder.
    # Returns a list of tuples containing the filename and the image.
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append((filename, img))
    return images


def launch_selector(images, matching_output_path, num_points=6):
    """
    Permet à l'utilisateur de sélectionner manuellement des points sur chaque image.
    Prend en entrée une liste de tuples (filename, image).
    Ouvre chaque image, attend num_points clics, puis passe à la suivante.
    Retourne une liste de matches au format :
    [img1_name, img2_name, x1, y1, x2, y2, w1, h1, w2, h2]
    pour chaque paire consécutive d'images.
    Ajoute la visualisation des correspondances (drawMatches, draw_title, sauvegarde) pour chaque paire.
    """
    selected_points = {}
    nb_max_img  = 5
    for filename, img in images:
        if img is None:
            print(f"Impossible de lire {filename}")
            continue
        points = []
        clone = img.copy()
        window_name = f"Sélectionnez {num_points} points sur {os.path.basename(filename)}"
        cv2.namedWindow(window_name)
        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                cv2.circle(clone, (x, y), 5, (255, 255, 255), -1)
                cv2.imshow(window_name, clone)
        cv2.setMouseCallback(window_name, click_event)
        print(f"[INFO] Cliquez {num_points} fois sur {filename}")
        while len(points) < num_points:
            cv2.imshow(window_name, clone)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC pour quitter
                break
        cv2.destroyWindow(window_name)
        selected_points[filename] = np.array(points)
        if(len(selected_points) >= nb_max_img):
            print(f"Nombre maximum d'images ({nb_max_img}) atteint, arrêt de la sélection.")
            break   
    # Construction du format de matches pour chaque paire consécutive
    matches = []
    keys = list(selected_points.keys())
    for i in range(1, len(keys)):
        img1_name, img2_name = keys[i-1], keys[i]
        pts1, pts2 = selected_points[img1_name], selected_points[img2_name]
        img1 = [img for name, img in images if name == img1_name][0]
        img2 = [img for name, img in images if name == img2_name][0]
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        n = min(len(pts1), len(pts2))
        # Visualisation des correspondances manuelles
        keypoints1 = [cv2.KeyPoint(float(x), float(y), 1) for x, y in pts1[:n]]
        keypoints2 = [cv2.KeyPoint(float(x), float(y), 1) for x, y in pts2[:n]]
        good_matches = [cv2.DMatch(_queryIdx=k, _trainIdx=k, _imgIdx=0, _distance=0) for k in range(n)]
        img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2,
                                      good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # Ajout du titre
        draw_title(img_matches, 'Manual Feature Matches')
        # Redimensionnement
        desired_width = 1200
        scale_ratio = desired_width / img_matches.shape[1]
        img_matches = cv2.resize(img_matches, None, fx=scale_ratio, fy=scale_ratio)
        # Sauvegarde
        pair_name = f"{os.path.splitext(os.path.basename(img1_name))[0]}_{os.path.splitext(os.path.basename(img2_name))[0]}"
        save_path = os.path.join(matching_output_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        output_file = os.path.join(save_path, f"matches_{pair_name}.png")
        cv2.imwrite(output_file, img_matches)
        print(f"Saved manual matching visualization to {output_file}")
        for k in range(n):
            x1, y1 = pts1[k]
            x2, y2 = pts2[k]
            matches.append([img1_name, img2_name, x1, y1, x2, y2, w1, h1, w2, h2])
    return matches



