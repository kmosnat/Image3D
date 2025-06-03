# camera_parameters_estimation.py
# Shi Zhang
# This file is for estimating camera parameters

# import statements
import cv2
import numpy as np
import open3d as o3d
import csv
import os
import glob


def estimate_intrinsic_parameters(matches, dimensions):
    # Estimates intrinsic camera parameters based on feature matches and image dimensions.
    # Utilizes the OpenCV calibration functions to compute the camera matrix and distortion coefficients.
    image_names = []
    for (image1, image2), match_points in matches.items():
        if image1 not in image_names:
            image_names.append(image1)
        if image2 not in image_names:
            image_names.append(image2)

    image_points = {}
    for (image1, image2), match_points in matches.items():
        if image1 not in image_points:
            image_points[image1] = []
        if image2 not in image_points:
            image_points[image2] = []

        for x1, y1, x2, y2 in match_points:
            image_points[image1].append((x1, y1))
            image_points[image2].append((x2, y2))

    image_points_list = [np.array(
        image_points[img], dtype=np.float32).reshape(-1, 1, 2) for img in image_names]

    # Initialize object points for each image
    object_points_list = []
    for img in image_names:
        num_points = len(image_points[img])
        object_points_3d = np.zeros((num_points, 3), dtype=np.float32)
        for i in range(num_points):
            # Assuming a 10x10 grid, modify as needed
            object_points_3d[i] = [i % 10, i // 10, 0]

        object_points_list.append(object_points_3d)

    camera_matrices = {}
    dist_coeffs = {}

    for idx, img_name in enumerate(image_names):
        img_points = image_points_list[idx]
        obj_points = object_points_list[idx]

        w, h = dimensions[img_name]
        camera_matrix = np.array(
            [[w, 0, w/2], [0, h, h/2], [0, 0, 1]], dtype=np.float32)
        dist_coeff = np.zeros((4, 1), dtype=np.float32)

        print(f"Processing image: {img_name}")
        print(f"Number of object points: {len(obj_points)}")
        print(f"Number of image points: {len(img_points)}")
        print(f"Shape of image points for current image: {img_points.shape}")

        _, camera_matrix, dist_coeff, _, _ = cv2.calibrateCamera(
            [obj_points], [img_points], (w, h), camera_matrix, dist_coeff,
            flags=cv2.CALIB_FIX_PRINCIPAL_POINT + cv2.CALIB_FIX_ASPECT_RATIO
        )

        camera_matrices[img_name] = camera_matrix
        dist_coeffs[img_name] = dist_coeff

    return camera_matrices, dist_coeffs

def calibrate_camera_with_chessboard(image_folder, pattern_size=(11, 8), square_size=1.5):
    objp = np.zeros((pattern_size[1] * pattern_size[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size  # Mettre à l'échelle selon la taille réelle des carrés

    objpoints = []
    imgpoints = []

    images = glob.glob(f'{image_folder}/*.jpg') + glob.glob(f'{image_folder}/*.png')
    if len(images) == 0:
        raise ValueError(f"Aucune image .jpg ou .png trouvée dans {image_folder}.")

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"[Warning] Impossible de lire l'image {fname}. Skipped.")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if not ret:
            print(f"[Info] Aucun damier trouvé dans {fname}.")
            continue

        corners_subpix = cv2.cornerSubPix(
            gray, corners, winSize=(11, 11), zeroZone=(-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )

        objpoints.append(objp.copy())
        imgpoints.append(corners_subpix)

    if len(objpoints) == 0:
        raise ValueError("Aucun damier détecté dans toutes les images fournies.")

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    if not ret:
        print("Échec de la calibration.")
    else:
        print("Calibration réussie.")
        print("Matrice de la caméra (K) :\n", camera_matrix)
        print("Coefficients de distorsion :\n", dist_coeffs)

    return camera_matrix, dist_coeffs

def load_camera_parameters(filename):
    camera_params = {}
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        for row in reader:
            if len(row) != 7:
                print(f"Skipping invalid row: {row}")
                continue
            image, rvec1, rvec2, rvec3, tvec1, tvec2, tvec3 = row
            rvec = np.array([float(rvec1), float(rvec2), float(rvec3)])
            tvec = np.array([float(tvec1), float(tvec2), float(tvec3)])

            # Convert rotation vector to rotation matrix
            r_mat, _ = cv2.Rodrigues(rvec)

            camera_params[image] = {'r_mat': r_mat, 'tvec': tvec}
    return camera_params


def load_feature_matches(csv_file):
    matches = {}
    dimensions = {}
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        for row in reader:
            # Unpack all values from each row
            image1, image2, x1, y1, x2, y2, w1, h1, w2, h2 = row
            if (image1, image2) not in matches:
                matches[(image1, image2)] = []
                dimensions[image1] = (int(w1), int(h1))
                dimensions[image2] = (int(w2), int(h2))
            matches[(image1, image2)].append(
                (float(x1), float(y1), float(x2), float(y2)))
    return matches, dimensions


def estimate_camera_parameters(matches, dimensions):
    camera_params = {}

    for (image1, image2), match_points in matches.items():
        w1, h1 = dimensions[image1]
        w2, h2 = dimensions[image2]
        avg_width  = (w1 + w2) / 2.0
        avg_height = (h1 + h2) / 2.0

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            int(avg_width), int(avg_height),
            avg_width, avg_height,
            avg_width / 2.0, avg_height / 2.0
        )
        K = intrinsic.intrinsic_matrix.astype(np.float64)

        D = np.zeros((5, 1), dtype=np.float64)

        pts1 = np.array([[x1, y1] for x1, y1, x2, y2 in match_points], dtype=np.float32)
        pts2 = np.array([[x2, y2] for x1, y1, x2, y2 in match_points], dtype=np.float32)

        if pts1.shape[0] < 5:
            print(f"[Warning] Moins de 5 correspondances pour {image1}–{image2}, on skip cette paire.")
            continue

        pts1_undist = cv2.undistortPoints(pts1, K, D)  # shape (N,1,2)
        pts2_undist = cv2.undistortPoints(pts2, K, D)  # shape (N,1,2)

        pts1_norm = pts1_undist.reshape(-1, 2)
        pts2_norm = pts2_undist.reshape(-1, 2)

        E, maskE = cv2.findEssentialMat(
            pts1_norm, pts2_norm,
            focal=1.0, pp=(0.0, 0.0),
            method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        if E is None or E.size == 0:
            print(f"[Error] Impossible de calculer E pour {image1}–{image2}.")
            continue

        inlier_ratio = float(np.sum(maskE)) / maskE.shape[0]
        print(f"  {image1}–{image2} : EssentialMatrix trouvée, inliers = {np.sum(maskE)}/{maskE.shape[0]} ({inlier_ratio*100:.1f}%)")

        _, R_rel, t_rel, mask_pose = cv2.recoverPose(
            E,
            pts1_norm, pts2_norm,
            focal=1.0, pp=(0.0, 0.0)
        )
        inlier_pose = int(np.sum(mask_pose))
        print(f"  {image1}–{image2} : recoverPose utilisé {inlier_pose} points sur {pts1_norm.shape[0]}")

        R1 = np.eye(3, dtype=np.float64)
        t1 = np.zeros((3, 1), dtype=np.float64)

        R2 = R_rel.astype(np.float64)
        t2 = t_rel.astype(np.float64)

        camera_params[image1] = {'r_mat': R1, 'tvec': t1, 'K': K.copy(), 'D': D.copy()}
        camera_params[image2] = {'r_mat': R2, 'tvec': t2, 'K': K.copy(), 'D': D.copy()}

    return camera_params

def save_camera_params_to_file(camera_params, filename):
    # Saves the estimated camera parameters to a CSV file.
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image', 'rvec1', 'rvec2',
                        'rvec3', 'tvec1', 'tvec2', 'tvec3'])
        for image, params in camera_params.items():
            r_mat, tvec = params['r_mat'], params['tvec']
            rvec, _ = cv2.Rodrigues(r_mat)
            file.write(
                f"{image},{rvec[0][0]},{rvec[1][0]},{rvec[2][0]},{tvec[0][0]},{tvec[1][0]},{tvec[2][0]}\n")


def save_intrinsic_params_to_file(intrinsic_matrices, dist_coeffs, filename):
    # Saves the estimated intrinsic parameters and distortion coefficients to a CSV file.
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image', 'Camera Matrix', 'Distortion Coefficients'])
        for image, intrinsic in intrinsic_matrices.items():
            dist = dist_coeffs[image]
            writer.writerow(
                [image, intrinsic.flatten().tolist(), dist.flatten().tolist()])

