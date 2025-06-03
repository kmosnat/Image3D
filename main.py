import argparse
import csv
import numpy as np
import open3d as o3d

from preprocess import *
from feature_matching import *
from camera_parameters_estimation import *
from reconstruction_visualization import *


def process_feature_matching(images, save_path):
    all_matches = []
    #for i in range(1, len(images)):
    for j in range(1, len(images)):
        i=j-1
        img1_name, img1 = images[i]
        img2_name, img2 = images[j]
        pair_name = f"{img1_name}_vs_{img2_name}"
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        matches, kp1, kp2 = detect_and_match_features(img1, img2, sift, pair_name, save_path)
        for m in matches:
            pt1, pt2 = kp1[m.queryIdx].pt, kp2[m.trainIdx].pt
            all_matches.append([img1_name, img2_name, pt1[0], pt1[1], pt2[0], pt2[1], w1, h1, w2, h2])
    return all_matches


def save_matches_to_csv(matches, filename):
    headers = [
        'Image1', 'Image2',
        'KeyPoint1_X', 'KeyPoint1_Y',
        'KeyPoint2_X', 'KeyPoint2_Y',
        'Width1', 'Height1',
        'Width2', 'Height2'
    ]
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(matches)


def main(batch, visualize=False):
    images_path = f'images/{batch}/'
    preprocessed_path = f'preprocessed_images/{batch}/'
    matching_output_path = f'output/feature_matching/{batch}/'

    # 1) Prétraitement
    process_images_in_folder(images_path, preprocessed_path)
    images = load_images_from_folder(preprocessed_path)  # renvoie liste de (nom, img)

    # 2) Appariement de caractéristiques
    matches_list = process_feature_matching(images, matching_output_path)
    print(f"Found {len(matches_list)} matches across {len(images)} images.")
    save_matches_to_csv(matches_list, 'feature_matches.csv')

    # 3) Chargement des correspondances et des dimensions
    #    load_feature_matches doit retourner :
    #      matches_dict : { (img1, img2): [ (x1, y1, x2, y2), ... ], ... }
    #      dimensions   : { img_name: (width, height), ... }
    matches_dict, dimensions = load_feature_matches('feature_matches.csv')

    # 4) Estimation des paramètres de chaque caméra (poses, intrinsics, distorsion)
    cam_params = estimate_camera_parameters(matches_dict, dimensions)
    # cam_params[img_name] = { 'r_mat': R, 'tvec': t, 'K': K, 'D': D }

    # 5) Calibration (ou chargement) de la même matrice K et des mêmes D pour toutes les vues
    K, D = calibrate_camera_with_chessboard('images/calib/')
    # On supposera que chaque caméra a la même K et D :
    intrinsics = (K, K)
    dist_coeffs = (D, D)

    # 6) Enregistrer les paramètres caméra dans un CSV (fonction à adapter le cas échéant)
    save_camera_params_to_file(cam_params, 'camera_parameters.csv')
    save_intrinsic_params_to_file(intrinsics, dist_coeffs, 'intrinsic_parameters.csv')

    # Reconstruction 3D
    points_3d = triangulate_points(matches, cam_params, intrinsics, dist_coeffs).astype(np.float64)
    point_cloud = create_point_cloud(points_3d)
    save_point_cloud(point_cloud, 'point_cloud.ply')

    if visualize:
        visualize_point_cloud(point_cloud)
        mesh = reconstruct_mesh(point_cloud)
        export_mesh(mesh, "reconstructed_mesh.obj")
        o3d.visualization.draw_geometries([mesh],
                                          window_name='3D Viewer',
                                          width=1920,
                                          height=1080)

    # Si l’on veut quand même renvoyer le nuage
    return all_points_3d


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Pipeline de reconstruction 3D à partir d’images."
    )
    parser.add_argument('-batch', required=True,
                        help='Nom du dossier contenant les images à traiter')
    parser.add_argument('-visualize', action='store_true',
                        help='Visualiser les résultats de la reconstruction 3D')
    args = parser.parse_args()
    main(args.batch, args.visualize)