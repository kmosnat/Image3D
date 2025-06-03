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
    headers = ['Image1', 'Image2', 'KeyPoint1_X', 'KeyPoint1_Y',
               'KeyPoint2_X', 'KeyPoint2_Y', 'Width1', 'Height1', 'Width2', 'Height2']
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(matches)

def main(batch, visualize=False):
    images_path = f'images/{batch}/'
    images_calib = f'images/{batch}/calib'
    preprocessed_path = f'preprocessed_images/{batch}/'
    matching_output_path = f'output/feature_matching/{batch}/'
    
    # Prétraitement
    process_images_in_folder(images_path, preprocessed_path)
    images = load_images_from_folder(preprocessed_path)

    # Appariement de caractéristiques
    manual_matching = True
    if manual_matching:
        matches = launch_selector(images, matching_output_path)
    else:
        matches = process_feature_matching(images, matching_output_path)
    save_matches_to_csv(matches, 'feature_matches.csv')

    # Chargement des données
    matches, dimensions = load_feature_matches('feature_matches.csv')

    # Estimation des paramètres caméra
    cam_params = estimate_camera_parameters(matches, dimensions, ransac_filter=not manual_matching)
    if manual_matching:
        intrinsics, dist_coeffs = calibrate_camera_with_chessboard(images_calib, (10,7), 2)
    else:
        intrinsics, dist_coeffs = estimate_intrinsic_parameters(matches, dimensions)

    save_camera_params_to_file(cam_params, 'camera_parameters.csv')
    #save_intrinsic_params_to_file(intrinsics, dist_coeffs, 'intrinsic_parameters.csv')

    # Reconstruction 3D
    points_3d = triangulate_points(matches, cam_params, intrinsics, dist_coeffs).astype(np.float64)
    point_cloud = create_point_cloud(points_3d)
    save_point_cloud(point_cloud, 'point_cloud.ply')

    if visualize:
        # Visualisation et export
        visualize_point_cloud(point_cloud)
        mesh = reconstruct_mesh(point_cloud)
        export_mesh(mesh, "reconstructed_mesh.obj")
        o3d.visualization.draw_geometries([mesh], window_name='3D Viewer', width=1920, height=1080)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pipeline de reconstruction 3D à partir d’images.')
    parser.add_argument('-batch', required=True, help='Nom du dossier contenant les images à traiter')
    parser.add_argument('-visualize', action='store_true', help='Visualiser les résultats de la reconstruction 3D')
    args = parser.parse_args()
    main(args.batch, args.visualize)