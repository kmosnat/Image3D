# reconstruction_visualization.py
# Shi Zhang
# This file is for 3D scene reconstruction and visualization

# import statements
import cv2
import os
import csv
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from camera_parameters_estimation import load_camera_parameters, load_feature_matches
from scipy.spatial.transform import Rotation as R


def load_intrinsic_parameters(filename):
    # Loads intrinsic camera parameters from a CSV file.
    # These parameters include the camera matrix and distortion coefficients for each image.

    intrinsic_matrices = {}
    dist_coeffs = {}
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        for row in reader:
            image, intrinsic_flat, dist_flat = row
            # Convert string representations back to numpy arrays
            intrinsic_matrix = np.fromstring(
                intrinsic_flat.strip("[]"), sep=', ').reshape(3, 3)
            dist_coeff = np.fromstring(dist_flat.strip("[]"), sep=', ')
            intrinsic_matrices[image] = intrinsic_matrix
            dist_coeffs[image] = dist_coeff
    return intrinsic_matrices, dist_coeffs


def triangulate_points(matches, camera_params, intrinsic_matrix, dist_coeffs):
    print(intrinsic_matrix)
    print(dist_coeffs)

    all_points_3d = []
    global_poses = {}
    global_poses[None] = (np.eye(3), np.zeros((3, 1)))
    prev_img = None

    for (img1, img2), match_pts in matches.items():
        print(f"Processing pair: {img1} and {img2}")

        if img1 not in camera_params or img2 not in camera_params:
            print(f"Camera parameters for {img1} or {img2} are missing. Skipping.")
            continue

        pts1 = np.float32([pt[0:2] for pt in match_pts])
        pts2 = np.float32([pt[2:4] for pt in match_pts])

        # Correction de la distorsion (désactivée pour le moment)
        pts1_undist = pts1  # Ou utiliser cv2.undistortPoints(...) si nécessaire
        pts2_undist = pts2

        visualize_distortion_field(img1, pts1, intrinsic_matrix, dist_coeffs)
        #visualize_distortion_field(img2, pts2, intrinsic_matrix, dist_coeffs)

        E, mask = cv2.findEssentialMat(
            pts1_undist, pts2_undist, np.eye(3), method=cv2.RANSAC, prob=0.999, threshold=1.0)

        if E is None or E.shape[0] < 3 or E.shape[1] != 3:
            print(f"Invalid essential matrix for pair {img1}-{img2}, skipping.")
            continue
        if E.shape[0] > 3:
            E = E[:3, :3]

        _, R_rel, t_rel, mask_pose = cv2.recoverPose(E, pts1_undist, pts2_undist, np.eye(3))

        inliers = mask_pose.ravel() > 0
        pts1_undist = pts1_undist[inliers]
        pts2_undist = pts2_undist[inliers]

        if img1 not in global_poses:
            if prev_img is None:
                global_poses[img1] = (np.eye(3), np.zeros((3, 1)))
            else:
                global_poses[img1] = global_poses[prev_img]

        R1, t1 = global_poses[img1]
        R2 = R1 @ R_rel
        t2 = R1 @ t_rel + t1
        global_poses[img2] = (R2, t2)

        decompose_pose(R_rel, t_rel, seq='xyz', degrees=True)

        P1 = np.hstack((R1, t1))
        P2 = np.hstack((R2, t2))
        pts_4d_hom = cv2.triangulatePoints(P1, P2, pts1_undist.T, pts2_undist.T)
        pts_3d = pts_4d_hom[:3] / pts_4d_hom[3]

        all_points_3d.append(pts_3d.T)

        plot_3d_points(pts_3d.T, title=f"Nuage 3D pour la paire {img1} - {img2}")
        project_and_show_on_image(img1, pts_3d.T, intrinsic_matrix, R1, t1, pts1_undist)

        prev_img = img2

    points_3d = np.concatenate(all_points_3d, axis=0) if all_points_3d else np.empty((0, 3))
    valid_points = points_3d[~np.isnan(points_3d).any(axis=1) & ~np.isinf(points_3d).any(axis=1)]
    print(f"Total valid 3D points: {len(valid_points)}")

    plot_3d_points(valid_points, title="Nuage de points 3D global")
    plot_depth_histogram(valid_points)
    plot_camera_trajectory(global_poses)
    plot_camera_orientations(global_poses)

    return valid_points


def create_point_cloud(points):
    # Creates a point cloud from a set of 3D points using Open3D.
    # This function is essential for visualizing the reconstructed 3D scene.

    point_cloud = o3d.geometry.PointCloud()
    if points.size > 0:
        point_cloud.points = o3d.utility.Vector3dVector(points)
        print("Creating point cloud with points:", points.shape)
    else:
        print("No points available to create point cloud.")
    return point_cloud


def visualize_point_cloud(point_cloud):
    # Visualizes a point cloud using Open3D's visualization capabilities.
    # This function allows for the inspection and analysis of the reconstructed 3D scene.

    # Remove outliers
    point_cloud, ind = point_cloud.remove_statistical_outlier(
        nb_neighbors=20, std_ratio=2.0)
    point_cloud = point_cloud.select_by_index(ind)
    print("Attempting to visualize point cloud...")
    o3d.visualization.draw_geometries([point_cloud])

def create_point_cloud(points):
    point_cloud = o3d.geometry.PointCloud()
    if points.size > 0:
        point_cloud.points = o3d.utility.Vector3dVector(points)
        print("Creating point cloud with points:", points.shape)
    else:
        print("No points available to create point cloud.")
    return point_cloud

def save_point_cloud(point_cloud, filename):
    """
    Exporte le nuage de points au format .ply (lisible par Blender).
    """
    success = o3d.io.write_point_cloud(filename, point_cloud)
    if success:
        print(f"Point cloud exported successfully to {filename}")
    else:
        print("Failed to export point cloud.")

def reconstruct_mesh(
    point_cloud: o3d.geometry.PointCloud,
    *,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
    voxel_size: float | None = None,
    normal_radius: float = 0.1,
    normal_max_nn: int = 30,
    poisson_depth: int = 10,
    poisson_scale: float = 1.1,
    poisson_linear_fit: bool = False,
    density_prune_ratio: float | None = 0.01,
    target_triangle_count: int | None = None,
    laplacian_iterations: int = 5,
):
    
    pc = point_cloud

    # 1. Voxel down-sampling pour uniformiser la densité et accélérer la suite
    if voxel_size:
        pc = pc.voxel_down_sample(voxel_size)

    # 2. Filtrage statistique des outliers
    pc, ind = pc.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    pc = pc.select_by_index(ind)

    # 3. Estimation & orientation cohérente des normales
    pc.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=normal_max_nn)
    )
    pc.orient_normals_consistent_tangent_plane(50)

    # 4. Reconstruction Poisson
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pc,
        depth=poisson_depth,
        scale=poisson_scale,
        linear_fit=poisson_linear_fit,
    )

    # 5. Suppression des triangles de faible densité
    if density_prune_ratio is not None:
        dens = np.asarray(densities)
        thresh = np.quantile(dens, density_prune_ratio)
        mesh.remove_vertices_by_mask(dens < thresh)

    # 6. Décimation pour contrôler la taille du maillage
    if target_triangle_count is not None and target_triangle_count < len(mesh.triangles):
        mesh = mesh.simplify_quadric_decimation(target_triangle_count)

    # 7. Lissage léger pour réduire les artefacts
    if laplacian_iterations > 0:
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=laplacian_iterations)

    # 8. Nettoyage topologique final
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    return mesh


def export_mesh(mesh, filename):  # [AJOUT]
    # Sauvegarde du maillage en fichier .obj
    success = o3d.io.write_triangle_mesh(filename, mesh)
    if success:
        print(f"Mesh exported successfully to {filename}")
    else:
        print("Failed to export mesh.")

def visualize_distortion_field(image_name, pts, intrinsic, dist):
    """
    Affiche le champ de correction de distorsion pour les points donnés sur l'image.
    - image_name: nom du fichier image (ex: 'pic.0280.jpg' ou 'model_000.png')
    - pts: np.array de shape (N,2), points d'intérêt (x, y)
    - intrinsic: matrice intrinsèque
    - dist: coefficients de distorsion
    """
    # Recherche du chemin de l'image dans le dossier images/
    found = False
    for root, dirs, files in os.walk('preprocessed_images'):
        if image_name in files:
            img_path = os.path.join(root, image_name)
            found = True
            break
    if not found:
        print(f"[Info] Image {image_name} non trouvée dans preprocessed_images/.")
        return
    image = cv2.imread(img_path)
    if image is None:
        print(f"[Info] Impossible de lire l'image {img_path}.")
        return
    pts = np.float32(pts)
    pts_undist = cv2.undistortPoints(np.expand_dims(pts, axis=1), intrinsic, dist, P=intrinsic).squeeze()
    plt.figure(figsize=(10, 10))
    if image.ndim == 2:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #plt.scatter(pts[:, 0], pts[:, 1], color='red', label='Original')
    print(len(pts), len(pts_undist))
    plt.scatter(pts_undist[:, 0], pts_undist[:, 1], color='blue', label='Corrigé')
    for (x0, y0), (x1, y1) in zip(pts, pts_undist):
        plt.arrow(x0, y0, x1 - x0, y1 - y0, color='green', head_width=6, length_includes_head=True)
    plt.legend()
    plt.title(f'Champ de correction de distorsion: {image_name}')
    plt.show(block=False)
    plt.pause(0.001)

def decompose_pose(R_mat, t_vec, seq='xyz', degrees=True):
    """
    Décompose une pose (matrice de rotation + translation) en angles d'Euler et translation.
    - R_mat: matrice de rotation 3x3
    - t_vec: vecteur de translation 3x1 ou 3,
    - seq: séquence d'axes pour les angles d'Euler ('xyz', 'zyx', etc.)
    - degrees: True pour obtenir les angles en degrés
    Retourne: angles (tuple), translation (tuple)
    """
    rot = R.from_matrix(R_mat)
    angles = rot.as_euler(seq, degrees=degrees)
    t = t_vec.flatten()
    print(f"Rotation (x, y, z): {angles}, Translation: {t}")
    return angles, t

def plot_3d_points(points_3d, title="Nuage de points 3D"):
    """
    Affiche un nuage de points 3D avec matplotlib.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    plt.show(block=False)
    plt.pause(0.001)

def plot_depth_histogram(points_3d):
    """
    Affiche l'histogramme des profondeurs (z) des points 3D.
    """
    import matplotlib.pyplot as plt
    plt.figure()
    plt.hist(points_3d[:, 2], bins=100)
    plt.xlabel('Z (profondeur)')
    plt.ylabel('Nombre de points')
    plt.title('Distribution des profondeurs')
    plt.show(block=False)
    plt.pause(0.001)

def plot_camera_trajectory(global_poses):
    """
    Affiche la trajectoire des caméras estimées.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    traj = np.array([t.flatten() for R, t in global_poses.values() if t is not None])
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Trajectoire des caméras')
    plt.show(block=False)
    plt.pause(0.001)

def plot_camera_orientations(global_poses):
    """
    Affiche la trajectoire des caméras estimées avec orientation (axes locaux).
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    traj = np.array([t.flatten() for R, t in global_poses.values() if t is not None])
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], marker='o', label='Trajectoire')
    # Affichage des axes locaux pour chaque caméra
    for R, t in global_poses.values():
        if t is not None:
            t = t.flatten()
            # Les trois axes locaux
            scale = 0.1  # Ajuste la taille des axes
            x_axis = R[:, 0] * scale
            y_axis = R[:, 1] * scale
            z_axis = R[:, 2] * scale
            ax.quiver(t[0], t[1], t[2], x_axis[0], x_axis[1], x_axis[2], color='r', length=scale, normalize=True)
            ax.quiver(t[0], t[1], t[2], y_axis[0], y_axis[1], y_axis[2], color='g', length=scale, normalize=True)
            ax.quiver(t[0], t[1], t[2], z_axis[0], z_axis[1], z_axis[2], color='b', length=scale, normalize=True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Trajectoire et orientation des caméras')
    ax.legend()
    plt.show(block=False)
    plt.pause(0.001)

def project_and_show_on_image(image_name, points_3d, intrinsic, R, t, original_2d_points=None):
    """
    Projette les points 3D sur l'image d'origine et affiche le résultat.
    - image_name: nom du fichier image (ex: 'pic.0280.jpg')
    - points_3d: (N,3) points 3D à projeter
    - intrinsic: matrice intrinsèque de la caméra
    - R, t: pose de la caméra (rotation, translation)
    - original_2d_points: (N,2) points 2D originaux (optionnel, pour comparaison)
    """
    found = False
    for root, dirs, files in os.walk('preprocessed_images'):
        if image_name in files:
            img_path = os.path.join(root, image_name)
            found = True
            break
    if not found:
        print(f"[Info] Image {image_name} non trouvée dans preprocessed_images/.")
        return
    image = cv2.imread(img_path)
    if image is None:
        print(f"[Info] Impossible de lire l'image {img_path}.")
        return
    points_3d = np.asarray(points_3d)
    # Mise à l'échelle des X,Y des points 3D selon les min/max des points 2D originaux
    rvec, _ = cv2.Rodrigues(R)
    tvec = t.reshape(3, 1)
    pts2d_proj, _ = cv2.projectPoints(points_3d, rvec, tvec, intrinsic, None)
    pts2d_proj = pts2d_proj.squeeze()
    if original_2d_points is not None and len(original_2d_points) > 0 and len(points_3d) > 0:
        min_2d, max_2d = original_2d_points.min(axis=0), original_2d_points.max(axis=0)
        min_proj, max_proj = pts2d_proj.min(axis=0), pts2d_proj.max(axis=0)
        min_3d, max_3d = points_3d[:, :2].min(axis=0), points_3d[:, :2].max(axis=0)
        
        scale = (max_2d - min_2d) / (max_3d - min_3d + 1e-8)
        scale_proj = (max_2d - min_2d) / (max_proj - min_proj + 1e-8)
        
        scale = np.mean(scale)  # Pour garder le ratio
        scale_proj = np.mean(scale_proj)  # Pour garder le ratio
        
        points_3d_scaled = points_3d.copy()
        pts2d_proj_scaled = pts2d_proj.copy()
        
        points_3d_scaled[:, 0] = (points_3d[:, 0] - min_3d[0]) * scale + min_2d[0]
        points_3d_scaled[:, 1] = (points_3d[:, 1] - min_3d[1]) * scale + min_2d[1]
        
        pts2d_proj_scaled[:, 0] = (pts2d_proj[:, 0] - min_proj[0]) * scale_proj + min_2d[0]
        pts2d_proj_scaled[:, 1] = (pts2d_proj[:, 1] - min_proj[1]) * scale_proj + min_2d[1]
    else:
        points_3d_scaled = points_3d
    # Projection des points 3D sur le plan image (avec la calibration)
   
    # Coloration des points 3D selon le niveau de gris de l'image (si possible)
    colors_3d = 'lime'
    if image is not None and len(points_3d_scaled) > 0:
        h, w = image.shape[:2]
        # On prend les coordonnées XY projetées (après mise à l'échelle) dans le plan image
        xy = np.round(pts2d_proj_scaled).astype(int)
        # Clamp pour rester dans l'image
        xy[:, 0] = np.clip(xy[:, 0], 0, w - 1)
        xy[:, 1] = np.clip(xy[:, 1], 0, h - 1)
        # Récupérer le niveau de gris (moyenne RGB si image couleur)
        if image.ndim == 3:
            gray = image[xy[:, 1], xy[:, 0]].mean(axis=1) / 255.0
        else:
            gray = image[xy[:, 1], xy[:, 0]] / 255.0
        # Utiliser une colormap matplotlib pour la couleur
        import matplotlib.cm as cm
        cmap = cm.get_cmap('gray')
        colors_3d = cmap(gray)
    # Affichage
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    # Afficher l'image dans le plan (x, y, 0)
    if image is not None:
        h, w = image.shape[:2]
        x_img = np.linspace(0, w, w)
        y_img = np.linspace(0, h, h)
        X_img, Y_img = np.meshgrid(x_img, y_img)
        Z_img = np.zeros_like(X_img)
        #img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        #ax.plot_surface(X_img, Y_img, Z_img, rstride=10, cstride=10, facecolors=img_rgb, shade=False, alpha=0.5)
    # Afficher les points 2D originaux dans le plan z=0
    if original_2d_points is not None and len(original_2d_points) > 0:
        ax.scatter(original_2d_points[:, 0], original_2d_points[:, 1], np.zeros_like(original_2d_points[:, 0]), color='red', s=10, label='Points 2D originaux')
    # Afficher les points 3D (mis à l'échelle) avec couleur
    ax.scatter(points_3d_scaled[:, 0], points_3d_scaled[:, 1], points_3d_scaled[:, 2], color=colors_3d, s=10, label="Points 3D (XY mis à l'échelle, niveau de gris)")
    # Afficher la projection des points 3D sur le plan z=0
    if len(pts2d_proj_scaled.shape) == 1:
        pts2d_proj_scaled = pts2d_proj_scaled[None, :]
    ax.scatter(pts2d_proj_scaled[:, 0], pts2d_proj_scaled[:, 1], np.zeros_like(pts2d_proj_scaled[:, 0]), color='blue', s=10, label='Points 3D projetés (z=0)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(f'Image, points 2D (z=0), points 3D (XY mis à l\'échelle) et projection 3D->2D')
    ax.legend()
    plt.show(block=False)
    plt.pause(0.001)