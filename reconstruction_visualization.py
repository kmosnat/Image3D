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
    intrinsic_matrices = {}
    dist_coeffs = {}
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            image, intrinsic_flat, dist_flat = row
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
    first_img = None
    
    undistorted_images = {}
    for img_name in set([k[0] for k in matches.keys()] + [k[1] for k in matches.keys()]):
        img_path = None
        for root, dirs, files in os.walk('preprocessed_images'):
            if img_name in files:
                img_path = os.path.join(root, img_name)
                break
        if img_path is not None:
            img = cv2.imread(img_path)
            if img is not None:
                undistorted = cv2.undistort(img, intrinsic_matrix, dist_coeffs)
                undistorted_images[img_name] = undistorted
            else:
                print(f"[Info] Impossible de lire l'image {img_path}.")
        else:
            print(f"[Info] Image {img_name} non trouvée dans preprocessed_images/.")

    processed_images = set()
    
    for (img1, img2), match_pts in matches.items():
        print(f"Processing pair: {img1} and {img2}")

        if img1 not in camera_params or img2 not in camera_params:
            print(f"Camera parameters for {img1} or {img2} are missing. Skipping.")
            continue

        if img1 in processed_images and img2 in processed_images:
            continue

        img1_undist = undistorted_images.get(img1, None)
        img2_undist = undistorted_images.get(img2, None)
        
        if img1_undist is None or img2_undist is None:
            print(f"Could not load undistorted images for {img1} or {img2}")
            continue

        pts1 = np.float32([pt[0:2] for pt in match_pts])
        pts2 = np.float32([pt[2:4] for pt in match_pts])
        
        # Normalisation des coordonnées pour améliorer la stabilité numérique
        h1, w1 = img1_undist.shape[:2]
        h2, w2 = img2_undist.shape[:2]
        
        # Normalisation : centre à (0,0) et échelle [-1,1]
        pts1_norm = np.copy(pts1)
        pts1_norm[:, 0] = (pts1[:, 0] - w1/2) / (w1/2)
        pts1_norm[:, 1] = (pts1[:, 1] - h1/2) / (h1/2)
        
        pts2_norm = np.copy(pts2)
        pts2_norm[:, 0] = (pts2[:, 0] - w2/2) / (w2/2)
        pts2_norm[:, 1] = (pts2[:, 1] - h2/2) / (h2/2)
        
        # Matrice intrinsèque normalisée
        K_norm = np.array([[2.0/w1, 0, 0],
                          [0, 2.0/h1, 0],
                          [0, 0, 1]])
        
        pts1_undist = cv2.undistortPoints(
            np.expand_dims(pts1, axis=1), intrinsic_matrix, dist_coeffs, P=intrinsic_matrix).squeeze()
        pts2_undist = cv2.undistortPoints(
            np.expand_dims(pts2, axis=1), intrinsic_matrix, dist_coeffs, P=intrinsic_matrix).squeeze()
        
        if pts1_undist.ndim == 1:
            pts1_undist = pts1_undist.reshape(1, -1)
        if pts2_undist.ndim == 1:
            pts2_undist = pts2_undist.reshape(1, -1)

        # Utiliser les points normalisés pour la matrice essentielle
        E, mask_E = cv2.findEssentialMat(
            pts1_norm, pts2_norm, K_norm, 
            method=cv2.RANSAC, prob=0.999, threshold=0.01)  # Seuil réduit pour points normalisés

        if E is None or E.shape[0] < 3 or E.shape[1] != 3:
            print(f"Invalid essential matrix for pair {img1}-{img2}, skipping.")
            continue
        if E.shape[0] > 3:
            E = E[:3, :3]

        _, R_rel, t_rel, mask_pose = cv2.recoverPose(E, pts1_norm, pts2_norm, K_norm)

        # Combine masks to get good inliers
        if mask_E.shape[0] != mask_pose.shape[0]:
            print(f"Mask size mismatch for {img1}-{img2}, using pose mask only")
            good_matches = mask_pose.ravel() > 0
        else:
            good_matches = (mask_E.ravel() > 0) & (mask_pose.ravel() > 0)
        
        # Filter points to inliers only
        pts1_inliers = pts1_undist[good_matches]
        pts2_inliers = pts2_undist[good_matches]
        
        print(f"Good matches: {np.sum(good_matches)} out of {len(pts1_undist)}")
        
        if np.sum(good_matches) < 8:
            print(f"Too few good matches ({np.sum(good_matches)}) for {img1}-{img2}")
            continue

        if first_img is None:
            first_img = img1
            global_poses[img1] = (np.eye(3), np.zeros((3, 1)))
            
        if img1 in global_poses:
            R1, t1 = global_poses[img1]
            R2 = R1 @ R_rel
            t2 = R1 @ t_rel + t1
            global_poses[img2] = (R2, t2)
        elif img2 in global_poses:
            R2, t2 = global_poses[img2]
            R1 = R2 @ R_rel.T
            t1 = R2 @ (-R_rel.T @ t_rel) + t2
            global_poses[img1] = (R1, t1)
        else:
            global_poses[img1] = (np.eye(3), np.zeros((3, 1)))
            R1, t1 = global_poses[img1]
            R2 = R1 @ R_rel
            t2 = R1 @ t_rel + t1
            global_poses[img2] = (R2, t2)

        R1, t1 = global_poses[img1]
        R2, t2 = global_poses[img2]

        P1 = intrinsic_matrix @ np.hstack((R1, t1))
        P2 = intrinsic_matrix @ np.hstack((R2, t2))
        
        pts_4d_hom = cv2.triangulatePoints(P1, P2, pts1_inliers.T, pts2_inliers.T)
        pts_3d = pts_4d_hom[:3] / (pts_4d_hom[3] + 1e-8)
        pts_3d = pts_3d.T
        
        # Filtrage amélioré avec seuils adaptatifs
        # Calculer la distance médiane pour adapter les seuils
        distances = np.linalg.norm(pts_3d, axis=1)
        median_dist = np.median(distances)
        
        # Filtres adaptatifs basés sur la distribution des points
        valid_depth = (pts_3d[:, 2] > 0.1) & (pts_3d[:, 2] < median_dist * 5)
        valid_distance = distances < median_dist * 3
        
        # Filtrer les outliers statistiques
        z_scores = np.abs((distances - np.mean(distances)) / (np.std(distances) + 1e-8))
        valid_stats = z_scores < 2.0  # Seuil à 2 sigma
        
        valid_mask = valid_depth & valid_distance & valid_stats
        
        pts_3d_valid = pts_3d[valid_mask]
        
        if len(pts_3d_valid) > 0:
            all_points_3d.append(pts_3d_valid)
            print(f"Added {len(pts_3d_valid)} valid 3D points from pair {img1}-{img2}")
            print(f"Distance range: {np.min(distances[valid_mask]):.3f} - {np.max(distances[valid_mask]):.3f}")
            plot_3d_points(pts_3d_valid, title=f"Nuage 3D pour la paire {img1} - {img2}")
        else:
            print(f"No valid 3D points for pair {img1}-{img2}")

        processed_images.add(img1)
        processed_images.add(img2)

    # Combine all points with bundle adjustment-like normalization
    if all_points_3d:
        points_3d = np.concatenate(all_points_3d, axis=0)
        
        # Normalisation globale pour corriger les problèmes d'échelle
        centroid = np.mean(points_3d, axis=0)
        points_3d_centered = points_3d - centroid
        
        # Normalisation par l'écart-type pour chaque axe
        std_devs = np.std(points_3d_centered, axis=0)
        std_devs[std_devs < 1e-8] = 1.0  # Éviter la division par zéro
        
        # Appliquer une normalisation adaptative
        scale_factor = np.median(std_devs)
        points_3d_normalized = points_3d_centered / scale_factor
        
        # Additional filtering
        finite_mask = np.isfinite(points_3d_normalized).all(axis=1)
        points_3d_final = points_3d_normalized[finite_mask]
        
        print(f"Total valid 3D points after normalization: {len(points_3d_final)}")
        print(f"Point cloud bounds: X[{np.min(points_3d_final[:,0]):.3f}, {np.max(points_3d_final[:,0]):.3f}], "
              f"Y[{np.min(points_3d_final[:,1]):.3f}, {np.max(points_3d_final[:,1]):.3f}], "
              f"Z[{np.min(points_3d_final[:,2]):.3f}, {np.max(points_3d_final[:,2]):.3f}]")
        
        if len(points_3d_final) > 0:
            plot_3d_points(points_3d_final, title="Nuage de points 3D global normalisé")
            plot_depth_histogram(points_3d_final)
            plot_camera_trajectory(global_poses)
            plot_camera_orientations(global_poses)
        
        return points_3d_final
    else:
        print("No valid 3D points reconstructed")
        return np.empty((0, 3))


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

    if point_cloud.is_empty():
        print("Point cloud is empty, cannot visualize.")
        return
        
    # Remove outliers
    point_cloud, ind = point_cloud.remove_statistical_outlier(
        nb_neighbors=20, std_ratio=2.0)
    point_cloud = point_cloud.select_by_index(ind)
    print("Attempting to visualize point cloud...")
    o3d.visualization.draw_geometries([point_cloud])


def save_point_cloud(point_cloud, filename):
    """
    Exporte le nuage de points au format .ply (lisible par Blender).
    """
    success = o3d.io.write_point_cloud(filename, point_cloud)
    if success:
        print(f"Point cloud exported successfully to {filename}")
    else:
        print("Failed to export point cloud.")


def reconstruct_mesh(point_cloud):
    # 1) Si on reçoit un Open3D PointCloud, on en extrait simplement les points
    if isinstance(point_cloud, o3d.geometry.PointCloud):
        pts_array = np.asarray(point_cloud.points)
    else:
        # Sinon on suppose que c'est déjà un np.ndarray de forme (N, 3)
        pts_array = np.asarray(point_cloud, dtype=np.float64)
        if pts_array.ndim != 2 or pts_array.shape[1] != 3:
            raise ValueError("point_cloud doit être un tableau de forme (N, 3) ou un Open3D PointCloud")

    # 2) Créer un nouveau PointCloud et lui assigner les points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_array)

    # 3) Estimer les normales (voisinage restreint, puisque nuage dense)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30)
    )

    # 4) Forcer l'orientation des normales vers l'extérieur (utile pour une sphère centrée en (0,0,0))
    pts = np.asarray(pcd.points)
    normals = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    # 5) Calculer les rayons pour la BPA à partir de la distance moyenne aux voisins
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist  = float(np.mean(distances))
    # Par exemple, on génère 5 rayons entre 1.0× et 2.0× l'espacement moyen
    radii = o3d.utility.DoubleVector([avg_dist * f for f in np.linspace(1.0, 2.0, 5)])

    # 6) Reconstruction par Ball Pivoting
    mesh_bpa = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)

    # 7) Exporter le maillage en OBJ
    chemin_obj = "reconstruct.obj"
    o3d.io.write_triangle_mesh(chemin_obj, mesh_bpa)

    # 8) Relire l'OBJ pour s'assurer que tout est correct
    mesh_chargé = o3d.io.read_triangle_mesh(chemin_obj)
    print(f"OBJ rechargé : {len(mesh_chargé.vertices)} sommets, {len(mesh_chargé.triangles)} faces")

    # 9) Ajouter les normales si elles manquent (nécessaire pour l'affichage)
    if not mesh_chargé.has_vertex_normals():
        mesh_chargé.compute_vertex_normals()

    # 10) Visualiser la sphère OBJ réimportée
    o3d.visualization.draw_geometries(
        [mesh_chargé],
        window_name="Reconstruction par BPA"
    )

    return mesh_bpa


def export_mesh(mesh, filename):  # [AJOUT]
    # Sauvegarde du maillage en fichier .obj
    success = o3d.io.write_triangle_mesh(filename, mesh)
    if success:
        print(f"Mesh exported successfully to {filename}")
    else:
        print("Failed to export mesh.")


def visualize_distortion_field(image_name, image, pts, intrinsic, dist):
    """
    Affiche le champ de correction de distorsion pour les points donnés sur l'image.
    - image_name: nom du fichier image (ex: 'pic.0280.jpg' ou 'model_000.png')
    - pts: np.array de shape (N,2), points d'intérêt (x, y)
    - intrinsic: matrice intrinsèque
    - dist: coefficients de distorsion
    """
    # Recherche du chemin de l'image dans le dossier images/
    found = False
    pts = np.float32(pts)
    pts_undist = cv2.undistortPoints(np.expand_dims(pts, axis=1), intrinsic, dist, P=intrinsic).squeeze()
    plt.figure(figsize=(10, 10))
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
    Affiche un nuage de points 3D avec matplotlib, trace une ligne entre chaque point et le suivant (et ferme la boucle),
    et affiche une légende (numéro) à côté de chaque point.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], s=15)
    # Tracer les lignes entre chaque point et le suivant, et fermer la boucle
    if len(points_3d) > 1:
        for i in range(len(points_3d)-1):
            p1 = points_3d[i]
            p2 = points_3d[(i + 1) % len(points_3d)]  # le suivant, ou le premier si dernier
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='orange', linewidth=1)
    # Afficher le numéro de chaque point
    for i, (x, y, z) in enumerate(points_3d):
        ax.text(x, y, z, str(i), color='black', fontsize=9, ha='left', va='bottom')
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