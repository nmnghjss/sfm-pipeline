import sys
import numpy as np
import sqlite3
import time
import os
import logging
from defs import ViewGraph, ImagePair, Cameras, Images, CameraModelId, ConfigurationType


IS_PYTHON3 = sys.version_info[0] >= 3
MAX_IMAGE_ID = 2**31 - 1

CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL)"""

CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    prior_qw REAL,
    prior_qx REAL,
    prior_qy REAL,
    prior_qz REAL,
    prior_tx REAL,
    prior_ty REAL,
    prior_tz REAL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))
""".format(MAX_IMAGE_ID)

CREATE_TWO_VIEW_GEOMETRIES_TABLE = """
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB)
"""

CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
"""

CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB)"""

CREATE_NAME_INDEX = \
    "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"

CREATE_ALL = "; ".join([
    CREATE_CAMERAS_TABLE,
    CREATE_IMAGES_TABLE,
    CREATE_KEYPOINTS_TABLE,
    CREATE_DESCRIPTORS_TABLE,
    CREATE_MATCHES_TABLE,
    CREATE_TWO_VIEW_GEOMETRIES_TABLE,
    CREATE_NAME_INDEX
])


def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * MAX_IMAGE_ID + image_id2


def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) / MAX_IMAGE_ID
    return image_id1, image_id2

def array_to_blob(array):
    # 确保输入是 numpy 数组
    if not isinstance(array, np.ndarray):
        array = np.asanyarray(array)

    # 在 NumPy 2.x 中，必须使用 tobytes()
    return array.tobytes()


def blob_to_array(blob, dtype, shape=(-1,)):
    return np.frombuffer(blob, dtype=dtype).reshape(*shape)

class COLMAPDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)


    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)

        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_cameras_table = \
            lambda: self.executescript(CREATE_CAMERAS_TABLE)
        self.create_descriptors_table = \
            lambda: self.executescript(CREATE_DESCRIPTORS_TABLE)
        self.create_images_table = \
            lambda: self.executescript(CREATE_IMAGES_TABLE)
        self.create_two_view_geometries_table = \
            lambda: self.executescript(CREATE_TWO_VIEW_GEOMETRIES_TABLE)
        self.create_keypoints_table = \
            lambda: self.executescript(CREATE_KEYPOINTS_TABLE)
        self.create_matches_table = \
            lambda: self.executescript(CREATE_MATCHES_TABLE)
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

    def add_camera(self, model, width, height, params,
                   prior_focal_length=False, camera_id=None):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
            (camera_id, model, width, height, array_to_blob(params),
             prior_focal_length))
        return cursor.lastrowid

    def add_image(self, name, camera_id,
                  prior_q=np.full(4, np.nan), prior_t=np.full(3, np.nan), image_id=None):
        cursor = self.execute(
            "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (image_id, name, camera_id, prior_q[0], prior_q[1], prior_q[2],
             prior_q[3], prior_t[0], prior_t[1], prior_t[2]))
        return cursor.lastrowid

    def add_keypoints(self, image_id, keypoints):
        assert(len(keypoints.shape) == 2)
        assert(keypoints.shape[1] in [2, 4, 6])

        keypoints = np.asarray(keypoints, np.float32)
        self.execute(
            "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
            (image_id,) + keypoints.shape + (array_to_blob(keypoints),))

    def add_descriptors(self, image_id, descriptors):
        descriptors = np.ascontiguousarray(descriptors, np.uint8)
        self.execute(
            "INSERT INTO descriptors VALUES (?, ?, ?, ?)",
            (image_id,) + descriptors.shape + (array_to_blob(descriptors),))

    def add_matches(self, image_id1, image_id2, matches):
        assert(len(matches.shape) == 2)
        assert(matches.shape[1] == 2)

        if image_id1 > image_id2:
            matches = matches[:,::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        self.execute(
            "INSERT INTO matches VALUES (?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches),))

    def add_two_view_geometry(self, image_id1, image_id2, matches,
                              F=np.eye(3), E=np.eye(3), H=np.eye(3), config=2):
        assert(len(matches.shape) == 2)
        assert(matches.shape[1] == 2)

        if image_id1 > image_id2:
            matches = matches[:,::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        F = np.asarray(F, dtype=np.float64)
        E = np.asarray(E, dtype=np.float64)
        H = np.asarray(H, dtype=np.float64)
        self.execute(
            "INSERT INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches), config,
             array_to_blob(F), array_to_blob(E), array_to_blob(H)))
        
def PairId2IdsInversed(pair_id):
    image_id2 = pair_id % 2147483647
    image_id1 = (pair_id - image_id2) // 2147483647
    return image_id1, image_id2

def ReadColmapDatabase(path):
    start_time = time.time()
    view_graph = ViewGraph()
    db = COLMAPDatabase.connect(path)
    
    # Read images into temporary dict for initial processing
    # Create temporary image data structures
    images_dict = {}
    for id, filename, cam_id in db.execute("SELECT image_id, name, camera_id FROM images"):
        images_dict[id] = {
            'id': id,
            'filename': filename,
            'cam_id': cam_id,
            'features': np.array([]),
            'is_registered': False,
            'cluster_id': -1,
            'world2cam': np.eye(4),
            'depths': np.array([]),
            'features_undist': np.array([]),
            'point3d_ids': [],
            'num_points3d': 0,
            'partner_ids': {}
        }
    # group images by their folder names
    image_folders = {}
    for image_data in images_dict.values():
        folder_name = os.path.dirname(image_data['filename'])
        if folder_name not in image_folders:
            image_folders[folder_name] = []
        image_folders[folder_name].append(image_data)

    # Create temporary camera data structures
    camera_records = {}
    for id, model_id, width, height, params, prior_focal_length in db.execute("SELECT * FROM cameras"):
        camera_records[id] = {
            'id': id,
            'model_id': CameraModelId(model_id),
            'width': width,
            'height': height,
            'params': blob_to_array(params, np.float64),
            'has_prior_focal_length': prior_focal_length > 0
        }
    
    keypoints = [(image_id, blob_to_array(data, np.float32, (-1, cols)))
                 for image_id, cols, data in db.execute("SELECT image_id, cols, data FROM keypoints") if not data is None]
    for image_id, data in keypoints:
        images_dict[image_id]['features'] = data[:, :2]

    query = """
    SELECT m.pair_id, m.data, t.data, t.config, t.F, t.E, t.H
    FROM matches AS m
    INNER JOIN two_view_geometries AS t ON m.pair_id = t.pair_id
    """
    matches_and_geometries = db.execute(query)
    image_pairs = {}
    invalid_count = 0

    for group in matches_and_geometries:
        pair_id, data, data2, config, F_blob, E_blob, H_blob = group
        image_id1, image_id2 = PairId2IdsInversed(pair_id)
        name1 = images_dict[image_id1]['filename']
        name2 = images_dict[image_id2]['filename']        
        if data2 is None:
            # invalid_count += 1
            # print(f"Warning: pair {name1} - {name2}: No two-view geometry found, skipping this pair.")
            continue

        # ======================================================================
        init_match = blob_to_array(data, np.uint32, (-1, 2))
        verified_match = blob_to_array(data2, np.uint32, (-1, 2))
        pair_key = (image_id1, image_id2)
        image_pairs[pair_key] = ImagePair(image_id1=image_id1, image_id2=image_id2)
        keypoints1 = images_dict[image_id1]['features']
        keypoints2 = images_dict[image_id2]['features']
        idx1 = verified_match[:, 0]
        idx2 = verified_match[:, 1]
        valid_indices = (idx1 != -1) & (idx2 != -1) & (idx1 < len(keypoints1)) & (idx2 < len(keypoints2))
        valid_matches = verified_match[valid_indices]
        image_pairs[pair_key].matches = valid_matches
        if len(valid_matches) / len(init_match) < 0.5:
            print(f"Warning: pair {name1} - {name2}: Only {len(valid_matches)} valid matches found out of {len(init_match)} initial matches.")
        # print(f"Pair {name1} - {name2}: {len(init_match)} initial matches, {len(verified_match)} verified, {len(valid_matches)} valid matches found.")
        # =========================================================================

        # data = blob_to_array(data, np.uint32, (-1, 2))
        # # Convert COLMAP pair_id to image IDs
        # image_id2 = pair_id % 2147483647
        # image_id1 = (pair_id - image_id2) // 2147483647
        # pair_key = (image_id1, image_id2)
        # image_pairs[pair_key] = ImagePair(image_id1=image_id1, image_id2=image_id2)
        # keypoints1 = images_dict[image_id1]['features']
        # keypoints2 = images_dict[image_id2]['features']
        # idx1 = data[:, 0]
        # idx2 = data[:, 1]
        # valid_indices = (idx1 != -1) & (idx2 != -1) & (idx1 < len(keypoints1)) & (idx2 < len(keypoints2))
        # valid_matches = data[valid_indices]
        # image_pairs[pair_key].matches = valid_matches
        # =====================================================================


        idx1_init = init_match[:, 0]
        idx2_init = init_match[:, 1]
        valid_indices_init = (idx1_init != -1) & (idx2_init != -1) & (idx1_init < len(keypoints1)) & (idx2_init < len(keypoints2))        
        image_pairs[pair_key].matches_init_num = len(init_match[valid_indices_init])

        config = ConfigurationType(config)
        image_pairs[pair_key].config = config
        if config in [ConfigurationType.UNDEFINED, ConfigurationType.DEGENERATE, ConfigurationType.WATERMARK, ConfigurationType.MULTIPLE]:
            image_pairs[pair_key].is_valid = False
            invalid_count += 1
            continue

        F = blob_to_array(F_blob, np.float64).reshape(3, 3)
        E = blob_to_array(E_blob, np.float64).reshape(3, 3)
        H = blob_to_array(H_blob, np.float64).reshape(3, 3)
        image_pairs[pair_key].F = F
        image_pairs[pair_key].E = E
        image_pairs[pair_key].H = H
        image_pairs[pair_key].config = config

    view_graph.image_pairs = {pair_key: image_pair for pair_key, image_pair in image_pairs.items() if image_pair.is_valid}
    print(f'Pairs read done. {invalid_count} / {len(image_pairs)+invalid_count} are invalid')

    # Convert dict to Images container with ID remapping
    camera_items = sorted(camera_records.items())
    cam_id2idx = {cam_id: idx for idx, (cam_id, _) in enumerate(camera_items)}
    cameras = Cameras(num_cameras=len(camera_items))
    for idx, (cam_id, cam_data) in enumerate(camera_items):
        # Camera ID is now the same as index, no need to set cameras.ids
        cameras.model_ids[idx] = cam_data['model_id'].value
        cameras.widths[idx] = cam_data['width']
        cameras.heights[idx] = cam_data['height']
        cameras.has_prior_focal_length[idx] = cam_data['has_prior_focal_length']
        cameras.set_params(idx, cam_data['params'], cam_data['model_id'])
    
    img_id2idx = {img_id:idx for idx, img_id in enumerate(images_dict.keys())}
    
    # Create Images container
    images = Images(num_images=len(images_dict))
    for idx, (img_id, image_data) in enumerate(sorted(images_dict.items())):
        images.ids[idx] = img_id2idx[img_id]
        images.cam_ids[idx] = cam_id2idx[image_data['cam_id']]
        images.filenames[idx] = image_data['filename']
        images.is_registered[idx] = image_data['is_registered']
        images.cluster_ids[idx] = image_data['cluster_id']
        images.world2cams[idx] = image_data['world2cam']
        images.features[idx] = image_data['features']
        images.depths[idx] = image_data['depths']
        images.features_undist[idx] = image_data['features_undist']
        images.point3d_ids[idx] = image_data['point3d_ids']
        images.num_points3d[idx] = image_data['num_points3d']
        images.partner_ids[idx] = image_data['partner_ids']
    
    # Update image pair IDs to use the new sequential indices
    updated_pairs = {}
    for (old_id1, old_id2), pair in view_graph.image_pairs.items():
        new_id1 = img_id2idx[old_id1]
        new_id2 = img_id2idx[old_id2]
        pair.image_id1 = new_id1
        pair.image_id2 = new_id2
        updated_pairs[(new_id1, new_id2)] = pair
    view_graph.image_pairs = updated_pairs

    # assign image partners here
    first_folder = list(image_folders.values())[0]
    for idx in range(len(first_folder)):
        image_group = {folder_name: img_id2idx[folder[idx]['id']] for folder_name, folder in image_folders.items()}
        for folder in image_folders.values():
            image_idx = img_id2idx[folder[idx]['id']]
            images.partner_ids[image_idx] = image_group

    print(f'Reading database took: {time.time() - start_time:.2f}')

    try:
        feature_name = db.execute("SELECT feature_name FROM feature_name").fetchone()[0]
    except:
        # if the database does not have feature_name, then assume it's originated from COLMAP-compatibale workflow
        feature_name = 'colmap'

    return view_graph, cameras, images, feature_name


def filter_matches_by_inliers(database_path, min_num_inliers=30, min_inlier_ratio=0.1, logger=None):
    """
    根据内点数量和内点比例阈值过滤数据库中的匹配对，删除不满足条件的匹配
    
    仿照 ReadColmapDatabase 函数的结构，该函数：
    1. 连接 COLMAP 数据库
    2. 遍历所有匹配对
    3. 检查每对的验证匹配数量和内点比例
    4. 删除不满足阈值条件的匹配对 (both from matches and two_view_geometries tables)
    
    Args:
        database_path (str): COLMAP 数据库文件路径
        min_num_inliers (int): 最少内点数量阈值，默认30
        min_inlier_ratio (float): 最少内点比例阈值，范围0.0-1.0，默认0.1
        logger (logging.Logger): 可选的日志记录器，如果为None则使用默认logger
    
    Returns:
        int: 删除的匹配对数量
        
    Example:
        >>> import logging
        >>> logger = logging.getLogger()
        >>> removed = filter_matches_by_inliers('database.db', min_num_inliers=40, min_inlier_ratio=0.15, logger=logger)
        >>> print(f"Removed {removed} match pairs")
    """
    if logger is None:
        logger = logging.getLogger()
    
    start_time = time.time()
    logger.info(f"Filtering database: {database_path}")
    logger.info(f"Thresholds: min_num_inliers={min_num_inliers}, min_inlier_ratio={min_inlier_ratio}")
    
    try:
        db = COLMAPDatabase.connect(database_path)
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        return 0
    
    # 读取所有图像信息用于显示文件名
    images_dict = {}
    for id, filename, cam_id in db.execute("SELECT image_id, name, camera_id FROM images"):
        images_dict[id] = {
            'id': id,
            'filename': filename,
            'cam_id': cam_id
        }
    
    # 从两视图几何表中读取所有匹配对信息
    # 注意：这里JOIN确保只读取有几何验证的匹配对
    query = """
    SELECT m.pair_id, m.data, t.data
    FROM matches AS m
    INNER JOIN two_view_geometries AS t ON m.pair_id = t.pair_id
    """
    matches_and_geometries = list(db.execute(query))
    logger.info(f"Found {len(matches_and_geometries)} image pairs in database")
    
    removed_count = 0
    kept_count = 0
    problematic_pairs = []  # 用于记录被删除的对信息
    
    for pair_id, init_match_data, verified_match_data in matches_and_geometries:
        # 解析图像ID
        image_id1, image_id2 = PairId2IdsInversed(pair_id)
        name1 = images_dict[image_id1]['filename']
        name2 = images_dict[image_id2]['filename']
        
        if verified_match_data is None:
            # 没有验证的几何数据，保留但记录警告
            logger.warning(f"Pair {name1} - {name2}: No verified geometry data found, keeping it")
            kept_count += 1
            continue
        
        try:
            # 解析匹配数据
            init_match = blob_to_array(init_match_data, np.uint32, (-1, 2)) if init_match_data is not None else np.array([])
            verified_match = blob_to_array(verified_match_data, np.uint32, (-1, 2))
            
            num_init = len(init_match)
            num_verified = len(verified_match)
            
            # 计算内点比例
            inlier_ratio = num_verified / num_init if num_init > 0 else 0.0
            
            # 检查是否满足阈值条件
            should_remove = False
            reason = []
            
            if num_verified < min_num_inliers:
                should_remove = True
                reason.append(f"inliers={num_verified}<{min_num_inliers}")
            
            if inlier_ratio < min_inlier_ratio:
                should_remove = True
                reason.append(f"ratio={inlier_ratio:.4f}<{min_inlier_ratio}")
            
            if should_remove:
                # 删除这对匹配
                db.execute("DELETE FROM matches WHERE pair_id = ?", (pair_id,))
                db.execute("DELETE FROM two_view_geometries WHERE pair_id = ?", (pair_id,))
                removed_count += 1
                
                reason_str = ", ".join(reason)
                logger.debug(f"Removed pair {name1} - {name2}: {reason_str} "
                           f"(init={num_init}, verified={num_verified})")
                problematic_pairs.append({
                    'pair': f"{name1} - {name2}",
                    'init_matches': num_init,
                    'verified_matches': num_verified,
                    'inlier_ratio': inlier_ratio,
                    'reason': reason_str
                })
            else:
                kept_count += 1
                
        except Exception as e:
            logger.error(f"Error processing pair {name1} - {name2}: {e}")
    
    # 提交更改
    db.commit()
    db.close()
    
    elapsed_time = time.time() - start_time
    logger.info(f"Database filtering completed in {elapsed_time:.2f}s")
    logger.info(f"Summary: Removed {removed_count} pairs, Kept {kept_count} pairs")
    
    if removed_count > 0:
        logger.info("Top problematic pairs removed:")
        for i, pair_info in enumerate(problematic_pairs[:10]):  # 显示前10个
            logger.info(f"  {i+1}. {pair_info['pair']}: "
                       f"init={pair_info['init_matches']}, "
                       f"verified={pair_info['verified_matches']}, "
                       f"ratio={pair_info['inlier_ratio']:.4f} "
                       f"({pair_info['reason']})")
        if len(problematic_pairs) > 10:
            logger.info(f"  ... and {len(problematic_pairs)-10} more pairs")
    
    return removed_count
