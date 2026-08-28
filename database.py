import sys
import numpy as np
import sqlite3
import time
import os
import pathlib
import logging
from PIL import Image
from defs import ViewGraph, ImagePair, Cameras, Images, CameraModelId, ConfigurationType
from utils import count_images_in_dir, get_subfolders

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
    type INTEGER NOT NULL DEFAULT 0,
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

# def array_to_blob(array):
#     if IS_PYTHON3:
#         return array.tostring()
#     else:
#         return np.getbuffer(array)

def array_to_blob(array):
    # 确保输入是 numpy 数组
    if not isinstance(array, np.ndarray):
        array = np.asanyarray(array)
        
    # 在 NumPy 2.x 中，必须使用 tobytes()
    return array.tobytes()


def blob_to_array(blob, dtype, shape=(-1,)):
    # if IS_PYTHON3:
    #     return np.fromstring(blob, dtype=dtype).reshape(*shape)
    # else:
    #     return np.frombuffer(blob, dtype=dtype).reshape(*shape)
    return np.frombuffer(blob, dtype=dtype).reshape(*shape)

def float_descriptors_to_uint8(descriptors: np.ndarray, descriptor_type: int) -> np.ndarray:
    """
    将float32描述子转换为uint8（对齐C++ FeatureDescriptors::FromFloat）
    
    Args:
        descriptors: np.ndarray, shape=(N, D), dtype=float32
        descriptor_type: int, 0=SIFT, 1=ALIKED_N16ROT, 2=ALIKED_N32
        
    Returns:
        np.ndarray, dtype=uint8
    """
    # 确保输入是连续内存的float32
    data = np.ascontiguousarray(descriptors, dtype=np.float32)
    
    if descriptor_type == 0:  # SIFT: 数值转换
        return data.astype(np.uint8)    
    elif descriptor_type in (1, 2):  # ALIKED: 内存重解释
        rows, cols = data.shape
        return data.view(np.uint8).reshape(rows, cols * 4)
    else:
        raise ValueError(f"Unsupported descriptor type: {descriptor_type}")


def read_all_keypoints(database_path):
    """
    从 COLMAP 数据库中读取所有图像的特征点。

    Args:
        database_path: COLMAP 数据库文件路径 (.db)

    Returns:
        dict: {image_id: np.ndarray of shape (N, 2)}, 其中每行为 (x, y) 坐标
              同时返回一个 image_name 映射: dict: {image_id: filename}
    """
    db = COLMAPDatabase.connect(database_path)
    keypoints_dict = {}
    image_names = {}

    # 读取图像名称映射
    for image_id, name in db.execute("SELECT image_id, name FROM images"):
        image_names[image_id] = name

    # 读取所有特征点
    for image_id, cols, data in db.execute(
        "SELECT image_id, cols, data FROM keypoints WHERE data IS NOT NULL"
    ):
        kpts = blob_to_array(data, np.float32, (-1, cols))
        keypoints_dict[image_id] = kpts[:, :2]  # 只取 (x, y)

    db.close()
    return keypoints_dict, image_names


def get_matched_image_pairs(database_path):
    """
    读取 COLMAP 数据库中匹配图像对的个数。

    Args:
        database_path: COLMAP 数据库文件路径 (.db)

    Returns:
        int: 匹配图像对的总数
    """
    db = COLMAPDatabase.connect(database_path)
    row = db.execute("SELECT COUNT(*) FROM matches").fetchone()
    db.close()
    return row[0] if row else 0


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
        keypoints_padded = np.zeros((keypoints.shape[0], 6), np.float32)
        keypoints_padded[:, :keypoints.shape[1]] = keypoints
        self.execute(
            "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
            (image_id,) + keypoints_padded.shape + (array_to_blob(keypoints_padded),))

    # def add_descriptors(self, image_id, descriptors):
    #     descriptors = np.ascontiguousarray(descriptors, np.uint8)
    #     self.execute(
    #         "INSERT INTO descriptors VALUES (?, ?, ?, ?)",
    #         (image_id,) + descriptors.shape + (array_to_blob(descriptors),))

    def add_descriptors(self, image_id, descriptors, descriptor_type = 1):
        descriptors_u8 = float_descriptors_to_uint8(descriptors, descriptor_type)
        descriptors_blob = descriptors_u8.tobytes()        
        self.execute(
            "INSERT INTO descriptors VALUES (?, ?, ?, ?, ?)",
            (
                image_id,                
                descriptors_u8.shape[0],  # rows
                descriptors_u8.shape[1] ,  # cols
                descriptors_blob,      # binary data
                int(descriptor_type)  # 描述子类型
            )
        )

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

        if E_blob is not None:
            image_pairs[pair_key].E = blob_to_array(E_blob, np.float64).reshape(3, 3)
        else:
            print(f"Warning: pair {name1} - {name2}: Essential matrix is None, marking this pair as invalid.")
        if F_blob is not None:
            image_pairs[pair_key].F = blob_to_array(F_blob, np.float64).reshape(3, 3)
        else:
            print(f"Warning: pair {name1} - {name2}: Fundamental matrix is None, marking this pair as invalid.")
        if H_blob is not None:
            image_pairs[pair_key].H = blob_to_array(H_blob, np.float64).reshape(3, 3)
        else:
            print(f"Warning: pair {name1} - {name2}: Homography matrix is None, marking this pair as invalid.")

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



def _validate_camera_params(model_name, model_id, params, expected_counts, logger):
    """Validate that params count matches the camera model's expected count.
    Returns True if valid, False otherwise (error already logged)."""
    expected_count = expected_counts.get(model_id)
    actual_count = len(params)
    if expected_count is not None and actual_count != expected_count:
        logger.error(
            f"Camera model '{model_name}' (id={model_id}) expects "
            f"{expected_count} parameters, but {actual_count} were provided."
        )
        return False
    return True


def _compute_default_params(
    camera_model_id: int, camera_model: str, width: int, height: int,
    prior_fx, prior_fy,
    _EXPECTED_PARAM_COUNTS: dict,
    logger: logging.Logger
):
    """Compute default zero-distortion camera parameters from image dimensions."""
    fx = prior_fx if prior_fx is not None else max(width, height)
    fy = prior_fy if prior_fy is not None else max(width, height)
    cx = width / 2
    cy = height / 2

    if camera_model_id == 0:  # SIMPLE_PINHOLE: f, cx, cy
        params = [max(fx, fy), cx, cy]
    elif camera_model_id == 1:  # PINHOLE: fx, fy, cx, cy
        params = [fx, fy, cx, cy]
    elif camera_model_id == 2:  # SIMPLE_RADIAL: f, cx, cy, k
        params = [max(fx, fy), cx, cy, 0.0]
    elif camera_model_id == 3:  # RADIAL: f, cx, cy, k1, k2
        params = [max(fx, fy), cx, cy, 0.0, 0.0]
    elif camera_model_id == 4:  # OPENCV: fx, fy, cx, cy, k1, k2, p1, p2
        params = [fx, fy, cx, cy, 0.0, 0.0, 0.0, 0.0]
    elif camera_model_id == 5:  # OPENCV_FISHEYE: fx, fy, cx, cy, k1, k2, k3, k4
        params = [fx, fy, cx, cy, 0.0, 0.0, 0.0, 0.0]
    elif camera_model_id == 6:  # FULL_OPENCV
        params = [fx, fy, cx, cy, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    elif camera_model_id == 7:  # FOV: fx, fy, cx, cy, omega
        params = [fx, fy, cx, cy, 0.0]
    elif camera_model_id == 8:  # SIMPLE_RADIAL_FISHEYE: f, cx, cy, k1
        params = [max(fx, fy), cx, cy, 0.0]
    elif camera_model_id == 9:  # RADIAL_FISHEYE: f, cx, cy, k1, k2
        params = [max(fx, fy), cx, cy, 0.0, 0.0]
    elif camera_model_id == 10:  # THIN_PRISM_FISHEYE
        params = [fx, fy, cx, cy, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    elif camera_model_id == 11:  # RAD_TAN_THIN_PRISM_FISHEYE
        params = [fx, fy, cx, cy, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    else:
        logger.warning(f"Unsupported camera model ID {camera_model_id}, defaulting to OPENCV parameters")
        params = [max(width, height), max(width, height), cx, cy, 0.0, 0.0, 0.0, 0.0]
        camera_model_id = 4
        camera_model = "OPENCV"
    return params, camera_model_id, camera_model


def _get_image_dimensions(image_path: str, logger: logging.Logger):
    """Open an image file and return (width, height). Returns None on failure."""
    try:
        img = Image.open(image_path)
        return img.size
    except Exception as e:
        logger.error(f"Failed to read image dimensions: {image_path}. Error: {e}")
        return None


def initialize_colmap_database(
    database_path: str,
    images_dir: str,
    input_images_path: list,
    camera_model: str = "OPENCV",
    camera_assignment: str = "per_subfolder",
    prior_cameras: dict = None,
    prior_images: list = None,
    logger: logging.Logger = None
) -> bool:
    """
    Initialize COLMAP database and add image metadata without extracting SIFT features.

    When ``prior_cameras`` / ``prior_images`` are provided (e.g. from a prior pose model),
    they take priority over ``camera_model`` and ``camera_assignment``: each image's
    camera is taken directly from the prior model.
    When ``prior_images`` is non-empty, ``input_images_path`` is ignored and the images
    listed in ``prior_images`` are written directly, keeping their image IDs and camera
    IDs identical to the prior model.
    Otherwise the original logic applies, with three camera assignment modes via the
    ``camera_assignment`` parameter:

    - ``"global"`` (1): All images share one camera model and intrinsics.
    - ``"per_subfolder"`` (2): Each subfolder under ``images_dir`` gets its own camera.
    - ``"per_image"`` (3): Each image gets its own camera.

    Args:
        database_path: Path to create/initialize COLMAP database
        images_dir: Path to images directory
        input_images_path: List of image file paths
        camera_model: Camera model (e.g., "OPENCV", "PINHOLE") — global fallback
        camera_assignment: Camera assignment mode. One of:
            ``"global"`` — one camera for all images;
            ``"per_subfolder"`` (default) — one camera per subfolder;
            ``"per_image"`` — one camera per image.
        prior_cameras: Optional dict of COLMAP Camera objects (id -> Camera).
            If provided, these cameras are registered directly and take priority
            over ``camera_model`` / ``camera_assignment``.
        prior_images: Optional list of tuples ``(image_id, image_path, camera_id)``.
            When provided, images are written directly from this list (ignoring
            ``input_images_path``), keeping the DB image_id and camera_id identical
            to the prior model.
        logger: Optional logger instance

    Returns:
        True if successful, False otherwise
    """

    # --- Camera model name → ID mapping (reused for per-subfolder configs) ---
    _CAMERA_MODEL_NAME_TO_ID = {
        "SIMPLE_PINHOLE": 0,
        "PINHOLE": 1,
        "SIMPLE_RADIAL": 2,
        "RADIAL": 3,
        "OPENCV": 4,
        "OPENCV_FISHEYE": 5,
        "FULL_OPENCV": 6,
        "FOV": 7,
        "SIMPLE_RADIAL_FISHEYE": 8,
        "RADIAL_FISHEYE": 9,
        "THIN_PRISM_FISHEYE": 10,
        "RAD_TAN_THIN_PRISM_FISHEYE": 11,
    }
    _EXPECTED_PARAM_COUNTS = {
        0: 3,   # SIMPLE_PINHOLE: f, cx, cy
        1: 4,   # PINHOLE: fx, fy, cx, cy
        2: 4,   # SIMPLE_RADIAL: f, cx, cy, k
        3: 5,   # RADIAL: f, cx, cy, k1, k2
        4: 8,   # OPENCV: fx, fy, cx, cy, k1, k2, p1, p2
        5: 8,   # OPENCV_FISHEYE: fx, fy, cx, cy, k1, k2, k3, k4
        6: 12,  # FULL_OPENCV: fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
        7: 5,   # FOV: fx, fy, cx, cy, omega
        8: 4,   # SIMPLE_RADIAL_FISHEYE: f, cx, cy, k1
        9: 5,   # RADIAL_FISHEYE: f, cx, cy, k1, k2
        10: 12, # THIN_PRISM_FISHEYE: fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1
        11: 16, # RAD_TAN_THIN_PRISM_FISHEYE: fx, fy, cx, cy, k0..k5, p0, p1, s0..s3
    }

    if logger is None:
        logger = logging.getLogger()

    valid_modes = {"global", "per_subfolder", "per_image"}
    if camera_assignment not in valid_modes:
        logger.error(f"camera_assignment must be one of {valid_modes}, got '{camera_assignment}'")
        return False

    logger.info("Initializing COLMAP database without feature extraction...")
    logger.info(f"Camera assignment mode: {camera_assignment}")

    # Remove existing database file to ensure a clean schema (avoids
    # column-count mismatch when a prior run left a standard-COLMAP
    # images table with only 3 columns instead of our extended 10).
    if os.path.exists(database_path):
        logger.info(f"Removing existing database: {database_path}")
        try:
            os.remove(database_path)
        except OSError as exc:
            logger.error(f"Failed to remove existing database: {exc}")
            return False

    try:
        db = COLMAPDatabase.connect(database_path)
        db.create_tables()

        if not input_images_path and not prior_images:
            logger.error(f"No images found in {images_dir}")
            db.close()
            return False

        logger.info(f"Input {len(input_images_path)} images")

        # --- Global camera model ID ---
        camera_model_id = _CAMERA_MODEL_NAME_TO_ID.get(camera_model.upper(), 4)
        logger.info(f"Global camera model: {camera_model}, id: {camera_model_id}")

        image_count = 0

        # =====================================================================
        # MODE: prior — use cameras and per-image camera IDs from a prior model
        # =====================================================================
        if prior_cameras is not None:
            # 若 prior_images 为空，直接报错退出。
            if not prior_images:
                logger.error(
                    "prior_cameras is provided but prior_images is empty. "
                    "Cannot initialize database without prior image mappings."
                )
                db.close()
                return False 
                       
            # Register prior cameras in the database, preserving their original IDs
            for cam_id, cam in prior_cameras.items():
                cam_model_id = _CAMERA_MODEL_NAME_TO_ID.get(cam.model.upper(), 4)
                db.add_camera(
                    model=cam_model_id, width=cam.width, height=cam.height,
                    params=np.asarray(cam.params, dtype=np.float64),
                    prior_focal_length=True,
                    camera_id=cam_id
                )
                logger.info(f"Added prior camera: model={cam.model}, id={cam_id}")
            db.commit()

            # 当 prior_images 非空时，忽略 input_images_path，
            # 直接将 prior_images 中的图像写入数据库。
            # 每个元素为 (image_id, image_path, camera_id)，
            # 保持数据库中的 image_id / camera_id 与先验模型一致。

            for (img_id, img_name, cam_id) in prior_images:
                img_name = img_name.replace('\\', '/')
                if cam_id not in prior_cameras:
                    logger.warning(f"Image {img_name}: prior camera id {cam_id} not in prior cameras, skipping")
                    continue
                try:
                    db.add_image(name=img_name, camera_id=cam_id, image_id=img_id)
                    image_count += 1
                except Exception as e:
                    logger.warning(f"Failed to add image {img_name}: {e}")
            db.commit()

        # =====================================================================
        # MODE: global — one camera for ALL images
        # =====================================================================
        elif camera_assignment == "global":
            # Determine image dimensions from the first image
            first_image = input_images_path[0]
            dims = _get_image_dimensions(first_image, logger)
            if dims is None:
                db.close()
                return False
            width, height = dims
            logger.info(f"Global mode — reference image dimensions: {width}x{height} (from {first_image})")

            cur_prior = False
            params, cur_model_id, cur_model_name = _compute_default_params(
                camera_model_id, camera_model, width, height,
                None, None, _EXPECTED_PARAM_COUNTS, logger
            )

            camera_id = db.add_camera(
                model=cur_model_id, width=width, height=height,
                params=params, prior_focal_length=cur_prior
            )
            logger.info(f"Added global camera: model={cur_model_name}, id={camera_id}")
            db.commit()

            for image_idx, image_file in enumerate(input_images_path):
                img_rel_path = pathlib.Path(os.path.relpath(image_file, images_dir)).as_posix()
                try:
                    db.add_image(name=img_rel_path, camera_id=camera_id)
                    image_count += 1
                except Exception as e:
                    logger.warning(f"Failed to add image {image_file}: {e}")
            db.commit()

        # =====================================================================
        # MODE: per_subfolder — one camera per subfolder (original behaviour)
        # =====================================================================
        elif camera_assignment == "per_subfolder":
            sub_folders = get_subfolders(images_dir)
            if len(sub_folders) == 0:
                logger.info("No subfolders found, treating entire images_dir as one subfolder.")
                sub_folders = [images_dir]

            for sub_folder in sub_folders:
                logger.info(f"Processing subfolder: {sub_folder}")
                images_num, all_files_num, images_path = count_images_in_dir(sub_folder)
                logger.info(f"  Found {images_num} images in this subfolder with {all_files_num} total files.")
                if images_num == 0 or images_num < all_files_num * 0.8:
                    logger.warning(f"  Warning: Only {images_num} images found out of {all_files_num} total files in {sub_folder}. Skipping.")
                    continue

                first_image_path = images_path[0]
                dims = _get_image_dimensions(first_image_path, logger)
                if dims is None:
                    db.close()
                    return False
                width, height = dims
                logger.info(f"Image dimensions: {width}x{height}")

                cur_prior = False
                params, cur_model_id, cur_model_name = _compute_default_params(
                    camera_model_id, camera_model, width, height,
                    None, None, _EXPECTED_PARAM_COUNTS, logger
                )

                camera_id = db.add_camera(
                    model=cur_model_id, width=width, height=height,
                    params=params, prior_focal_length=cur_prior
                )
                logger.info(f"Added camera: model={cur_model_name}, id={camera_id}")
                db.commit()

                for image_idx, image_file in enumerate(images_path):
                    img_rel_path = pathlib.Path(os.path.relpath(image_file, images_dir)).as_posix()
                    try:
                        db.add_image(name=img_rel_path, camera_id=camera_id)
                        image_count += 1
                        if (image_idx + 1) % 100 == 0:
                            logger.info(f"  Added {image_idx + 1}/{len(input_images_path)} images")
                    except Exception as e:
                        logger.warning(f"Failed to add image {image_file}: {e}")
                db.commit()

        # =====================================================================
        # MODE: per_image — each image gets its own camera
        # =====================================================================
        elif camera_assignment == "per_image":
            for image_idx, image_file in enumerate(input_images_path):
                dims = _get_image_dimensions(image_file, logger)
                if dims is None:
                    logger.warning(f"Skipping image due to dimension read failure: {image_file}")
                    continue
                width, height = dims
                image_basename = os.path.basename(image_file)

                cur_prior = False
                params, cur_model_id, cur_model_name = _compute_default_params(
                    camera_model_id, camera_model, width, height,
                    None, None, _EXPECTED_PARAM_COUNTS, logger
                )

                camera_id = db.add_camera(
                    model=cur_model_id, width=width, height=height,
                    params=params, prior_focal_length=cur_prior
                )
                logger.debug(f"Added camera for '{image_basename}': model={cur_model_name}, id={camera_id}")

                img_rel_path = pathlib.Path(os.path.relpath(image_file, images_dir)).as_posix()
                try:
                    db.add_image(name=img_rel_path, camera_id=camera_id)
                    image_count += 1
                    if (image_idx + 1) % 100 == 0:
                        logger.info(f"  Processed {image_idx + 1}/{len(input_images_path)} images")
                except Exception as e:
                    logger.warning(f"Failed to add image {image_file}: {e}")
                db.commit()

        db.close()
        logger.info(f"Successfully initialized database with {image_count} images")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return False


def _parse_prior_camera_file(
    prior_camera_file: str,
    logger: logging.Logger,
    valid_models: set = None,
):
    """
    解析先验相机文件，返回相机配置列表（每个含 camera_id/model/width/height/params，
    若文件带 FOLD 列则还含 fold 字段）。

    支持两种格式（'#' 开头为注释）：
      1) 带 FOLD 列:  CAMERA_ID FOLD MODEL WIDTH HEIGHT PARAMS[]
         例: 1 B SIMPLE_RADIAL 6000 4000 9395.63 ... -0.08156
      2) 无 FOLD 列:  CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]
         例: 1 SIMPLE_RADIAL 6000 4000 9395.63 ... -0.08156
    当 parts[1] 是已知相机模型名时按格式 2 解析，否则按格式 1。

    Args:
        prior_camera_file: 先验相机文件路径
        logger: logger 实例
        valid_models: 合法相机模型名集合，用于自动判断是否带 FOLD 列

    Returns:
        相机配置列表，顺序与文件一致。
    """
    configs = []
    if not os.path.isfile(prior_camera_file):
        logger.error(f"Prior camera file not found: {prior_camera_file}")
        return configs

    with open(prior_camera_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                logger.warning(
                    f"Prior camera file line {line_no} has too few fields, skipped: {line}"
                )
                continue
            try:
                camera_id = int(parts[0])
                # 第二列是合法相机模型名 => 无 FOLD 列；否则视为带 FOLD 列
                has_fold = not (valid_models and parts[1].upper() in valid_models)
                if has_fold:
                    fold, model = parts[1], parts[2]
                    width, height = int(parts[3]), int(parts[4])
                    params = [float(p) for p in parts[5:]]
                else:
                    fold, model = None, parts[1]
                    width, height = int(parts[2]), int(parts[3])
                    params = [float(p) for p in parts[4:]]
            except (ValueError, IndexError) as e:
                logger.warning(
                    f"Prior camera file line {line_no} could not be parsed: {line} ({e})"
                )
                continue
            config = {
                "camera_id": camera_id,
                "model": model,
                "width": width,
                "height": height,
                "params": params,
            }
            if has_fold:
                config["fold"] = fold
            configs.append(config)
    logger.info(f"Parsed {len(configs)} camera config(s) from {prior_camera_file}")
    return configs


def initialize_colmap_database_from_prior_camera_file(
    database_path: str,
    images_dir: str,
    prior_camera_file: str,
    logger: logging.Logger = None
) -> bool:
    """
    根据先验相机文件初始化 COLMAP 数据库，并添加各子文件夹下的图像。

    先验相机文件中的每一行相机与 ``images_dir`` 下的一个子文件夹（FOLD）对应；
    该子文件夹内的所有图像会以相对路径写入数据库，并关联到文件中指定的相机
    （相机 ID 与内参均取自文件，且标记为先验焦距 prior_focal_length=True）。

    先验相机文件支持两种格式（一行一个相机，'#' 开头为注释）：
      1) 带 FOLD 列:  CAMERA_ID FOLD MODEL WIDTH HEIGHT PARAMS[]
         例: 1 B SIMPLE_RADIAL 6000 4000 9395.63 ... -0.08156
      2) 无 FOLD 列:  CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]
         例: 1 SIMPLE_RADIAL 6000 4000 9395.63 ... -0.08156
    带 FOLD 列的相机按 FOLD 名匹配子文件夹；无 FOLD 列的相机按文件顺序
    依次分配给 images_dir 下排序后的子文件夹。

    Args:
        database_path: 要创建/初始化的 COLMAP 数据库路径
        images_dir: 图像根目录（其子文件夹与 FOLD 对应）
        prior_camera_file: 先验相机文件路径
        logger: 可选 logger 实例

    Returns:
        成功返回 True，失败返回 False
    """
    _CAMERA_MODEL_NAME_TO_ID = {
        "SIMPLE_PINHOLE": 0,
        "PINHOLE": 1,
        "SIMPLE_RADIAL": 2,
        "RADIAL": 3,
        "OPENCV": 4,
        "OPENCV_FISHEYE": 5,
        "FULL_OPENCV": 6,
        "FOV": 7,
        "SIMPLE_RADIAL_FISHEYE": 8,
        "RADIAL_FISHEYE": 9,
        "THIN_PRISM_FISHEYE": 10,
        "RAD_TAN_THIN_PRISM_FISHEYE": 11,
    }
    _EXPECTED_PARAM_COUNTS = {
        0: 3,   # SIMPLE_PINHOLE: f, cx, cy
        1: 4,   # PINHOLE: fx, fy, cx, cy
        2: 4,   # SIMPLE_RADIAL: f, cx, cy, k
        3: 5,   # RADIAL: f, cx, cy, k1, k2
        4: 8,   # OPENCV: fx, fy, cx, cy, k1, k2, p1, p2
        5: 8,   # OPENCV_FISHEYE: fx, fy, cx, cy, k1, k2, k3, k4
        6: 12,  # FULL_OPENCV
        7: 5,   # FOV: fx, fy, cx, cy, omega
        8: 4,   # SIMPLE_RADIAL_FISHEYE: f, cx, cy, k1
        9: 5,   # RADIAL_FISHEYE: f, cx, cy, k1, k2
        10: 12, # THIN_PRISM_FISHEYE
        11: 16, # RAD_TAN_THIN_PRISM_FISHEYE
    }

    if logger is None:
        logger = logging.getLogger()

    # 1. 解析先验相机文件（支持带/不带 FOLD 列两种格式）
    camera_config_list = _parse_prior_camera_file(
        prior_camera_file, logger, valid_models=set(_CAMERA_MODEL_NAME_TO_ID)
    )
    if not camera_config_list:
        logger.error("No camera configurations parsed from prior camera file.")
        return False

    # 2. 删除已有数据库，确保使用干净的 schema
    if os.path.exists(database_path):
        logger.info(f"Removing existing database: {database_path}")
        try:
            os.remove(database_path)
        except OSError as exc:
            logger.error(f"Failed to remove existing database: {exc}")
            return False

    try:
        db = COLMAPDatabase.connect(database_path)
        db.create_tables()

        # 3. 确定 images_dir 下的子文件夹（排序保证确定性），
        #    并把每个相机映射到对应子文件夹：
        #    - 带 FOLD 列的相机：按 FOLD 名匹配子文件夹
        #    - 无 FOLD 列的相机：按文件顺序依次分配给排序后尚未占用的子文件夹
        sub_folders = sorted(get_subfolders(images_dir))
        if len(sub_folders) == 0:
            logger.warning("No subfolders found under images_dir; nothing to add.")
            db.close()
            return False
        subfolder_names = [os.path.basename(sf) for sf in sub_folders]

        camera_configs = {}
        used = set()
        for config in camera_config_list:
            fold = config.get("fold")
            if fold is not None:
                if fold not in subfolder_names:
                    logger.warning(
                        f"Camera id={config['camera_id']} fold '{fold}' has no matching "
                        f"subfolder under images_dir, skipping."
                    )
                    continue
                if fold in used:
                    logger.warning(
                        f"Fold '{fold}' is used by multiple cameras, "
                        f"skipping camera id={config['camera_id']}."
                    )
                    continue
                camera_configs[fold] = config
                used.add(fold)

        no_fold_configs = [c for c in camera_config_list if c.get("fold") is None]
        available = [n for n in subfolder_names if n not in used]
        for config, fold in zip(no_fold_configs, available):
            camera_configs[fold] = config
            used.add(fold)
        if len(no_fold_configs) > len(available):
            logger.warning(
                f"{len(no_fold_configs)} cameras have no FOLD but only {len(available)} "
                f"subfolders are available; extra cameras are skipped."
            )

        image_count = 0
        for sub_folder in sub_folders:
            fold_name = os.path.basename(sub_folder)
            if fold_name not in camera_configs:
                logger.warning(
                    f"Subfolder '{fold_name}' has no matching camera in the prior file, skipping."
                )
                continue

            config = camera_configs[fold_name]
            model_id = _CAMERA_MODEL_NAME_TO_ID.get(config["model"].upper())
            if model_id is None:
                logger.warning(
                    f"Unsupported camera model '{config['model']}' for fold '{fold_name}', skipping."
                )
                continue
            if not _validate_camera_params(
                config["model"], model_id, config["params"],
                _EXPECTED_PARAM_COUNTS, logger
            ):
                logger.warning(f"Camera params mismatch for fold '{fold_name}', skipping.")
                continue

            images_num, all_files_num, images_path = count_images_in_dir(sub_folder)
            logger.info(f"Subfolder '{fold_name}': {images_num} images.")
            if images_num == 0:
                logger.warning(f"No images in subfolder '{fold_name}', skipping.")
                continue

            # 添加相机（使用文件中的相机 ID）
            db.add_camera(
                model=model_id,
                width=config["width"],
                height=config["height"],
                params=np.asarray(config["params"], dtype=np.float64),
                prior_focal_length=True,
                camera_id=config["camera_id"],
            )
            logger.info(
                f"Added camera id={config['camera_id']} model={config['model']} "
                f"{config['width']}x{config['height']} for fold '{fold_name}'"
            )
            db.commit()

            # 添加该子文件夹下的图像，并关联到对应相机
            for image_idx, image_file in enumerate(images_path):
                img_rel_path = pathlib.Path(os.path.relpath(image_file, images_dir)).as_posix()
                try:
                    db.add_image(name=img_rel_path, camera_id=config["camera_id"])
                    image_count += 1
                    if (image_idx + 1) % 100 == 0:
                        logger.info(f"  Added {image_idx + 1} images in '{fold_name}'")
                except Exception as e:
                    logger.warning(f"Failed to add image {image_file}: {e}")
            db.commit()

        db.close()
        logger.info(f"Successfully initialized database with {image_count} images")
        return image_count > 0

    except Exception as e:
        logger.error(f"Failed to initialize database from prior camera file: {e}")
        return False


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
            # 没有验证的几何数据，删除并警告
            # logger.warning(f"Pair {name1} - {name2}: No verified geometry data found, removing it")
            db.execute("DELETE FROM matches WHERE pair_id = ?", (pair_id,))
            db.execute("DELETE FROM two_view_geometries WHERE pair_id = ?", (pair_id,))
            removed_count += 1
            continue
        
        try:
            # 解析匹配数据
            init_match = blob_to_array(init_match_data, np.uint32, (-1, 2)) if init_match_data is not None else np.array([])
            verified_match = blob_to_array(verified_match_data, np.uint32, (-1, 2))
            # logger.debug(f"init_match shape: {init_match.shape}, verified_match shape: {verified_match.shape}")
            
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
                reason.append(f"inlier_ratio={inlier_ratio:.4f}<{min_inlier_ratio}")

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
    
    return removed_count


def write_keypoints_to_database(db, image_id: int, keypoints: np.ndarray, descriptors: np.ndarray, logger=None):
    """
    Write SuperPoint keypoints and descriptors to COLMAP database.
    Replaces COLMAP SIFT features with SuperPoint features.
    
    Args:
        db: COLMAPDatabase connection
        image_id: Image ID in database
        keypoints: (N, 2) array of keypoint coordinates
        descriptors: (N, 256) array of feature descriptors (SuperPoint outputs 256-dim)
        logger: Optional logger instance
    """
    if logger is None:
        logger = logging.getLogger()
    
    try:
        # Delete existing features for this image first
        cursor = db.cursor()
        cursor.execute("DELETE FROM keypoints WHERE image_id = ?", (image_id,))
        cursor.execute("DELETE FROM descriptors WHERE image_id = ?", (image_id,))
        db.commit()
        
        # Ensure correct data types
        keypoints = np.asarray(keypoints, np.float32)
        descriptors = np.asarray(descriptors, np.uint8)  # COLMAP expects uint8
        
        # Use database methods to write
        db.add_keypoints(image_id, keypoints)
        db.add_descriptors(image_id, descriptors)
        db.commit()
        
        # logger.debug(f"Wrote {len(keypoints)} keypoints for image_id {image_id}")
        
    except Exception as e:
        logger.warning(f"Failed to write features to database for image {image_id}: {e}")


def batch_write_keypoints_to_database(db, keypoints_list: list, feature_type: int = 0, logger=None):
    """
    Batch write multiple images' keypoints and descriptors to COLMAP database in a single transaction.
    This is much faster than writing keypoints one by one.
    
    Args:
        db: COLMAPDatabase connection
        keypoints_list: List of tuples (image_id, keypoints_array, descriptors_array)
        logger: Optional logger instance
    
    Returns:
        Number of successfully written images
    """
    if logger is None:
        logger = logging.getLogger()
    
    if not keypoints_list:
        return 0
    
    try:
        cursor = db.cursor()
        
        # Begin transaction
        cursor.execute("BEGIN TRANSACTION")
        
        # Delete existing features for all images (batch delete)
        for image_id, _, _ in keypoints_list:
            cursor.execute("DELETE FROM keypoints WHERE image_id = ?", (image_id,))
            cursor.execute("DELETE FROM descriptors WHERE image_id = ?", (image_id,))
        
        # Add all new keypoints and descriptors (use database methods which handle encoding)
        for image_id, keypoints, descriptors in keypoints_list:
            db.add_keypoints(image_id, keypoints)
            db.add_descriptors(image_id, descriptors, feature_type)
        
        # Commit once for all operations
        cursor.execute("COMMIT")
        db.commit()
        
        logger.info(f"Batch wrote {len(keypoints_list)} images' keypoints and descriptors to database")
        return len(keypoints_list)
        
    except Exception as e:
        logger.error(f"Failed to batch write keypoints to database: {e}")
        try:
            db.commit()  # Rollback
        except:
            pass
        return 0


def write_matches_to_database(db, image_id0: int, image_id1: int, matches: np.ndarray, logger=None):
    """
    Write LightGlue matches to COLMAP database.
    
    Args:
        db: COLMAPDatabase connection
        image_id0: First image ID
        image_id1: Second image ID
        matches: (K, 2) array of keypoint indices that match
        logger: Optional logger instance
    """
    if logger is None:
        logger = logging.getLogger()
    
    try:
        # Ensure image_id0 < image_id1 for consistent pair_id encoding
        if image_id0 > image_id1:
            image_id0, image_id1 = image_id1, image_id0
            matches = matches[:, [1, 0]]  # Swap match indices accordingly
        
        # Delete existing matches for this pair first (important!)
        cursor = db.cursor()
        MAX_IMAGE_ID = 2**31 - 1
        pair_id = image_id0 * MAX_IMAGE_ID + image_id1
        cursor.execute("DELETE FROM matches WHERE pair_id = ?", (pair_id,))
        db.commit()
        
        # Ensure correct data type
        matches = np.asarray(matches, np.uint32)
        
        # Use database method to write (handles pair_id encoding automatically)
        db.add_matches(image_id0, image_id1, matches)
        db.commit()
        
        # logger.debug(f"Wrote {len(matches)} matches between images {image_id0} and {image_id1}")
        
    except Exception as e:
        logger.warning(f"Failed to write matches to database for image pair ({image_id0}, {image_id1}): {e}")


def batch_write_matches_to_database(db, matches_list: list, logger=None):
    """
    Batch write multiple match pairs to COLMAP database in a single transaction.
    This is much faster than writing matches one by one.
    
    Args:
        db: COLMAPDatabase connection
        matches_list: List of tuples (image_id0, image_id1, matches_array)
        logger: Optional logger instance
    
    Returns:
        Number of successfully written match pairs
    """
    if logger is None:
        logger = logging.getLogger()
    
    if not matches_list:
        return 0
    
    try:
        cursor = db.cursor()
        MAX_IMAGE_ID = 2**31 - 1
        
        # Pre-process all matches: ensure correct image_id ordering and data type
        processed_matches = []
        for image_id0, image_id1, matches in matches_list:
            if image_id0 > image_id1:
                image_id0, image_id1 = image_id1, image_id0
                matches = matches[:, [1, 0]]  # Swap match indices accordingly
            
            matches = np.asarray(matches, np.uint32)
            processed_matches.append((image_id0, image_id1, matches))
        
        # Begin transaction
        cursor.execute("BEGIN TRANSACTION")
        
        # Delete existing matches for all pairs (batch delete)
        for image_id0, image_id1, _ in processed_matches:
            pair_id = image_id0 * MAX_IMAGE_ID + image_id1
            cursor.execute("DELETE FROM matches WHERE pair_id = ?", (pair_id,))
        
        # Add all new matches (use database method which handles pair_id encoding)
        for image_id0, image_id1, matches in processed_matches:
            db.add_matches(image_id0, image_id1, matches)
        
        # Commit once for all operations
        cursor.execute("COMMIT")
        db.commit()
        
        logger.info(f"Batch wrote {len(processed_matches)} match pairs to database")
        return len(processed_matches)
        
    except Exception as e:
        logger.error(f"Failed to batch write matches to database: {e}")
        try:
            db.commit()  # Rollback
        except:
            pass
        return 0

