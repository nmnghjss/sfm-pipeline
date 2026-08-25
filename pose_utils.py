import numpy as np

# ==============================================================================
#  工具函数
# ==============================================================================

def quat_to_rot(q):
    """四元数 (qw, qx, qy, qz) → 3×3 旋转矩阵"""
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,         1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,         2*qy*qz + 2*qx*qw,     1 - 2*qx*qx - 2*qy*qy],
    ])


def rot_to_quat(R):
    """3×3 旋转矩阵 → 四元数 (qw, qx, qy, qz)"""
    R = np.asarray(R, dtype=np.float64)
    q = np.zeros(4, dtype=np.float64)
    trace = np.trace(R)

    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        q[0] = 0.25 * s
        q[1] = (R[2, 1] - R[1, 2]) / s
        q[2] = (R[0, 2] - R[2, 0]) / s
        q[3] = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q[0] = (R[2, 1] - R[1, 2]) / s
        q[1] = 0.25 * s
        q[2] = (R[0, 1] + R[1, 0]) / s
        q[3] = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q[0] = (R[0, 2] - R[2, 0]) / s
        q[1] = (R[0, 1] + R[1, 0]) / s
        q[2] = 0.25 * s
        q[3] = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q[0] = (R[1, 0] - R[0, 1]) / s
        q[1] = (R[0, 2] + R[2, 0]) / s
        q[2] = (R[1, 2] + R[2, 1]) / s
        q[3] = 0.25 * s

    q /= np.linalg.norm(q)
    return q[0], q[1], q[2], q[3]


def rotation_from_two_vectors(src, dst):
    """计算将单位向量 src 旋转到 dst 的 3×3 旋转矩阵（Rodrigues）"""
    src = src / np.linalg.norm(src)
    dst = dst / np.linalg.norm(dst)

    if np.allclose(src, dst):
        return np.eye(3)
    if np.allclose(src, -dst):
        # 180° 旋转：绕任意垂直轴
        perp = np.array([1.0, 0.0, 0.0])
        if np.abs(np.dot(perp, src)) > 0.9:
            perp = np.array([0.0, 1.0, 0.0])
        axis = np.cross(src, perp)
        axis /= np.linalg.norm(axis)
        cos_a, sin_a = -1.0, 0.0
    else:
        axis = np.cross(src, dst)
        axis /= np.linalg.norm(axis)
        cos_a = np.dot(src, dst)
        sin_a = np.linalg.norm(np.cross(src, dst))

    K = np.array([
        [0,        -axis[2],  axis[1]],
        [axis[2],   0,       -axis[0]],
        [-axis[1],  axis[0],  0],
    ])
    return np.eye(3) + sin_a * K + (1 - cos_a) * (K @ K)
