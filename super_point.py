"""PyTorch implementation of the SuperPoint model,
   derived from the TensorFlow re-implementation (2018).
   Authors: Rémi Pautrat, Paul-Edouard Sarlin
"""
import os.path
import sys
import time
import cv2
import torch.nn as nn
import torch
import numpy as np
from collections import OrderedDict
from types import SimpleNamespace
from kornia.feature import LightGlue, OnnxLightGlue, LightGlueMatcher

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


def sample_descriptors(keypoints, descriptors, s: int = 8):
    """Interpolate descriptors at keypoint locations"""
    b, c, h, w = descriptors.shape
    keypoints = (keypoints + 0.5) / (keypoints.new_tensor([w, h]) * s)
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode="bilinear", align_corners=False
    )
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1
    )
    return descriptors


def batched_nms(scores, nms_radius: int):
    assert nms_radius >= 0

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius
        )

    zeros = torch.zeros_like(scores)
    max_mask = (scores == max_pool(scores))
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = (supp_scores == max_pool(supp_scores))
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def select_top_k_keypoints(keypoints, scores, k):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0, sorted=True)
    return keypoints[indices], scores


class VGGBlock(nn.Sequential):
    def __init__(self, c_in, c_out, kernel_size, relu=True):
        padding = (kernel_size - 1) // 2
        conv = nn.Conv2d(
            c_in, c_out, kernel_size=kernel_size, stride=1, padding=padding
        )
        activation = nn.ReLU(inplace=True) if relu else nn.Identity()
        bn = nn.BatchNorm2d(c_out, eps=0.001)
        super().__init__(
            OrderedDict(
                [
                    ("conv", conv),
                    ("activation", activation),
                    ("bn", bn),
                ]
            )
        )


class SuperPoint(nn.Module):
    default_conf = {
        "nms_radius": 4,
        "max_num_keypoints": None,
        "detection_threshold": 0.005,
        "remove_borders": 4,
        "descriptor_dim": 256,
        "channels": [64, 64, 128, 128, 256],
    }

    def __init__(self, **conf):
        super().__init__()
        conf = {**self.default_conf, **conf}
        self.conf = SimpleNamespace(**conf)
        self.stride = 2 ** (len(self.conf.channels) - 2)
        channels = [1, *self.conf.channels[:-1]]

        backbone = []
        for i, c in enumerate(channels[1:], 1):
            layers = [VGGBlock(channels[i - 1], c, 3), VGGBlock(c, c, 3)]
            if i < len(channels) - 1:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            backbone.append(nn.Sequential(*layers))
        self.backbone = nn.Sequential(*backbone)

        c = self.conf.channels[-1]
        self.detector = nn.Sequential(
            VGGBlock(channels[-1], c, 3),
            VGGBlock(c, self.stride**2 + 1, 1, relu=False),
        )
        self.descriptor = nn.Sequential(
            VGGBlock(channels[-1], c, 3),
            VGGBlock(c, self.conf.descriptor_dim, 1, relu=False),
        )

        self.track_time : bool = False

    def convert_to_gray(self, image : torch.Tensor) -> torch.Tensor:
        if image.shape[1] == 3:  # RGB to gray
            scale = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            image = (image * scale).sum(1, keepdim=True)
        return image

    def forward(self, image : torch.Tensor) -> dict:
        tic = time.time()
        # print('image: ', image.size())
        features = self.backbone(image)
        # print('feature: ', features.size())
        descriptors_dense = torch.nn.functional.normalize(
            self.descriptor(features), p=2, dim=1
        )
        # print('descriptors_dense: ', descriptors_dense.size())

        # Decode the detection scores
        scores = self.detector(features)
        # print('scores-detector: ', scores.size())
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        # print('scores:-softmax ', scores.size())
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, self.stride, self.stride)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(
            b, h * self.stride, w * self.stride
        )
        scores = batched_nms(scores, self.conf.nms_radius)
        # print('scores:-batched_nms ', scores.size())
        if self.track_time:
            toc = time.time()
            print(f'    [super net] inference: {1000. * (toc - tic) :.2f} ms')

        # Discard keypoints near the image borders
        # tic = time.time()
        if self.conf.remove_borders:
            pad = self.conf.remove_borders
            scores[:, :pad] = -1
            scores[:, :, :pad] = -1
            scores[:, -pad:] = -1
            scores[:, :, -pad:] = -1
        # print('scores:-remove_borders ', scores.size())
        # toc = time.time()
        # print(f'    [super net] remove borders: {1000. * (toc - tic) :.2f} ms')

        # Extract keypoints
        # tic = time.time()
        if b > 1:
            idxs = torch.where(scores > self.conf.detection_threshold)
            mask = idxs[0] == torch.arange(b, device=scores.device)[:, None]
        else:  # Faster shortcut
            scores = scores.squeeze(0)
            idxs = torch.where(scores > self.conf.detection_threshold)

        # print('idxs: ', idxs[0].size())
        # print(idxs[0])
        # print('idxs: ', idxs[1].size())
        # print(idxs[1])
        # print('scores: ', scores.size())
        # toc = time.time()
        # print(f'    [super net] extract keypoints: {1000. * (toc - tic) :.2f} ms')

        # Convert (i, j) to (x, y)
        # tic = time.time()
        keypoints_all = torch.stack(idxs[-2:], dim=-1).flip(1).float()
        scores_all = scores[idxs]
        # print('keypoints_all: ', keypoints_all.size())
        # print('scores_all: ', scores_all.size())
        # toc = time.time()
        # print(f'    [super net] convert: {1000. * (toc - tic) :.2f} ms')

        tic = time.time()
        keypoints = []
        scores = []
        descriptors = []
        for i in range(b):
            if b > 1:
                k = keypoints_all[mask[i]]
                s = scores_all[mask[i]]
            else:
                k = keypoints_all
                s = scores_all
            if self.conf.max_num_keypoints is not None:
                k, s = select_top_k_keypoints(k, s, self.conf.max_num_keypoints)
            d = sample_descriptors(k[None], descriptors_dense[i, None], self.stride)
            keypoints.append(k)
            scores.append(s)
            descriptors.append(d.squeeze(0).transpose(0, 1))
        # toc = time.time()
        # print(f'    [super net] select topk: {1000. * (toc - tic) :.2f} ms')
        if self.track_time:
            toc = time.time()
            # print(f'    [super net] post process: {1000. * (toc - tic) :.2f} ms')

        return {
            "keypoints": keypoints,
            "keypoint_scores": scores,
            "descriptors": descriptors,
        }

def convert_super_points_to_cv(keypoints, keypoint_scores):
    keypoints_cv = []
    for i, (x, y) in enumerate(keypoints):
        kp = cv2.KeyPoint()
        kp.pt = (float(x), float(y))
        kp.size = 1.0
        kp.angle = -1.0
        kp.response = float(keypoint_scores[i])
        kp.octave = 0
        kp.class_id = i
        keypoints_cv.append(kp)
    return keypoints_cv

def brute_force_match(descriptors1 : np.ndarray, descriptors2 : np.ndarray, bf_matcher : cv2.BFMatcher, method = "knn", ratio_thresh= 0.75, k= 2):
    if descriptors1 is None or descriptors2 is None or len(descriptors1) == 0 or len(descriptors2) == 0:
        return []

    if method == "basic":
        matches = bf_matcher.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)

    elif method == "knn":
        knn_matches = bf_matcher.knnMatch(descriptors1, descriptors2, k=k)

        matches = []
        for match_pair in knn_matches:
            if len(match_pair) == k:
                m, n = match_pair
                if m.distance < ratio_thresh * n.distance:
                    matches.append(m)

    elif method == "crosscheck":
        bf_cross = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf_cross.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)

    else:
        raise ValueError(f"Unknown Method: {method}")

    return matches

def create_super_point_model() -> SuperPoint :
    conf = {
        "nms_radius": 4,
        "max_num_keypoints": 1024,
        "detection_threshold": 0.005,
        "remove_borders": 16,
        "descriptor_dim": 256,
        "channels": [64, 64, 128, 128, 256],
    }
    super_point = SuperPoint(**conf)
    super_point = super_point.to(DEVICE)

    ckpt_path = "../pretrained/superpoint_v6_from_tf.pth"
    state_dict = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    super_point.load_state_dict(state_dict)
    super_point.eval()
    return super_point

def create_light_glue_model() -> LightGlue:
    conf = {
        "name": "lightglue",
        "input_dim": 256,  # input descriptor dimension (autoselected from weights)
        "descriptor_dim": 256,
        "add_scale_ori": False,
        "add_laf": False,  # for KeyNetAffNetHardNet
        "scale_coef": 1.0,  # to compensate for the SIFT scale bigger than KeyNet
        "n_layers": 5,
        "num_heads": 4,
        "flash": True,  # enable FlashAttention if available.
        "mp": True,  # enable mixed precision
        "depth_confidence": -1,  # early stopping, disable with -1
        "width_confidence": -1,  # point pruning, disable with -1
        "filter_threshold": 0.01,  # match threshold
        "weights": None,
    }
    matcher = LightGlue(features=None, **conf).to(DEVICE)
    ckpt_path = "../pretrained/superpoint_lightglue_v0-1_arxiv-pth"
    state_dict = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    for i in range(matcher.conf.n_layers):
        pattern = f"self_attn.{i}", f"transformers.{i}.self_attn"
        state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
        pattern = f"cross_attn.{i}", f"transformers.{i}.cross_attn"
        state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
    matcher.load_state_dict(state_dict, strict=False)
    matcher.eval()
    return matcher

def draw_keypoints(image, keypoints, color = (0, 255, 255)):
    if isinstance(keypoints[0], torch.Tensor) or isinstance(keypoints[0], np.ndarray):
        for kp in keypoints:
            center = (int(kp[0]), int(kp[1]))
            cv2.circle(image, center, 2, color, 2, cv2.LINE_4)
    elif isinstance(keypoints[0], cv2.KeyPoint):
        for kp in keypoints:
            center = (int(kp.pt[0]), int(kp.pt[1]))
            cv2.circle(image, center, 2, color, 2, cv2.LINE_4)

def draw_matches(img1_bgr, keypoints1, img2_bgr, keypoints2, matches, num_matches):
    h, w = img1_bgr.shape[:2]
    output_image = cv2.hconcat([img1_bgr, img2_bgr])
    draw_every_n = 1 if num_matches < 100 else 5

    for k in range(num_matches):
        if k % draw_every_n != 0:
            continue
        idx1, idx2 = None, None
        if isinstance(matches, list):
            match : cv2.DMatch = matches[k]
            idx1, idx2 = match.queryIdx, match.trainIdx
        elif isinstance(matches, torch.Tensor):
            idx1, idx2 = matches[k, ...]
        else:
            print("Non-supported match structure!!!")
            continue

        kp1 = keypoints1[idx1, ...]
        kp2 = keypoints2[idx2, ...]
        center1 = (int(kp1[0]), int(kp1[1]))
        center2 = (int(kp2[0]) + w, int(kp2[1]))

        cv2.circle(output_image, center1, 3, (0, 0, 255), 3, cv2.LINE_4)
        cv2.circle(output_image, center2, 3, (0, 0, 255), 3, cv2.LINE_4)
        cv2.line(output_image, center1, center2, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(output_image, f"Match: {num_matches}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 0, 0),
                2)

    return output_image

@torch.no_grad()
def main():
    super_point = create_super_point_model()
    orb = cv2.ORB.create(super_point.conf.max_num_keypoints, 1.1, 8)

    save_dir = "../outputs"
    # root_dir = 'D:/dataset/DroneSwarm/heyu20240718/24-07-19_12-49-51_1/'
    # root_dir = 'Z:/wkspace/proj/dsonlinemapping/data/huizhou20251107/25-09-19_19-00-42_20/'
    root_dir = 'Z:/wkspace/proj/dsonlinemapping/data/chibishixunjidiM3E/DJI_202512051045_022_70/'
    filepaths = sorted(os.listdir(root_dir))
    num_images = len(filepaths)

    timer = cv2.TickMeter()

    count = 0
    for i, filepath in enumerate(filepaths):
        # if i < 500:
        #     continue
        # if i % 2 == 0:
        #     continue

        filename = os.path.join(root_dir, filepath)
        img_bgr = cv2.imread(filename, cv2.IMREAD_COLOR)
        img_bgr = cv2.resize(img_bgr, (960, 540), None, 0, 0, cv2.INTER_LINEAR)

        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        print(f'Process: {i + 1}/{num_images}: {filepath}, reso: {img.shape[1]} x {img.shape[0]}')

        timer.start()
        img_tensor = torch.from_numpy(img).to(DEVICE).unsqueeze(0).unsqueeze(0).float().div_(255.)
        h2 = (img.shape[0] // 8) * 8
        w2 = (img.shape[1] // 8) * 8
        img_tensor = img_tensor[:, :, :h2, :w2]

        output = super_point(img_tensor)
        timer.stop()
        count += 1
        print(f"timing: {timer.getAvgTimeMilli():.2f} ms")
        # break

        keypoints = output['keypoints'][0].squeeze().cpu().numpy()
        descriptors = output['descriptors'][0].squeeze().cpu().numpy()
        keypoint_scores = output['keypoint_scores'][0].squeeze().cpu().numpy()

        keypoints_orb = orb.detect(img)
        _, descriptors_orb = orb.compute(img, keypoints_orb)

        img_bgr_copy = img_bgr.copy()
        color = (0, 255, 255)
        draw_keypoints(img_bgr, keypoints, color)
        draw_keypoints(img_bgr_copy, keypoints_orb, color)

        img_bgr = cv2.hconcat([img_bgr, img_bgr_copy])
        cv2.imshow('image', img_bgr)
        cv2.waitKey(30)

        filename_save = os.path.join(save_dir, filepath)
        cv2.imwrite(filename_save, img_bgr)

@torch.no_grad()
def light_glue_match():
    torch.cuda.empty_cache()

    super_point = create_super_point_model()
    light_glue = create_light_glue_model()

    orb = cv2.ORB.create(super_point.conf.max_num_keypoints, 1.1, 8)
    bf_matcher_L2 = cv2.BFMatcher.create(cv2.NORM_L2, crossCheck=False)
    bf_matcher_Ham = cv2.BFMatcher.create(cv2.NORM_HAMMING, crossCheck=False)

    save_dir = "../outputs"
    # root_dir = 'Z:/wkspace/proj/dsonlinemapping/data/chibishixunjidiM3E/DJI_202512051045_022_70/'
    # root_dir = 'D:/dataset/DroneSwarm/heyu20240718/24-07-19_12-49-51_1/'
    # root_dir = 'Z:/wkspace/proj/dsonlinemapping/data/huizhou20251107/25-09-19_19-00-42_20/'
    root_dir = '../examples/match/input'
    # root_dir = '../examples/match/data64/input'
    filepaths = sorted(os.listdir(root_dir))
    num_images = len(filepaths)

    timer_sp = cv2.TickMeter()
    timer_lg = cv2.TickMeter()

    # work_size = (320, 240)
    # work_size = (640, 480)
    # work_size = (699, 382)
    work_size = (960, 540)
    # work_size = (528, 395)
    # work_size = (1920, 1080)

    cv2.namedWindow("matches", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("matches", (work_size[0] * 2, work_size[1]))

    di : int = 1
    df : int = 1
    start = 0 # max(0, num_images - 200)
    end = num_images - max(di + 1, df + 1)
    for i in range(start, end, di):
        # if i != 11:
        #     continue
        filename1 = os.path.join(root_dir, filepaths[i])
        filename2 = os.path.join(root_dir, filepaths[i + df])
        # if i == 0:
        #     filename1 = "D:\\Projects\\pyprojects\\LightGlue\\assets\\DSC_0410.jpg"
        #     filename2 = "D:\\Projects\\pyprojects\\LightGlue\\assets\\DSC_0411.jpg"
        img1_bgr = cv2.imread(filename1, cv2.IMREAD_COLOR)
        img2_bgr = cv2.imread(filename2, cv2.IMREAD_COLOR)
        img2_bgr = cv2.flip(img2_bgr, 0)
        img2_bgr = cv2.flip(img2_bgr, 1)
        if img1_bgr is None:
            raise Exception("read image1 failed")
        if img2_bgr is None:
            raise Exception("read image2 failed")
        img1_bgr = cv2.resize(img1_bgr, work_size, None, 0, 0, cv2.INTER_LINEAR)
        img2_bgr = cv2.resize(img2_bgr, work_size, None, 0, 0, cv2.INTER_LINEAR)
        img1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY)

        h, w = img1.shape[:2]
        print(f"\nProcess pair: ({i}, {i+1}) / {num_images}, reso: {w} x {h}")

        img_tensor1 = torch.from_numpy(img1).to(DEVICE).unsqueeze(0).unsqueeze(0).float().div_(255.)
        img_tensor2 = torch.from_numpy(img2).to(DEVICE).unsqueeze(0).unsqueeze(0).float().div_(255.)
        assert img1.shape == img2.shape, "input image shape must be equal!"

        h2 = (img1.shape[0] // 8) * 8
        w2 = (img1.shape[1] // 8) * 8
        img_tensor1 = img_tensor1[:, :, :h2, :w2]
        img_tensor2 = img_tensor2[:, :, :h2, :w2]

        timer_sp.start()
        features1: dict = super_point(img_tensor1)
        features2: dict = super_point(img_tensor2)
        timer_sp.stop()
        print(f"Timing [Extract Feature]: {timer_sp.getAvgTimeMilli() :.2f} ms")
        keypoints1 = features1['keypoints'][0]
        descriptors1 = features1['descriptors'][0]
        keypoint_scores1 = features1['keypoint_scores'][0]
        keypoints2 = features2['keypoints'][0]
        descriptors2 = features2['descriptors'][0]
        keypoint_scores2 = features2['keypoint_scores'][0]

        input_dict = {
            "image0": {
                "keypoints": keypoints1.unsqueeze(0),
                "descriptors": descriptors1.unsqueeze(0),
                "image_size": torch.Size((w2, h2)),
            },
            "image1": {
                "keypoints": keypoints2.unsqueeze(0),
                "descriptors": descriptors2.unsqueeze(0),
                "image_size": torch.Size((w2, h2)),
            }
        }

        timer_lg.start()
        output = light_glue(input_dict)
        timer_lg.stop()
        print(f"Timing [Match]: {timer_lg.getAvgTimeMilli():.2f} ms")

        # print(output.keys())
        matches = output["matches"][0]
        num_matches = matches.shape[0]
        print(f"\tNumber Matches: {num_matches}")

        draw_keypoints(img1_bgr, keypoints1, (0, 255, 0))
        draw_keypoints(img2_bgr, keypoints2, (0, 255, 0))
        output_image = draw_matches(img1_bgr, keypoints1, img2_bgr, keypoints2, matches, num_matches)
        cv2.imshow("matches", output_image)
        # cv2.waitKey(0)

        keypoints_orb1 = orb.detect(img1)
        keypoints_orb2 = orb.detect(img2)
        _, descriptors_orb1 = orb.compute(img1, keypoints_orb1)
        _, descriptors_orb2 = orb.compute(img2, keypoints_orb2)
        matches_bf = brute_force_match(descriptors1.squeeze().cpu().numpy(), descriptors2.squeeze().cpu().numpy(), bf_matcher_L2, "knn", ratio_thresh=0.95, k = 2)
        # matches_bf = brute_force_match(descriptors_orb1, descriptors_orb2, bf_matcher_Ham, "knn", ratio_thresh=0.85, k = 2)
        output_image2 = draw_matches(img1_bgr, keypoints1, img2_bgr, keypoints2, matches_bf, len(matches_bf))
        cv2.imshow("matches BF", output_image2)
        cv2.waitKey(0)


        filename_save = os.path.join(save_dir, f"Frame{i}_{i + df}.jpg")
        cv2.imwrite(filename_save, output_image)


if __name__ == "__main__":
    # main()
    light_glue_match()
