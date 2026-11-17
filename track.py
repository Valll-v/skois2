import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np


def project(points, homography):
    return cv2.perspectiveTransform(np.float32(points).reshape(-1, 1, 2), homography)


def valid_polygon(polygon, shape, previous=None):
    height, width = shape
    if not np.isfinite(polygon).all() or not cv2.isContourConvex(polygon):
        return False
    area = cv2.contourArea(polygon, oriented=True)
    if area < 100 or area > 4 * width * height or np.abs(polygon).max() > 4 * max(width, height):
        return False
    edges = np.linalg.norm(polygon[:, 0] - np.roll(polygon[:, 0], 1, axis=0), axis=1)
    if edges.min() < 6 or edges.max() / edges.min() > 12:
        return False
    if previous is not None:
        ratio = area / cv2.contourArea(previous)
        movement = np.linalg.norm(polygon - previous, axis=2).max()
        if not 0.65 < ratio < 1.5 or movement > max(25, np.sqrt(area) * 0.25):
            return False
    return True


def estimate(source, target, corners, shape, threshold=2.0):
    if len(source) < 8:
        return None, 0
    homography, mask = cv2.findHomography(source, target, cv2.RANSAC, threshold, maxIters=3000, confidence=0.995)
    count = int(mask.sum()) if mask is not None else 0
    if homography is None or count < 8 or count < 0.4 * len(source):
        return None, count
    polygon = project(corners, homography)
    selected = np.float32(target)[mask.ravel() != 0].reshape(-1, 1, 2)
    coverage = cv2.contourArea(cv2.convexHull(selected)) / max(cv2.contourArea(polygon), 1)
    if coverage < 0.12 or not valid_polygon(polygon, shape):
        return None, count
    return homography, count


class Tracker:
    def __init__(self, gray, roi):
        x, y, w, h = roi
        self.corners = np.float32([[x, y], [x+w-1, y], [x+w-1, y+h-1], [x, y+h-1]]).reshape(-1, 1, 2)
        self.sift = cv2.SIFT_create(contrastThreshold=0.02)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        mask = np.zeros_like(gray)
        mask[y:y+h, x:x+w] = 255
        points, descriptors = [], []
        for scale in (1.0, 0.75, 0.5, 0.35, 0.25):
            size = (round(gray.shape[1] * scale), round(gray.shape[0] * scale))
            small = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
            small_mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
            keys, desc = self.sift.detectAndCompute(small, small_mask)
            if desc is not None:
                # Координаты всех масштабов храним в системе первого кадра.
                factor = np.float32([gray.shape[1] / size[0], gray.shape[0] / size[1]])
                points.extend([np.float32(k.pt) * factor for k in keys])
                descriptors.append(desc)
        if not descriptors or len(points) < 8:
            raise ValueError('В эталоне недостаточно ключевых точек')
        self.reference_points = np.float32(points)
        self.descriptors = np.vstack(descriptors)
        self.previous = gray.copy()
        self.homography = np.eye(3)
        self.polygon = self.corners.copy()
        self.current_points = None
        self.reseed(gray)

    def reseed(self, gray):
        mask = np.zeros_like(gray)
        cv2.fillConvexPoly(mask, np.int32(np.rint(self.polygon)), 255)
        # Не брать фон за границей картины в окно оптического потока.
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
        points = cv2.goodFeaturesToTrack(gray, 350, 0.005, 2, mask=mask, blockSize=3)
        self.current_points = points

    def detect(self, gray):
        keys, descriptors = self.sift.detectAndCompute(gray, None)
        if descriptors is None or len(descriptors) < 2:
            return None, 0, 0
        pairs = self.matcher.knnMatch(self.descriptors, descriptors, k=2)
        good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
        # Один признак кадра не должен давать несколько голосов от разных масштабов.
        unique = {}
        for match in sorted(good, key=lambda item: item.distance):
            unique.setdefault(match.trainIdx, match)
        good = list(unique.values())
        source = np.float32([self.reference_points[m.queryIdx] for m in good])
        target = np.float32([keys[m.trainIdx].pt for m in good])
        homography, inliers = estimate(source, target, self.corners, gray.shape)
        return homography, len(good), inliers

    def flow(self, gray):
        if self.current_points is None or len(self.current_points) < 6:
            return None, 0, 0
        options = dict(winSize=(15, 15), maxLevel=3,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        forward, status, error = cv2.calcOpticalFlowPyrLK(self.previous, gray, self.current_points, None, **options)
        if forward is None:
            return None, 0, 0
        backward, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous, forward, None, **options)
        if backward is None:
            return None, 0, 0
        consistent = np.linalg.norm(backward - self.current_points, axis=2).ravel() < 0.8
        keep = (status.ravel() != 0) & (back_status.ravel() != 0) & consistent & (error.ravel() < 25)
        source = self.current_points[keep]
        target = forward[keep]
        # При малом числе точек используем аффинную модель вместо полной H.
        homography, inliers = None, 0
        if len(source) >= 30:
            step, inliers = estimate(source, target, self.polygon, gray.shape, 1.0)
            if step is not None:
                homography = step @ self.homography
        if len(source) >= 6:
            affine, mask = cv2.estimateAffine2D(source, target, method=cv2.RANSAC, ransacReprojThreshold=1.0)
            affine_inliers = int(mask.sum()) if mask is not None else 0
            if homography is None and affine is not None and affine_inliers >= 6 and affine_inliers >= 0.6 * len(source):
                homography = np.vstack([affine, [0, 0, 1]]) @ self.homography
                inliers = affine_inliers
        if homography is not None and not valid_polygon(project(self.corners, homography), gray.shape, self.polygon):
            homography = None
        return homography, int(keep.sum()), inliers

    def update(self, gray, index):
        if index == 0:
            return self.polygon.copy(), 'INIT', len(self.current_points) if self.current_points is not None else 0, 0, 0
        flow_h, tracks, _ = self.flow(gray) if self.homography is not None else (None, 0, 0)
        detection, matches, sift_inliers = None, 0, 0
        if index % 5 == 0 or flow_h is None:
            detection, matches, sift_inliers = self.detect(gray)
        chosen = flow_h
        mode = 'FLOW' if flow_h is not None else 'LOST'
        if detection is not None:
            detected = project(self.corners, detection)
            if flow_h is None:
                chosen, mode = detection, 'SIFT'
            else:
                flowing = project(self.corners, flow_h)
                difference = np.linalg.norm(detected - flowing, axis=2).max()
                agrees = difference < max(6, np.sqrt(cv2.contourArea(flowing)) * 0.08)
                strong = sift_inliers >= 15 and sift_inliers >= 0.6 * matches
                if agrees or strong:
                    weight = 0.4 if strong else 0.2
                    corrected = np.float32((1 - weight) * flowing + weight * detected)
                    if valid_polygon(corrected, gray.shape, self.polygon):
                        chosen = cv2.getPerspectiveTransform(self.corners, corrected)
                        mode = 'FLOW+SIFT'
        self.homography = chosen
        if chosen is not None:
            self.polygon = project(self.corners, chosen)
            self.reseed(gray)
        else:
            self.current_points = None
        self.previous = gray.copy()
        return self.polygon.copy() if chosen is not None else None, mode, tracks, matches, sift_inliers


def run(args):
    cv2.setRNGSeed(0)
    cv2.setNumThreads(1)
    capture = cv2.VideoCapture(str(args.video))
    ok, first = capture.read()
    if not ok:
        raise ValueError(f'Не удалось прочитать видео: {args.video}')
    height, width = first.shape[:2]
    x, y, w, h = args.roi or (0, 0, width, height)
    if x < 0 or y < 0 or w < 2 or h < 2 or x + w > width or y + h > height:
        raise ValueError('ROI должен находиться внутри первого кадра')
    tracker = Tracker(cv2.cvtColor(first, cv2.COLOR_BGR2GRAY), (x, y, w, h))
    args.output.mkdir(parents=True, exist_ok=True)
    base = args.output / args.video.stem
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    writer = cv2.VideoWriter(str(base) + '_out.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))
    if not writer.isOpened():
        raise ValueError('Не удалось открыть VideoWriter')
    rows = []
    frame = first
    started = perf_counter()
    try:
        while frame is not None:
            polygon, mode, tracks, matches, inliers = tracker.update(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), len(rows))
            found = polygon is not None
            coordinates = polygon.reshape(-1).tolist() if found else [float('nan')] * 8
            if found:
                cv2.polylines(frame, [np.int32(np.rint(polygon))], True, (0, 255, 0), 2, cv2.LINE_AA)
                px, py = np.rint(polygon[0, 0]).astype(int)
                cv2.putText(frame, args.label, (int(np.clip(px, 0, width-120)), int(np.clip(py-8, 20, height-10))), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f'{len(rows)}: {mode} LK:{tracks} SIFT:{inliers}/{matches}', (10, height-12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
            writer.write(frame)
            rows.append((len(rows), matches, inliers, int(found), mode, tracks, *coordinates))
            ok, frame = capture.read()
            if not ok:
                frame = None
    finally:
        capture.release()
        writer.release()
    elapsed = perf_counter() - started
    with Path(str(base) + '.csv').open('w', newline='', encoding='utf-8') as file:
        csv.writer(file).writerows([('frame', 'matches', 'inliers', 'found', 'mode', 'tracks', 'x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3'), *rows])
    detection_rows = [r for r in rows if r[1] > 0]
    summary = dict(video=args.video.name, frames=len(rows), reference_keypoints=len(tracker.reference_points), roi=[x,y,w,h],
                   found=sum(r[3] for r in rows), sift_frames_with_matches=sum(r[1] > 0 for r in rows),
                   mean_matches=float(np.mean([r[1] for r in detection_rows])) if detection_rows else 0,
                   mean_inliers=float(np.mean([r[2] for r in detection_rows])) if detection_rows else 0,
                   seconds=elapsed, processing_fps=len(rows)/elapsed, opencv=cv2.__version__)
    Path(str(base) + '.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Трекинг плоского объекта: SIFT и оптический поток')
    parser.add_argument('video', type=Path)
    parser.add_argument('--roi', type=int, nargs=4, metavar=('X', 'Y', 'W', 'H'))
    parser.add_argument('--label', default='Object')
    parser.add_argument('--output', type=Path, default=Path('results'))
    run(parser.parse_args())
