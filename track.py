import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np


def locate(reference_points, descriptors, frame, sift, matcher, corners):
    keypoints, current = sift.detectAndCompute(frame, None)
    if current is None or len(current) < 2:
        return None, 0, 0
    pairs = matcher.knnMatch(descriptors, current, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < 4:
        return None, len(good), 0
    source = np.float32([reference_points[m.queryIdx].pt for m in good])
    target = np.float32([keypoints[m.trainIdx].pt for m in good])
    homography, mask = cv2.findHomography(source, target, cv2.RANSAC, 3.0)
    inliers = int(mask.sum()) if mask is not None else 0
    if homography is None or inliers < 6 or inliers / len(good) < 0.25:
        return None, len(good), inliers
    polygon = cv2.perspectiveTransform(corners, homography)
    height, width = frame.shape
    area = abs(cv2.contourArea(polygon))
    if (not np.isfinite(polygon).all() or not cv2.isContourConvex(polygon)
            or area < 100 or area > 4 * width * height
            or np.abs(polygon).max() > 10 * max(width, height)):
        return None, len(good), inliers
    return polygon, len(good), inliers


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
    gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    mask = np.zeros_like(gray)
    mask[y:y+h, x:x+w] = 255
    keypoints, descriptors = sift.detectAndCompute(gray, mask)
    if descriptors is None or len(keypoints) < 6:
        raise ValueError('В эталоне недостаточно ключевых точек')
    corners = np.float32([[x, y], [x+w-1, y], [x+w-1, y+h-1], [x, y+h-1]]).reshape(-1, 1, 2)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
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
            polygon, matches, inliers = locate(keypoints, descriptors, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), sift, matcher, corners)
            found = polygon is not None
            if found:
                cv2.polylines(frame, [np.int32(np.rint(polygon))], True, (0, 255, 0), 2, cv2.LINE_AA)
                px, py = np.rint(polygon[0, 0]).astype(int)
                cv2.putText(frame, args.label, (int(np.clip(px, 0, width-120)), int(np.clip(py-8, 20, height-10))), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f'{len(rows)}: {"FOUND" if found else "LOST"} {inliers}/{matches}', (10, height-12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
            writer.write(frame)
            rows.append((len(rows), matches, inliers, int(found)))
            ok, frame = capture.read()
            if not ok:
                frame = None
    finally:
        capture.release()
        writer.release()
    elapsed = perf_counter() - started
    with Path(str(base) + '.csv').open('w', newline='', encoding='utf-8') as file:
        csv.writer(file).writerows([('frame', 'matches', 'inliers', 'found'), *rows])
    summary = dict(video=args.video.name, frames=len(rows), reference_keypoints=len(keypoints), roi=[x,y,w,h],
                   found=sum(r[3] for r in rows), mean_matches=float(np.mean([r[1] for r in rows])),
                   mean_inliers=float(np.mean([r[2] for r in rows])), seconds=elapsed, processing_fps=len(rows)/elapsed,
                   opencv=cv2.__version__)
    Path(str(base) + '.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Трекинг плоского объекта по SIFT и гомографии')
    parser.add_argument('video', type=Path)
    parser.add_argument('--roi', type=int, nargs=4, metavar=('X', 'Y', 'W', 'H'))
    parser.add_argument('--label', default='Object')
    parser.add_argument('--output', type=Path, default=Path('results'))
    run(parser.parse_args())
