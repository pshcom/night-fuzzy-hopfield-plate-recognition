"""
infer_preprocessing.py
번호판 전처리 핵심 함수 모음 (batch_preprocess.py, run.py 공용)
YOLOv8 기반 번호판 검출 + MAX-MIN 퍼지 스트레칭 및 이진화

[수정 내역]
1. 테두리 크롭 비율: 12% → 15% (테두리 노이즈 차단)
2. 이진화: 배경 밝기 감지 기반 파라미터 자동 선택 (밝은 번호판: 11/5, 어두운/야간: 21/7)
3. min_w 기준 완화: w_img*0.04 → w_img*0.025 (얇은 '1' 누락 방지)
4. 종횡비 필터: h/w < 4.0 (테두리 조각 제거)
5. 상단 테두리 노이즈 필터: y=0 이고 h >= h_img*0.9 인 박스 제거
6. 우측 테두리 조각 필터: x+w >= w_img*0.97 이고 w <= w_img*0.15 인 박스 제거 (0.08→0.15 강화)
7. [신규] 이진화 후 연결 컴포넌트 라벨링으로 노이즈 제거 (논문 방식)
   - 전체너비(w>70%) 덩어리, 가로줄(w/h>8), 작은 점(면적<0.3%) 제거
8. [신규] DI 박스필터 임계값 조정: < 10 → < 6
9. [신규] 한글 위치 고정 zone 방식 (gap_fill 제거) — core v5c 포팅
   - 한글은 240px 기준 x≈55~118(2/7~3/7) 고정 zone
   - 연결박스 경계 강제분할: _split_at_hangul_boundary (양방향)
   - hangul_x0 경계선 위 박스는 숫자로 분류
   - 한글 조각(ㅅ+ㅜ 등) bounding box 합침
   - 중복 digit 제거 (한글박스 범위 내 겹치는 digit 제거)
   - 왼쪽 숫자 suspicious 보호: x<8%, w<=4% 얇은 박스는 삭제 금지 ('1' 누락 방지)
   - zone fallback h 기준 완화: 0.20 → 0.10 (ㅗ 등 작은 조각 포함)
10. [신규] fullwidth 컴포넌트 column-wise 세로획 복원 — core 포팅
    - 경계 열(x<5%, x>90%) 제외하고 세로획(h>=30%)만 복원
11. [신규] _is_border_artifact 필터 추가 — core 포팅
    - deskew 후 테두리 이물질 제거 (left_thin, right_thin, left_tall_wide)
    - x=0 에 붙은 w<=5px 는 '1'로 보호
"""
import cv2
import numpy as np
from ultralytics import YOLO
import os as _os
_SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))

PLATE_W, PLATE_H  = 240, 60
TARGET_CHAR_COUNT = 7
CHAR_SIZE         = 32

model = YOLO(_os.path.join(_SCRIPT_DIR, 'license_plate_detector.pt'))

# =============================================
# 1단계: 번호판 검출 (YOLOv8 인식모델 가중치 기반)
# =============================================
def detect_plate_night(img, gray):
    results = model(img, verbose=False)

    for result in results:
        boxes = result.boxes
        if len(boxes) > 0:
            x, y, w, h = boxes[0].xywh[0].tolist()
            x, y, w, h = int(x - w/2), int(y - h/2), int(w), int(h)

            H, W = gray.shape
            x = max(0, x)
            y = max(0, y)
            w = min(w, W - x)
            h = min(h, H - y)
            if w <= 0 or h <= 0:
                return None, None

            plate_crop = gray[y:y+h, x:x+w]
            return plate_crop, (x, y, w, h)
    return None, None


# =============================================
# [추가] 번호판 기울기 보정 (deskew)
# =============================================
def deskew_plate(plate_crop):
    edges = cv2.Canny(plate_crop, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                             threshold=40, minLineLength=plate_crop.shape[1] // 4,
                             maxLineGap=10)
    if lines is None:
        return plate_crop

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < 10:
            angles.append(angle)

    if not angles:
        return plate_crop

    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.5:
        return plate_crop

    h, w = plate_crop.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    corrected = cv2.warpAffine(plate_crop, M, (w, h),
                                flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_REPLICATE)
    return corrected


# =============================================
# 2단계: 퍼지 스트레칭 (MAX-MIN 대비 극대화)
# =============================================
def fuzzy_membership(x, a, b, c, d):
    """
    사다리꼴 소속도 함수
    a~b: 올라가는 구간 (0→1)
    b~c: 완전 소속 구간 (1)
    c~d: 내려가는 구간 (1→0)
    """
    mu = np.zeros_like(x, dtype=np.float32)
    # 올라가는 구간
    rising = (x >= a) & (x < b)
    mu[rising] = (x[rising] - a) / (b - a + 1e-6)
    # 완전 소속 구간
    mu[(x >= b) & (x <= c)] = 1.0
    # 내려가는 구간
    falling = (x > c) & (x <= d)
    mu[falling] = (d - x[falling]) / (d - c + 1e-6)
    return mu

def fuzzy_stretch(plate_gray):
    x = plate_gray.astype(np.float32)

    # 각 클래스 사다리꼴 구간 (a, b, c, d)
    # Dark:   0~10 올라가고, 10~60 완전소속, 60~110 내려감
    # Mid:   60~80 올라가고, 80~160 완전소속, 160~180 내려감
    # Bright: 130~160 올라가고, 160~230 완전소속, 230~245 내려감
    mu_dark   = fuzzy_membership(x,   0,  10,  60, 110)
    mu_mid    = fuzzy_membership(x,  60,  80, 160, 180)
    mu_bright = fuzzy_membership(x, 130, 160, 230, 245)

    # 각 클래스별 min-max 스트레칭
    def stretch(img, mn, mx):
        return np.clip((img - mn) / (mx - mn + 1e-6) * 255, 0, 255)

    s_dark   = stretch(x,  10, 110)
    s_mid    = stretch(x,  60, 180)
    s_bright = stretch(x, 130, 245)

    # 소속도 가중 블렌딩
    weight_sum = mu_dark + mu_mid + mu_bright + 1e-6
    blended = (mu_dark * s_dark + mu_mid * s_mid + mu_bright * s_bright) / weight_sum

    return np.clip(blended, 0, 255).astype(np.uint8)


# =============================================
# [추가] CLAHE 빛번짐 대응
# =============================================
def enhance_plate(plate_gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    return clahe.apply(plate_gray)


# =============================================
# 3단계: 가우시안 + 이진화 + 모폴로지
# =============================================
def binarize_and_morphology(plate_resized):
    blurred = cv2.GaussianBlur(plate_resized, (5, 5), 0)

    h, w = blurred.shape
    cy0, cy1 = h // 4, h * 3 // 4
    cx0, cx1 = w // 4, w * 3 // 4
    center_median = float(np.median(blurred[cy0:cy1, cx0:cx1]))

    if center_median > 210:
        # 과노출(헤드라이트 직접 반사): 로컬 편차 작음 → 블록 작게, C 낮게
        block_size = 15
        c_val = 5
    elif center_median > 180:
        block_size = 19   # 밝은 번호판: 넓은 블록
        c_val = 10        # C 높여서 배경 억제, 글자 간 분리
    elif center_median > 150:
        block_size = 19
        c_val = 8
    elif center_median > 120:
        # 중간 어두움: 블록 약간 크게, c_val 중간으로 (노이즈 억제 + 획 살림 균형)
        block_size = 17
        c_val = 8
    else:
        # 극한 야간(median<120): 블록 작게, c_val 낮게(글자 픽셀 보존 우선)
        block_size = 15
        c_val = 4

    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=block_size,
        C=c_val
    )

    if center_median <= 120:
        # 극한 야간: 획이 끊어짐 → 더 큰 CLOSE로 먼저 연결
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 3))
        morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel_small)
        kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 3))
        final = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_large)
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel_small)
        kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        final = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_large)

    return final


# C값 낮춘 fallback 이진화 (획 보존 우선, 세그 7개 미달 시 재시도용)
def _binarize_low_c(plate_resized):
    blurred = cv2.GaussianBlur(plate_resized, (5, 5), 0)
    h, w = blurred.shape
    cy0, cy1 = h // 4, h * 3 // 4
    cx0, cx1 = w // 4, w * 3 // 4
    center_median = float(np.median(blurred[cy0:cy1, cx0:cx1]))

    if center_median > 210:   block_size, c_val = 15, 2
    elif center_median > 180: block_size, c_val = 19, 5
    elif center_median > 150: block_size, c_val = 19, 4
    elif center_median > 120: block_size, c_val = 17, 4
    else:                     block_size, c_val = 15, 2

    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=block_size, C=c_val)

    if center_median <= 120:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 3))
        morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel_small)
        kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 3))
        return cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_large)
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel_small)
        kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        return cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_large)


# =============================================
# [신규 7] 연결 컴포넌트 라벨링 노이즈 제거 (8방향 윤곽선 추적 기반)
# =============================================
def remove_noise_components(binary):
    """
    8방향 연결 컴포넌트 라벨링으로 노이즈 제거:
    - 전체 너비에 걸친 덩어리(w > 70%) 제거 → 세로획만 복원
    - 가로줄(w/h > 8, h < 25%) 제거
    - 너무 작은 점/노이즈(면적 < 0.3%, h < 20%) 제거
    - [신규] 볼트/나사: 작고 둥근 형태(종횡비 0.5~2.0), 양 끝단(x<13% or x+w>87%) 제거
    - [신규] 번호판 테두리 가로 띠: 상단/하단 얇은 가로 덩어리(h<15%, w>25%) 제거
    """
    result = np.zeros_like(binary)
    h_img, w_img = binary.shape

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    min_area = h_img * w_img * 0.002   # 완화: 0.003→0.002
    min_h = max(6, h_img * 0.15)      # 완화: 0.20→0.15 (얇은 획 보존)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        x    = stats[i, cv2.CC_STAT_LEFT]
        y    = stats[i, cv2.CC_STAT_TOP]
        w    = stats[i, cv2.CC_STAT_WIDTH]
        h    = stats[i, cv2.CC_STAT_HEIGHT]

        aspect = (w / h) if h > 0 else 0

        is_hline     = (h > 0 and aspect > 8 and h < h_img * 0.25)
        too_small    = (area < min_area and h < min_h)
        is_fullwidth = (w > w_img * 0.70)

        # 볼트/나사: 번호판 양 끝단 가장자리(좌 8%, 우 8%)에 위치하고
        # 면적 작고(1.0% 미만), 종횡비 거의 1:1(둥근 형태), 높이 40% 미만
        is_bolt = (
            area < h_img * w_img * 0.010
            and 0.5 <= aspect <= 2.0
            and (x < w_img * 0.08 or (x + w) > w_img * 0.92)
            and h < h_img * 0.40
        )

        # 번호판 테두리 가로 띠: 상단/하단에 붙은 가로로 긴 얇은 덩어리
        is_border_band = (
            h < h_img * 0.15
            and w > w_img * 0.25
            and (y < h_img * 0.15 or (y + h) > h_img * 0.85)
        )

        if is_hline or too_small or is_bolt or is_border_band:
            continue

        if is_fullwidth:
            # fullwidth 덩어리: 경계 열(x<5%, x>90%) 제외하고 세로획만 복원
            comp_mask = (labels == i).astype(np.uint8) * 255
            vertical = np.zeros_like(comp_mask)
            min_char_h = max(4, h_img * 0.30)
            col_start = int(w_img * 0.05)
            col_end   = int(w_img * 0.90)
            for col in range(col_start, col_end):
                col_pixels = np.where(comp_mask[:, col] > 0)[0]
                if len(col_pixels) == 0:
                    continue
                gaps = np.where(np.diff(col_pixels) > 2)[0]
                starts = np.concatenate([[col_pixels[0]], col_pixels[gaps + 1]])
                ends   = np.concatenate([col_pixels[gaps], [col_pixels[-1]]])
                for s, e in zip(starts, ends):
                    if e - s >= min_char_h:
                        vertical[s:e+1, col] = 255
            result = cv2.bitwise_or(result, vertical)
        else:
            result[labels == i] = 255

    return result


# =============================================
# DI (Direction Index) 계산 — 박스 필터용
# =============================================
def _pt_to_dir(dx, dy):
    if dx == 1  and dy == 0:  return 0
    if dx == 1  and dy == -1: return 1
    if dx == 0  and dy == -1: return 2
    if dx == -1 and dy == -1: return 3
    if dx == -1 and dy == 0:  return 4
    if dx == -1 and dy == 1:  return 5
    if dx == 0  and dy == 1:  return 6
    if dx == 1  and dy == 1:  return 7
    return -1

def compute_DI(contour):
    pts = contour[:, 0, :]
    n = len(pts)
    if n < 2:
        return 0
    angle_weight = [0, 1, 2, 3, 4, 3, 2, 1]
    dirs = []
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        d = _pt_to_dir(int(x2 - x1), int(y2 - y1))
        if d >= 0:
            dirs.append(d)
    di = 0
    for i in range(1, len(dirs)):
        diff = abs(dirs[i] - dirs[i-1])
        if diff > 4:
            diff = 8 - diff
        di += angle_weight[diff]
    return di


# =============================================
# 4단계 보조: 한글 zone 경계에서 강제 분할 (core v5c 포팅)
# =============================================
def _split_at_hangul_boundary(morph, box, hangul_x0, hangul_x1):
    """
    큰 박스(연결박스)를 한글 zone 경계에서 강제 분할.
    세 구간으로 나눔: [bx ~ hangul_x0] / [hangul_x0 ~ hangul_x1] / [hangul_x1 ~ bx+bw]
    각 구간에 픽셀이 있으면 tight bbox 추출해서 반환.
    """
    bx, by, bw, bh = box
    h_img, w_img = morph.shape
    box_end = bx + bw
    results = []

    def extract(x_start, x_end):
        if x_start >= x_end:
            return
        patch = morph[by:by+bh, x_start:x_end]
        if patch.size == 0 or not np.any(patch > 0):
            return
        # 연결 컴포넌트별로 각각 박스 반환 (5와 6이 붙어있어도 분리 가능)
        n_labels, labels = cv2.connectedComponents(patch.astype(np.uint8))
        for lbl in range(1, n_labels):
            ys, xs = np.where(labels == lbl)
            px0, px1 = int(xs.min()), int(xs.max()) + 1
            py0, py1 = int(ys.min()), int(ys.max()) + 1
            pw, ph = px1 - px0, py1 - py0
            if pw >= w_img * 0.025 and ph >= h_img * 0.08:
                results.append((x_start + px0, by + py0, pw, ph))

    def extract_with_vsplit(x_start, x_end, expected_w):
        """구간 내 박스가 1개 컴포넌트로 연결돼도 vertical projection으로 추가 분리 시도"""
        before = len(results)
        extract(x_start, x_end)
        added = results[before:]
        if len(added) == 1:
            # 1개만 나왔고 너무 넓으면 vertical projection으로 추가 split
            ax, ay, aw, ah = added[0]
            if aw > expected_w * 1.1:
                pieces = _split_wide_box(morph, added[0], expected_w, min_valley_ratio=0.50)
                if len(pieces) > 1:
                    results.pop()
                    for p in pieces:
                        px, py, pw, ph = p
                        if pw >= w_img * 0.025 and ph >= h_img * 0.08:
                            results.append(p)

    expected_char_w = w_img / 7.0

    # 왼쪽 숫자 구간 (bx ~ hangul_x0)
    if bx < hangul_x0:
        extract(bx, hangul_x0)
    # 한글 zone 구간 (hangul_x0 ~ hangul_x1) — box가 걸치는 경우만
    h_start = max(bx, hangul_x0)
    h_end   = min(box_end, hangul_x1)
    if h_start < h_end:
        extract_with_vsplit(h_start, h_end, expected_char_w)
    # 오른쪽 숫자 구간 (hangul_x1 ~ box_end)
    if box_end > hangul_x1:
        extract(hangul_x1, box_end)

    return results if results else [box]


# =============================================
# 4단계 보조: 넓은 박스를 수직 투영 골짜기로 분리
# =============================================
def _split_wide_box(morphed, box, expected_char_w, min_valley_ratio=0.70):
    x, y, w, h = box
    strip = morphed[y:y+h, x:x+w]
    v = np.sum(strip > 0, axis=0).astype(float)
    if v.max() > 0:
        v /= v.max()

    cx = w // 2
    search_l = max(1, cx - int(w * 0.40))
    search_r = min(w - 1, cx + int(w * 0.40))
    valley_idx = search_l + int(np.argmin(v[search_l:search_r]))

    left_max  = v[:valley_idx].max()  if valley_idx > 0     else 1.0
    right_max = v[valley_idx:].max()  if valley_idx < w - 1 else 1.0
    peak      = min(left_max, right_max)

    if peak > 0 and v[valley_idx] <= peak * min_valley_ratio:
        split = valley_idx
        left_strip  = strip[:, :split]
        right_strip = strip[:, split:]

        def tight_box(sub, ox, oy):
            ys, xs = np.where(sub > 0)
            if len(xs) == 0:
                return None
            return (ox + int(xs.min()), oy + int(ys.min()),
                    int(xs.max()) - int(xs.min()) + 1,
                    int(ys.max()) - int(ys.min()) + 1)

        lb = tight_box(left_strip,  x,         y)
        rb = tight_box(right_strip, x + split, y)
        if lb and rb:
            return [lb, rb]

    return [box]


# =============================================
# 4단계: 글자 분리 + 32x32 정규화 (v5c 고정 zone 방식, core 포팅)
# =============================================
def segment_and_normalize(morphed, target_count=TARGET_CHAR_COUNT, return_boxes=False):
    """
    [v5c 포팅] 한글 위치 고정 zone + 양방향 연결박스 강제분할
    - 한글은 240px 기준 x≈55~118(2/7~3/7) 고정 zone
    - 연결박스 경계 강제분할 (_split_at_hangul_boundary)
    - hangul_x0 경계 조건: 중심 > hangul_x0 AND bx >= hangul_x0
    - 한글 조각(ㅅ+ㅜ 등) bounding box 합침
    - 중복 digit 제거 (한글박스 범위 내 겹치는 digit 제거)
    - 왼쪽 숫자 suspicious 보호 (x<8%, w<=4% 얇은 박스 삭제 금지)
    """
    h_img, w_img = morphed.shape
    hangul_x0 = int(w_img * (2/7 - 0.05))   # ≈ 55px
    hangul_x1 = int(w_img * (3/7 + 0.05))   # ≈ 118px

    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = 25
    min_w    = w_img * 0.060   # 0.025→0.060: 너무 얇은 조각(w<6px) 제거
    max_w    = w_img * 0.30    # 0.22→0.30: 넓은 연결박스도 받아서 split 대상으로
    min_h    = h_img * 0.20    # 원래값 유지
    max_h    = h_img * 0.85

    raw_boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if area > min_area:
            raw_boxes.append((x, y, w, h))

    if not raw_boxes:
        return ([], []) if return_boxes else []

    def _is_border_artifact(b):
        """deskew 후 테두리 이물질 판별"""
        bx, by, bw, bh = b
        is_thin      = (bw <= 6)
        left_thin    = (bx == 0 and is_thin and bh > h_img * 0.65)
        right_thin   = (bx + bw >= w_img * 0.97 and is_thin)
        left_tall_wide = (bx < w_img * 0.08 and bh > h_img * 0.65
                          and not (bx == 0 and bw <= 5))
        # 하단 테두리 조각: 하단에 붙어있고 높이가 35% 미만인 가로로 긴 덩어리
        bottom_band = ((by + bh) >= h_img * 0.88 and bh < h_img * 0.35
                       and bw > w_img * 0.08)
        return left_thin or right_thin or left_tall_wide or bottom_band

    # min_w: 기본 6% 이상. 단, h>=40%인 얇은 숫자(1 등)는 3% 이상으로 완화
    normal_boxes = [b for b in raw_boxes
                     if (b[2] >= min_w or b[3] >= h_img * 0.40) and b[2] <= max_w
                     and min_h <= b[3] <= max_h
                     and b[3] / b[2] < 4.0
                     and b[3] < h_img * 0.80
                     and not (b[1] == 0 and b[3] >= h_img * 0.9)
                     and not (b[0] + b[2] >= w_img * 0.97 and b[2] <= w_img * 0.15)
                     and not _is_border_artifact(b)]
    normal_boxes.sort(key=lambda b: b[0])

    # 한글 zone 내 작은 조각(ㅗ, ㅜ 등) 별도 수집 — min_h 기준 완화
    hangul_frags = [b for b in raw_boxes
                    if hangul_x0 <= b[0] + b[2]/2 <= hangul_x1   # 한글 zone 중심
                    and b[2] >= min_w                              # 최소 폭
                    and b[3] >= 4                                  # 최소 4px 높이
                    and b[2] / max(b[3], 1) > 1.5                 # 가로가 세로보다 넓은 조각(ㅗ,ㅜ)
                    and b[3] < h_img * 0.20                        # min_h 미달인 것만
                    and b not in normal_boxes]

    if len(normal_boxes) < max(3, target_count - 3):
        max_w_relaxed = w_img * 0.30
        normal_boxes = [b for b in raw_boxes
                         if min_w <= b[2] <= max_w_relaxed and min_h <= b[3] <= max_h
                         and b[3] / b[2] < 4.0
                         and b[3] < h_img * 0.80
                         and not (b[1] == 0 and b[3] >= h_img * 0.9)
                         and not (b[0] + b[2] >= w_img * 0.97 and b[2] <= w_img * 0.15)
                         and not _is_border_artifact(b)]
        normal_boxes.sort(key=lambda b: b[0])

    # 한글zone 경계 넘는 큰 박스 → 경계에서 강제 분할 (최대 2회 반복)
    for _pass in range(2):
        split_boxes = []
        changed = False
        for b in normal_boxes:
            bx, by, bw, bh = b
            crosses_left  = (bx <= hangul_x0 and bx + bw > hangul_x0)  # 등호: bx==hangul_x0도 포함
            crosses_right = (bx < hangul_x1 and bx + bw > hangul_x1)
            is_wide = (bw >= w_img * 0.07)
            if (crosses_left or crosses_right) and is_wide:
                pieces = _split_at_hangul_boundary(morphed, b, hangul_x0, hangul_x1)
                split_boxes.extend(pieces)
                if len(pieces) > 1:
                    changed = True
            else:
                split_boxes.append(b)
        normal_boxes = sorted(split_boxes, key=lambda b: b[0])
        if not changed:
            break

    # 한글zone 무관 넓은 박스(숫자끼리 연결) → vertical projection으로 split
    expected_char_w = w_img / 7.0
    split_boxes2 = []
    for b in normal_boxes:
        bx, by, bw, bh = b
        # hangul zone을 완전히 벗어나고 너무 넓은 박스만 대상
        in_hangul = not (bx + bw <= hangul_x0 or bx >= hangul_x1)
        if not in_hangul and bw > expected_char_w * 1.5:
            pieces = _split_wide_box(morphed, b, expected_char_w, min_valley_ratio=0.60)
            split_boxes2.extend(pieces)
        else:
            split_boxes2.append(b)
    normal_boxes = sorted(split_boxes2, key=lambda b: b[0])

    # 한글 zone / 숫자 zone 분리
    def overlaps_hangul(b):
        center = b[0] + b[2] / 2
        return hangul_x0 <= center <= hangul_x1

    contour_hangul = [b for b in normal_boxes if overlaps_hangul(b)]
    digit_boxes    = [b for b in normal_boxes if not overlaps_hangul(b)]

    # 한글 zone 직접 픽셀 추출 (fallback)
    hangul_zone = morphed[:, hangul_x0:hangul_x1]
    zone_white  = int(np.sum(hangul_zone > 0))
    zone_box = None
    if zone_white > 20:
        ys, xs = np.where(hangul_zone > 0)
        y0z, y1z = int(ys.min()), int(ys.max()) + 1
        x0z, x1z = int(xs.min()), int(xs.max()) + 1
        if y1z - y0z >= h_img * 0.10:   # 0.20→0.10: ㅗ 등 작은 조각 포함
            zone_box = (hangul_x0 + x0z, y0z, x1z - x0z, y1z - y0z)

    # 최종 한글 박스 선택
    if contour_hangul:
        # 중심이 hangul_x0 오른쪽 AND bx >= hangul_x0 인 것만 한글로 판단
        # bx < hangul_x0 인 박스 = 한글 왼쪽에서 시작 = 숫자 (22보4690의 '2' 오인 방지)
        # hangul zone 내 박스 중, bx가 hangul_x0보다 왼쪽에 걸쳐있으면
        # hangul 부분만 tight-crop하고 왼쪽 숫자 부분은 digit_boxes로 복원
        rescued_digits = []
        trimmed_hangul = []
        for b in contour_hangul:
            bx, by, bw, bh = b
            center = bx + bw / 2
            if center > hangul_x0 and bx >= hangul_x0:
                trimmed_hangul.append(b)
            elif center > hangul_x0 and bx < hangul_x0:
                # bx < hangul_x0: 왼쪽에 숫자가 걸쳐있음 → 3구간 분리
                pieces = _split_at_hangul_boundary(morphed, b, hangul_x0, hangul_x1)
                for p in pieces:
                    px, py, pw, ph = p
                    pc = px + pw / 2
                    if hangul_x0 <= pc <= hangul_x1:
                        trimmed_hangul.append(p)
                    else:
                        rescued_digits.append(p)
            else:
                rescued_digits.append(b)
        digit_boxes = sorted(digit_boxes + rescued_digits, key=lambda b: b[0])

        # hangul zone 내 박스 중 너무 넓은 것 → vertical projection으로 추가 split
        # (6로처럼 숫자+한글이 한글zone 안에서 붙어있는 케이스)
        char_w = w_img / 7.0  # 한 글자 예상 폭
        final_hangul = []
        for b in trimmed_hangul:
            bx, by, bw, bh = b
            if bw > char_w * 1.3:
                # 너비가 한 글자보다 30% 이상 크면 vertical projection split 시도
                pieces = _split_wide_box(morphed, b, char_w)
                if len(pieces) > 1:
                    # 분리된 조각들을 x 순서로 정렬: 가장 오른쪽 조각만 한글, 나머지는 숫자
                    pieces_sorted = sorted(pieces, key=lambda p: p[0])
                    for i, p in enumerate(pieces_sorted):
                        px, py, pw, ph = p
                        pc = px + pw / 2
                        if i == len(pieces_sorted) - 1 and hangul_x0 <= pc <= hangul_x1:
                            # 마지막(가장 오른쪽) 조각이 hangul zone 내에 있으면 한글
                            final_hangul.append(p)
                        else:
                            digit_boxes.append(p)
                    digit_boxes.sort(key=lambda b: b[0])
                    continue
            # bx+bw가 hangul_x1 넘으면 오른쪽도 잘라내기
            if bx + bw > hangul_x1:
                pieces = _split_at_hangul_boundary(morphed, b, hangul_x0, hangul_x1)
                for p in pieces:
                    px, py, pw, ph = p
                    pc = px + pw / 2
                    if hangul_x0 <= pc <= hangul_x1:
                        final_hangul.append(p)
                    else:
                        digit_boxes.append(p)
                digit_boxes.sort(key=lambda b: b[0])
            else:
                final_hangul.append(b)
        contour_hangul = final_hangul

        hangul_only  = [b for b in contour_hangul if (b[0] + b[2]/2) > hangul_x0 and b[0] >= hangul_x0]
        spill_digits = [b for b in contour_hangul if not ((b[0] + b[2]/2) > hangul_x0 and b[0] >= hangul_x0)]
        if spill_digits:
            digit_boxes = digit_boxes + spill_digits
            digit_boxes.sort(key=lambda b: b[0])
        if hangul_only:
            # hangul_only가 여러 개면: 가장 오른쪽 조각이 진짜 한글,
            # 나머지(왼쪽 조각들)는 숫자로 되돌림
            if len(hangul_only) > 1:
                hangul_only_sorted = sorted(hangul_only, key=lambda b: b[0])
                extra_digits = hangul_only_sorted[:-1]  # 왼쪽 조각들
                hangul_only = [hangul_only_sorted[-1]]  # 가장 오른쪽만 한글
                digit_boxes = sorted(digit_boxes + extra_digits, key=lambda b: b[0])
            # hangul_frags(ㅗ,ㅜ 등 작은 조각)도 포함해서 bounding box 합침
            # x 범위는 hangul_only 기준, y 범위는 frags 포함해서 확장
            all_x0 = min(b[0] for b in hangul_only)
            all_x1 = max(b[0] + b[2] for b in hangul_only)
            all_y0 = min(b[1] for b in hangul_only + hangul_frags)
            all_y1 = max(b[1] + b[3] for b in hangul_only + hangul_frags)
            final_hangul = (all_x0, all_y0, all_x1 - all_x0, all_y1 - all_y0)
        else:
            final_hangul = zone_box
    else:
        final_hangul = zone_box

    # y방향 패딩만 (x패딩 금지 — 인접 숫자와 겹침 방지)
    if final_hangul:
        hx, hy, hw, hh = final_hangul
        py = max(2, int(hh * 0.08))
        final_hangul = (hx, max(0, hy - py),
                        hw,
                        min(h_img - max(0, hy - py), hh + 2 * py))

    # 한글 박스 범위와 겹치는 digit 중복 제거
    if final_hangul:
        hx_f, hy_f, hw_f, hh_f = final_hangul
        def overlaps_final_hangul(b):
            bx, by, bw, bh = b
            return not (bx + bw <= hx_f or bx >= hx_f + hw_f)
        digit_boxes = [b for b in digit_boxes if not overlaps_final_hangul(b)]

    hangul_center = (hangul_x0 + hangul_x1) / 2
    left_digits  = sorted([b for b in digit_boxes if b[0] + b[2]/2 < hangul_center],  key=lambda b: b[0])
    right_digits = sorted([b for b in digit_boxes if b[0] + b[2]/2 >= hangul_center], key=lambda b: b[0])

    final_left  = left_digits   # [-2:] 제한 제거
    final_right = right_digits[:4]

    final_candidates = sorted(final_left, key=lambda b: b[0])
    if final_hangul:
        final_candidates.append(final_hangul)
    final_candidates += sorted(final_right, key=lambda b: b[0])
    final_candidates.sort(key=lambda b: b[0])

    # DI 박스필터
    contour_map = {}
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        contour_map[(x, y, w, h)] = c

    filtered_candidates = []
    for b in final_candidates:
        c = contour_map.get(b)
        di = compute_DI(c) if c is not None else -1
        too_tall = (b[3] >= h_img * 0.85)  # DI 무관: 번호판 높이 85% 이상은 무조건 제거
        low_di   = (di != -1 and di < 4)   # 완화: 6→4
        # 크기 필터: w<6% 이지만 h>=40%면 얇은 숫자(1 등)로 허용
        too_thin  = (b[2] < w_img * 0.06) and (b[3] < h_img * 0.40)
        too_short = (b[3] < h_img * 0.25)  # 완화: 0.35→0.25
        if not too_tall and not low_di and not too_thin and not too_short:
            filtered_candidates.append(b)
    final_candidates = sorted(filtered_candidates, key=lambda b: b[0])

    # 7개 미만이면 raw_boxes에서 겹치지 않는 박스 추가로 채움
    if len(final_candidates) < TARGET_CHAR_COUNT:
        for b in sorted(raw_boxes, key=lambda b: b[2], reverse=True):
            if len(final_candidates) >= TARGET_CHAR_COUNT:
                break
            if not (min_w <= b[2] <= w_img * 0.35 and min_h * 0.5 <= b[3] <= max_h):
                continue
            overlap = any(
                not (b[0] + b[2] <= eb[0] or b[0] >= eb[0] + eb[2])
                for eb in final_candidates
            )
            if not overlap:
                final_candidates.append(b)
        final_candidates.sort(key=lambda b: b[0])

    # 왼쪽 숫자 2개 보장 — 한글 zone 왼쪽에 1개 이하이면 raw_boxes tiny도 허용
    hangul_xc = (hangul_x0 + hangul_x1) / 2
    left_count = sum(1 for b in final_candidates if b[0] + b[2]/2 < hangul_xc)
    if left_count < 2:
        # raw_boxes 중 왼쪽 zone에 있고 겹치지 않는 박스를 최대 (2-left_count)개 보충
        # 높이 기준 완화: min_h * 0.3 이상
        for b in sorted(raw_boxes, key=lambda b: b[3], reverse=True):
            if left_count >= 2:
                break
            bx, by, bw, bh = b
            if bx + bw/2 >= hangul_xc:  # 오른쪽 zone은 제외
                continue
            if bw < 3 or bh < h_img * 0.30:  # 너무 tiny는 제외
                continue
            overlap = any(
                not (bx + bw <= eb[0] or bx >= eb[0] + eb[2])
                for eb in final_candidates
            )
            if not overlap:
                final_candidates.append(b)
                left_count += 1
        final_candidates.sort(key=lambda b: b[0])

    normalized_chars = []
    boxes = []

    for (x, y, w, h) in final_candidates:
        if w > 0 and h > 0:
            pad = max(3, int(h * 0.15))
            y0 = max(0, y - pad)
            y1 = min(h_img, y + h + pad)
            char_img = morphed[y0:y1, x:x+w]
            ch = y1 - y0

            if w > 0 and ch > 0:
                resized = cv2.resize(char_img, (26, 26), interpolation=cv2.INTER_AREA)
                canvas = np.zeros((CHAR_SIZE, CHAR_SIZE), dtype=np.uint8)
                canvas[3:29, 3:29] = resized
                normalized = np.where(canvas > 127, 1, -1).astype(np.int8)
                normalized_chars.append(normalized)
                boxes.append((x, y0, w, ch))

    if return_boxes:
        return normalized_chars, boxes
    return normalized_chars


# =============================================
# 단일 이미지 전체 파이프라인
# =============================================
def preprocess_one_full(image_path, return_debug=False):
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return (None, None) if return_debug else None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plate_crop, plate_box = detect_plate_night(img, gray)
    if plate_crop is None or plate_crop.size == 0:
        return (None, None) if return_debug else None

    try:
        plate_crop = deskew_plate(plate_crop)
    except Exception:
        pass  # deskew 실패시 원본 사용

    ch, cw = plate_crop.shape
    top_cut    = int(ch * 0.15)
    bottom_cut = int(ch * 0.15)
    left_cut   = int(cw * 0.03)
    right_cut  = int(cw * 0.03)
    inner = plate_crop[top_cut:ch - bottom_cut, left_cut:cw - right_cut]
    if inner.size == 0:
        inner = plate_crop

    inner = enhance_plate(inner)

    ih, iw = inner.shape
    if iw < 120 or ih < 30:
        inner = cv2.resize(inner, (iw * 2, ih * 2), interpolation=cv2.INTER_LANCZOS4)

    ih2, iw2 = inner.shape
    if iw2 > PLATE_W or ih2 > PLATE_H:
        plate_resized = cv2.resize(inner, (PLATE_W, PLATE_H), interpolation=cv2.INTER_AREA)
    else:
        plate_resized = cv2.resize(inner, (PLATE_W, PLATE_H), interpolation=cv2.INTER_LANCZOS4)

    morphed_raw = binarize_and_morphology(plate_resized)
    morphed = remove_noise_components(morphed_raw)

    chars, boxes = segment_and_normalize(morphed, return_boxes=True)

    # fallback: 세그 6개 이하면 C값 낮춘 파라미터로 재시도 (획 보존 우선)
    if len(chars) <= 6:
        morphed_fb = _binarize_low_c(plate_resized)
        morphed_fb = remove_noise_components(morphed_fb)
        chars_fb, boxes_fb = segment_and_normalize(morphed_fb, return_boxes=True)
        if len(chars_fb) > len(chars):
            chars, boxes = chars_fb, boxes_fb
            morphed = morphed_fb

    if return_debug:
        debug_info = {
            'img': img, 'gray': gray, 'plate_box': plate_box,
            'plate_resized': plate_resized, 'morph': morphed, 'boxes': boxes
        }
        return chars, debug_info

    return chars