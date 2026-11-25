"""
run.py
─────────────────────────────────────────────
학습된 홉필드 가중치(hopfield_digit.npy, hopfield_hangul.npy)를 불러와서
임의의 저조도(야간) 번호판 jpg 이미지 1장을 입력하면
전처리 → 글자 분리 → 홉필드 복원 → 클래스 인식을 거쳐
번호판 문자열(예: '48러6567')을 출력하고 시각화합니다.

사용법:
    python run.py "C:\ai\car_plate\Plate_MM\testplate\test1.jpg"
"""
import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from infer_preprocessing import preprocess_one_full, CHAR_SIZE
from hopfield_network import HopfieldNetwork

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# =============================================
# 설정값 (실행 위치 기준 상대경로 -> USB로 옮겨도 동작)
# =============================================
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
HOPFIELD_DIR = os.path.join(SCRIPT_DIR, 'hopfield_final')
DIGIT_PATH   = os.path.join(HOPFIELD_DIR, 'hopfield_digit.npy')
HANGUL_PATH  = os.path.join(HOPFIELD_DIR, 'hopfield_hangul.npy')
DEFAULT_IMAGE = os.path.join(SCRIPT_DIR, 'testplate', 'test2.jpg')

DIGITS = set('0123456789')

# 번호판 자리별 허용 문자 종류 (구조적 제약으로 오인식 방지)
POSITION_TYPE = ['digit', 'digit', 'hangul', 'digit', 'digit', 'digit', 'digit']

# =============================================
# 단일 글자 인식 (위치별로 숫자/한글 망 선택)
# =============================================
def recognize_char(pattern, position, hop_digit, hop_hangul):
    """
    position: 0~6 (번호판 내 위치)
    POSITION_TYPE에 따라 숫자망/한글망 선택
    → 구조적 제약으로 숫자 자리에 한글이 매칭되는 오류 자체를 차단

    [핵심] 실제 분류 판단은 classify_direct(노이즈 낀 입력을 직접 비교)로 한다.
    recall(반복수렴)은 시각화를 위한 "정제된 이미지"를 보여주는 용도로만 사용.
    (검증 결과: recall 후 분류는 35% 수준, 직접분류는 99.8% 수준)
    """
    char_type = POSITION_TYPE[position]
    hop = hop_digit if char_type == 'digit' else hop_hangul

    pattern_flat = pattern.flatten()   # (32,32) -> (1024,)

    # 실제 인식 결과는 직접분류로 결정
    pred_char, cos_sim = hop.classify(pattern_flat, debug=True)
    confidence = (cos_sim + 1) / 2     # 0~1로 정규화 (표시용)

    # 시각화용 복원 이미지 (참고용, 분류 판단에는 사용 안 함)
    recalled_flat, n_iter = hop.recall(pattern_flat)
    recalled = recalled_flat.reshape(pattern.shape)

    return pred_char, confidence, recalled, n_iter

# =============================================
# 전체 번호판 인식 파이프라인
# =============================================
def recognize_plate(image_path, hop_digit, hop_hangul):
    chars, debug = preprocess_one_full(image_path, return_debug=True)

    if chars is None or len(chars) == 0:
        return None, None, debug

    results = []
    for pos, pattern in enumerate(chars):
        if pos >= 7:
            break
        pred_char, conf, recalled, n_iter = recognize_char(
            pattern, pos, hop_digit, hop_hangul
        )
        results.append({
            'position': pos,
            'input_pattern': pattern,
            'recalled': recalled,
            'pred_char': pred_char,
            'confidence': conf,
            'n_iter': n_iter
        })

    plate_string = ''.join(r['pred_char'] for r in results)
    return plate_string, results, debug

# =============================================
# 시각화
# =============================================
def visualize_result(image_path, plate_string, results, debug):
    if debug is None or results is None:
        print(f"  [실패] 번호판을 검출/분리할 수 없습니다: {image_path}")
        return

    n_chars = len(results)
    fig = plt.figure(figsize=(max(14, n_chars * 1.8), 8))
    gs  = fig.add_gridspec(3, max(n_chars, 1), height_ratios=[2, 1, 1])

    # 1행: 원본 이미지 + 검출된 번호판 위치
    ax_orig = fig.add_subplot(gs[0, :])
    vis = cv2.cvtColor(debug['img'], cv2.COLOR_BGR2RGB).copy()
    if debug.get('plate_box'):
        x, y, w, h = debug['plate_box']
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 3)
    ax_orig.imshow(vis)
    ax_orig.set_title(f"입력 이미지 — 인식 결과: '{plate_string}'",
                       fontsize=14, fontweight='bold')
    ax_orig.axis('off')

    # 2행: 글자별 입력 패턴 (전처리 후, 노이즈/번짐 있는 원본)
    for i, r in enumerate(results):
        ax = fig.add_subplot(gs[1, i])
        img = np.where(r['input_pattern'].reshape(CHAR_SIZE, CHAR_SIZE) == 1, 255, 0)
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        ax.set_title(f"입력 #{i+1}", fontsize=9)
        ax.axis('off')

    # 3행: 홉필드 복원 결과 + 예측 글자 + 신뢰도
    for i, r in enumerate(results):
        ax = fig.add_subplot(gs[2, i])
        img = np.where(r['recalled'].reshape(CHAR_SIZE, CHAR_SIZE) == 1, 255, 0)
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        color = 'green' if r['confidence'] > 0.5 else 'orange'
        ax.set_title(f"'{r['pred_char']}'\n신뢰도:{r['confidence']*100:.0f}%",
                     fontsize=10, color=color, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    save_dir  = os.path.dirname(image_path) or '.'
    save_path = os.path.join(save_dir, f"result_{os.path.basename(image_path)}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  결과 이미지 저장: {save_path}")
    plt.show()

# =============================================
# 메인
# =============================================
def main():
    if len(sys.argv) < 2:
        # 인자가 없으면 기본 경로(스크립트 기준 상대경로) 사용
        image_paths = [DEFAULT_IMAGE]
        print(f"이미지 경로 미입력, 기본 경로 사용: {DEFAULT_IMAGE}")
    else:
        image_paths = sys.argv[1:]

    print("=" * 55)
    print("  번호판 인식 — 홉필드 네트워크 (숫자/한글 분리)")
    print("=" * 55)

    print("\n학습된 가중치 로딩 중...")
    hop_digit  = HopfieldNetwork('숫자')
    hop_hangul = HopfieldNetwork('한글')
    hop_digit.load(DIGIT_PATH)
    hop_hangul.load(HANGUL_PATH)
    print(f"  숫자 클래스 {len(hop_digit.rep_chs)}개, 한글 클래스 {len(hop_hangul.rep_chs)}개 로드 완료")

    for raw_path in image_paths:
        # 상대경로면 스크립트 위치 기준으로 해석 (USB로 옮겨도 동작하도록)
        image_path = raw_path if os.path.isabs(raw_path) else os.path.join(SCRIPT_DIR, raw_path)

        print(f"\n{'─'*55}")
        print(f"처리 중: {image_path}")

        if not os.path.exists(image_path):
            print(f"  [오류] 파일을 찾을 수 없습니다.")
            continue

        plate_string, results, debug = recognize_plate(
            image_path, hop_digit, hop_hangul
        )

        if plate_string is None:
            print("  [실패] 번호판 검출 또는 글자 분리에 실패했습니다.")
            continue

        print(f"\n  ▶ 인식된 번호판: {plate_string}")
        for r in results:
            mark = "✓" if r['confidence'] > 0.5 else "?"
            print(f"    [{r['position']}] '{r['pred_char']}'  "
                  f"신뢰도:{r['confidence']*100:5.1f}%  "
                  f"수렴반복:{r['n_iter']}회  {mark}")

        visualize_result(image_path, plate_string, results, debug)

    print(f"\n{'='*55}")
    print("  완료!")
    print("=" * 55)

if __name__ == '__main__':
    main()