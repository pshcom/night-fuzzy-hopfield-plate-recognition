"""
run_batch.py
─────────────────────────────────────────────
사용법:
    python run_batch_v8.py          # testplate/ 전체 처리
    python run_batch_v8.py 이미지1.jpg 이미지2.jpg  # 개별 지정도 가능

[변경사항 v8]
- v7의 로/오 tie-break 제거 (단순 classify_direct 결과만 사용)
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
# 설정값
# =============================================
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
HOPFIELD_DIR = os.path.join(SCRIPT_DIR, 'hopfield_final')
DIGIT_PATH   = os.path.join(HOPFIELD_DIR, 'hopfield_digit.npy')
HANGUL_PATH  = os.path.join(HOPFIELD_DIR, 'hopfield_hangul.npy')
TESTPLATE_DIR = os.path.join(SCRIPT_DIR, 'testplate')
RESULT_DIR    = os.path.join(SCRIPT_DIR, 'run_result_testplate')

POSITION_TYPE = ['digit', 'digit', 'hangul', 'digit', 'digit', 'digit', 'digit']


# =============================================
# 단일 글자 인식
# =============================================
def recognize_char(pattern, position, hop_digit, hop_hangul):
    char_type = POSITION_TYPE[position]
    hop = hop_digit if char_type == 'digit' else hop_hangul

    pattern_flat = pattern.flatten()

    pred_char, cos_sim = hop.classify_direct(pattern_flat)
    confidence = (cos_sim + 1) / 2

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
# 시각화 + 저장 (show 없음)
# =============================================
def visualize_and_save(image_path, plate_string, results, debug, save_dir):
    if debug is None or results is None:
        return

    n_chars = len(results)
    fig = plt.figure(figsize=(max(14, n_chars * 1.8), 8))
    gs  = fig.add_gridspec(3, max(n_chars, 1), height_ratios=[2, 1, 1])

    # 1행: 원본 이미지 + 인식 결과
    ax_orig = fig.add_subplot(gs[0, :])
    vis = cv2.cvtColor(debug['img'], cv2.COLOR_BGR2RGB).copy()
    if debug.get('plate_box'):
        x, y, w, h = debug['plate_box']
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 3)
    ax_orig.imshow(vis)
    ax_orig.set_title(f"입력: {os.path.basename(image_path)}  —  인식 결과: '{plate_string}'",
                       fontsize=14, fontweight='bold')
    ax_orig.axis('off')

    # 2행: 글자별 입력 패턴
    for i, r in enumerate(results):
        ax = fig.add_subplot(gs[1, i])
        img = np.where(r['input_pattern'].reshape(CHAR_SIZE, CHAR_SIZE) == 1, 255, 0)
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        ax.set_title(f"입력 #{i+1}", fontsize=9)
        ax.axis('off')

    # 3행: 홉필드 복원 + 예측 글자 + 신뢰도
    for i, r in enumerate(results):
        ax = fig.add_subplot(gs[2, i])
        img = np.where(r['recalled'].reshape(CHAR_SIZE, CHAR_SIZE) == 1, 255, 0)
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        color = 'green' if r['confidence'] > 0.5 else 'orange'
        ax.set_title(f"'{r['pred_char']}'\n{r['confidence']*100:.0f}%",
                     fontsize=10, color=color, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    fname = f"result_{os.path.splitext(os.path.basename(image_path))[0]}.png"
    save_path = os.path.join(save_dir, fname)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


# =============================================
# 메인
# =============================================
def main():
    # 결과 폴더 생성
    os.makedirs(RESULT_DIR, exist_ok=True)

    # 처리할 이미지 목록 결정
    if len(sys.argv) >= 2:
        image_paths = []
        for p in sys.argv[1:]:
            image_paths.append(p if os.path.isabs(p) else os.path.join(SCRIPT_DIR, p))
    else:
        exts = ('.jpg', '.jpeg', '.png', '.bmp')
        image_paths = sorted([
            os.path.join(TESTPLATE_DIR, f)
            for f in os.listdir(TESTPLATE_DIR)
            if f.lower().endswith(exts)
        ])
        if not image_paths:
            print(f"[오류] {TESTPLATE_DIR} 에 이미지가 없습니다.")
            return

    print("=" * 60)
    print("  번호판 일괄 인식 — 홉필드 네트워크 v8")
    print(f"  처리 이미지 수: {len(image_paths)}장")
    print(f"  결과 저장 위치: {RESULT_DIR}")
    print("=" * 60)

    # 가중치 로딩 (한 번만)
    print("\n학습된 가중치 로딩 중...")
    hop_digit  = HopfieldNetwork('숫자')
    hop_hangul = HopfieldNetwork('한글')
    hop_digit.load(DIGIT_PATH)
    hop_hangul.load(HANGUL_PATH)
    print(f"  숫자 {len(hop_digit.rep_chs)}개 / 한글 {len(hop_hangul.rep_chs)}개 로드 완료\n")

    # 결과 누적
    summary = []

    for image_path in image_paths:
        fname = os.path.basename(image_path)
        print(f"{'─'*60}")
        print(f"처리 중: {fname}")

        if not os.path.exists(image_path):
            print(f"  [오류] 파일 없음")
            summary.append((fname, '[파일없음]', 0))
            continue

        plate_string, results, debug = recognize_plate(image_path, hop_digit, hop_hangul)

        if plate_string is None or results is None:
            print(f"  [실패] 번호판 검출 또는 글자 분리 실패")
            summary.append((fname, '[검출실패]', 0))
            if debug is not None and debug.get('img') is not None:
                fig, ax = plt.subplots(figsize=(6, 4))
                vis = cv2.cvtColor(debug['img'], cv2.COLOR_BGR2RGB)
                ax.imshow(vis)
                ax.set_title(f"{fname} — [검출 실패]", fontsize=12, color='red')
                ax.axis('off')
                plt.tight_layout()
                fail_path = os.path.join(RESULT_DIR, f"result_{os.path.splitext(fname)[0]}.png")
                plt.savefig(fail_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
            continue

        print(f"  ▶ 인식 결과: {plate_string}  ({len(results)}글자)")
        low_conf = []
        for r in results:
            mark = "✓" if r['confidence'] > 0.5 else "?"
            print(f"    [{r['position']}] '{r['pred_char']}'  "
                  f"신뢰도:{r['confidence']*100:5.1f}%  {mark}")
            if r['confidence'] <= 0.5:
                low_conf.append(r['position'])

        save_path = visualize_and_save(image_path, plate_string, results, debug, RESULT_DIR)
        if save_path:
            print(f"  저장: {os.path.basename(save_path)}")

        summary.append((fname, plate_string, len(low_conf)))

    # 전체 요약
    print(f"\n{'='*60}")
    print("  전체 요약")
    print(f"{'='*60}")
    print(f"  {'파일명':<25} {'인식결과':<15} {'저신뢰도글자'}")
    print(f"  {'─'*55}")
    success = 0
    for fname, plate, low in summary:
        warn = f"{low}개 주의" if isinstance(low, int) and low > 0 else ("" if plate not in ('[검출실패]', '[파일없음]') else plate)
        status = "✓" if plate not in ('[검출실패]', '[파일없음]') else "✗"
        print(f"  {status} {fname:<23} {plate:<15} {warn}")
        if status == "✓":
            success += 1
    print(f"\n  성공: {success}/{len(summary)}장")
    print(f"  결과 폴더: {RESULT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()