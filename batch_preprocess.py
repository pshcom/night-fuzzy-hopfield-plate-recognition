"""
batch_preprocess.py
train/test 폴더 전체를 preprocessing_core.py 함수로 일괄 전처리
"""
import numpy as np
import os
import glob
from pathlib import Path
from preprocessing_core import preprocess_one, CHAR_SIZE, TARGET_CHAR_COUNT

# =============================================
# 배치 처리 함수 (train/valid 공용)
# =============================================
def batch_preprocess(input_dir, output_dir, label='train'):
    os.makedirs(output_dir, exist_ok=True)

    all_jpg = glob.glob(os.path.join(input_dir, '**', '*.jpg'), recursive=True)
    total   = len(all_jpg)
    print(f"\n[{label}] 총 {total}장 전처리 시작")

    all_plates   = []
    all_labels   = []
    failed_list  = []
    partial_list = []

    for i, img_path in enumerate(all_jpg):
        if i % 200 == 0:
            print(f"  진행: {i}/{total} ({i/total*100:.1f}%)")
        try:
            chars = preprocess_one(img_path)
            if chars is None or len(chars) == 0:
                failed_list.append(img_path)
                continue

            stem      = Path(img_path).stem
            label_str = stem.split('_')[-1]

            if len(chars) < TARGET_CHAR_COUNT:
                # 7글자 미만이면 라벨-위치 매핑이 어긋나 학습 데이터가 오염됨 → skip
                partial_list.append((img_path, len(chars)))
                continue

            plate_array = np.stack(chars[:TARGET_CHAR_COUNT], axis=0)
            all_plates.append(plate_array)
            all_labels.append(label_str)

        except Exception:
            failed_list.append(img_path)

    if all_plates:
        plates_arr = np.stack(all_plates, axis=0)
        flat_arr   = plates_arr.reshape(-1, CHAR_SIZE * CHAR_SIZE).astype(np.int8)
        np.save(os.path.join(output_dir, 'plates_all.npy'), plates_arr)
        np.save(os.path.join(output_dir, 'labels_all.npy'),
                np.array(all_labels, dtype=object))
        np.save(os.path.join(output_dir, 'chars_flat.npy'), flat_arr)
        print(f"  [저장] plates: {plates_arr.shape}, chars: {flat_arr.shape}")

    if failed_list:
        with open(os.path.join(output_dir, 'failed_files.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(failed_list))
    if partial_list:
        with open(os.path.join(output_dir, 'partial_files.txt'), 'w', encoding='utf-8') as f:
            for p, cnt in partial_list:
                f.write(f"{cnt}글자\t{p}\n")

    print(f"  성공: {len(all_plates)}장 / 실패: {len(failed_list)}장 / 글자부족: {len(partial_list)}장")
    return len(all_plates)

# =============================================
# 메인: train + valid 모두 전처리
# =============================================
if __name__ == '__main__':
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    TRAIN_IN  = os.path.join(SCRIPT_DIR, 'trainplate')
    TEST_IN   = os.path.join(SCRIPT_DIR, 'validplate')
    TRAIN_OUT = os.path.join(SCRIPT_DIR, 'prep_train')
    TEST_OUT  = os.path.join(SCRIPT_DIR, 'prep_valid')

    print("=" * 55)
    print("  전처리 (한글 고정zone + 연결박스 강제분할)")
    print("  CLAHE → 적응형이진화 → 연결컴포넌트노이즈제거 → 한글zone분리")
    print("=" * 55)

    batch_preprocess(TRAIN_IN,  TRAIN_OUT, label='TRAIN')
    batch_preprocess(TEST_IN,    TEST_OUT,   label='VALID')

    print("\n전처리 완료!")