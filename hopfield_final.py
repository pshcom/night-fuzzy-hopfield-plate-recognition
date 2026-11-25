"""
hopfield_final.py (보완판)
─────────────────────────
[보완 1] 노이즈 증강 학습 (hopfield_network_v2에서 처리)
[보완 2] 혼동 행렬(Confusion Matrix) 시각화 추가
         어떤 글자끼리 헷갈리는지 히트맵으로 확인 가능
[보완 3] 평가 지표 수정
         - 완전복원율: recalled == pat (원본 입력 복원 여부) 로 수정
         - 클래스 인식률: recall → classify_direct 2단계 파이프라인으로 수정
           (기존: classify_direct(noisy) 단독 → recall 성능이 아니었음)
[유지]   Pseudo-inverse 학습 / classify_direct / train/valid 분리 평가
"""
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hopfield_network import HopfieldNetwork, NOISE_RATIO

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PREP = os.path.join(SCRIPT_DIR, 'prep_train')
VALID_PREP = os.path.join(SCRIPT_DIR, 'prep_valid')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'hopfield_final')

DIGITS = set('0123456789')
HANGUL = set('가나다라마거너더러머버서어저고노도로모보소오조구누두루무부수우주아바사자카타파하허배울')

def load_and_split(prep_dir):
    chars  = np.load(os.path.join(prep_dir, 'chars_flat.npy'))
    labels = np.load(os.path.join(prep_dir, 'labels_all.npy'), allow_pickle=True)

    digit_dict  = {}
    hangul_dict = {}
    for plate_idx, label in enumerate(labels):
        for pos, ch in enumerate(list(str(label))[:7]):
            pattern = chars[plate_idx * 7 + pos]
            if np.all(pattern == -1):
                continue
            if ch in DIGITS:
                digit_dict.setdefault(ch, []).append(pattern)
            elif ch in HANGUL:
                hangul_dict.setdefault(ch, []).append(pattern)

    digit_arrays  = {ch: np.stack(p) for ch, p in digit_dict.items()}
    hangul_arrays = {ch: np.stack(p) for ch, p in hangul_dict.items()}
    print(f"  숫자 클래스: {sorted(digit_arrays.keys())} ({len(digit_arrays)}종)")
    print(f"  한글 클래스: {sorted(hangul_arrays.keys())} ({len(hangul_arrays)}종)")
    return digit_arrays, hangul_arrays

def add_noise(pattern, ratio=NOISE_RATIO):
    noisy  = pattern.copy()
    n_flip = int(len(pattern) * ratio)
    idx    = np.random.choice(len(pattern), n_flip, replace=False)
    noisy[idx] *= -1
    return noisy

def evaluate(hop_digit, hop_hangul, class_arrays, label=''):
    bit_accs, exact_hits, recog_hits = [], [], []
    for ch, patterns in class_arrays.items():
        hop = hop_digit if ch in DIGITS else hop_hangul
        if ch not in hop.rep_chs:
            continue

        n_valid = min(20, len(patterns))
        sample = patterns[np.random.choice(len(patterns), n_valid, replace=False)]
        for pat in sample:
            noisy = add_noise(pat)

            # [수정] recall 수행
            recalled, _ = hop.recall(noisy)

            # [수정] 완전복원율: 원본 입력(pat)과 비교
            bit_acc = np.mean(recalled == pat)
            exact   = int(np.array_equal(recalled, pat))

            # [수정] 클래스 인식률: recall 결과를 classify_direct에 통과 (2단계 파이프라인)
            pred_ch, sim = hop.classify_direct(recalled)
            recog = int(pred_ch == ch)

            bit_accs.append(bit_acc)
            exact_hits.append(exact)
            recog_hits.append(recog)

    avg_bit   = np.mean(bit_accs) * 100
    avg_exact = np.mean(exact_hits) * 100
    avg_recog = np.mean(recog_hits) * 100
    print(f"\n  [{label}] 비트정확도:{avg_bit:.2f}%  완전복원율:{avg_exact:.2f}%  클래스인식률:{avg_recog:.2f}%")
    return avg_bit, avg_exact, avg_recog

def get_class_recog(hop_digit, hop_hangul, class_arrays):
    result = {}
    for ch, patterns in class_arrays.items():
        hop = hop_digit if ch in DIGITS else hop_hangul
        if ch not in hop.rep_chs:
            continue
        n_valid = min(20, len(patterns))
        sample = patterns[np.random.choice(len(patterns), n_valid, replace=False)]
        hits = []
        for pat in sample:
            noisy = add_noise(pat)
            # [수정] recall → classify_direct 2단계
            recalled, _ = hop.recall(noisy)
            pred_ch, sim = hop.classify_direct(recalled)
            hits.append(int(pred_ch == ch))
        result[ch] = np.mean(hits)
    return result

# =============================================
# [보완] 혼동 행렬
# =============================================
def build_confusion_matrix(hop, class_arrays, char_type='digit'):
    classes = sorted(class_arrays.keys())
    n = len(classes)
    idx_map = {ch: i for i, ch in enumerate(classes)}
    cm = np.zeros((n, n), dtype=int)

    for true_ch, patterns in class_arrays.items():
        n_valid = min(20, len(patterns))
        sample = patterns[np.random.choice(len(patterns), n_valid, replace=False)]
        for pat in sample:
            noisy = add_noise(pat)
            # [수정] recall → classify_direct 2단계
            recalled, _ = hop.recall(noisy)
            pred_ch, _ = hop.classify_direct(recalled)
            if pred_ch in idx_map:
                cm[idx_map[true_ch], idx_map[pred_ch]] += 1

    return cm, classes

def plot_confusion_matrix(cm, classes, title, save_path):
    fig, ax = plt.subplots(figsize=(max(6, len(classes)*0.55), max(5, len(classes)*0.5)))
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sum > 0, cm / row_sum, 0)

    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, fontsize=8, rotation=45, ha='right')
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel('예측(Predicted)', fontsize=10)
    ax.set_ylabel('실제(True)', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')

    thresh = cm_norm.max() / 2.0
    for i in range(len(classes)):
        for j in range(len(classes)):
            val = cm_norm[i, j]
            if val > 0.01:
                ax.text(j, i, f'{val:.2f}',
                        ha='center', va='center',
                        color='white' if val > thresh else 'black',
                        fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  혼동 행렬 저장: {save_path}")

def plot_results(train_results, valid_results, class_recog_train, class_recog_valid):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('홉필드 네트워크 복원 성능 (Pseudo-inverse, recall→classify 2단계)', fontsize=13, fontweight='bold')

    metrics = ['비트 정확도(%)', '완전 복원율(%)', '클래스 인식률(%)']
    x = np.arange(len(metrics))
    width = 0.35

    ax = axes[0]
    b1 = ax.bar(x - width/2, train_results, width, label='Train', color='steelblue', alpha=0.85)
    b2 = ax.bar(x + width/2, valid_results, width, label='Valid', color='coral', alpha=0.85)
    ax.set_title('Train vs Valid 전체 성능', fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylim(0, 110); ax.legend(); ax.axhline(90, color='gray', linestyle='--', linewidth=0.8)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+1,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)

    ax2 = axes[1]
    all_chars = sorted(set(list(class_recog_train.keys()) + list(class_recog_valid.keys())))
    tr = [class_recog_train.get(c, 0) * 100 for c in all_chars]
    te = [class_recog_valid.get(c, 0) * 100 for c in all_chars]
    x2 = np.arange(len(all_chars))
    ax2.bar(x2 - width/2, tr, width, label='Train', color='steelblue', alpha=0.8)
    ax2.bar(x2 + width/2, te, width, label='valid', color='coral', alpha=0.8)
    ax2.set_title('클래스별 인식률 (Train vs valid)', fontsize=11)
    ax2.set_xticks(x2); ax2.set_xticklabels(all_chars, fontsize=7, rotation=45)
    ax2.set_ylim(0, 110)
    ax2.axhline(50, color='red', linestyle='--', linewidth=0.8, label='50%기준')
    ax2.axhline(90, color='gray', linestyle='--', linewidth=0.8, label='90%기준')
    ax2.legend(fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'final_result.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  그래프 저장: {save_path}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("="*55)
    print("  홉필드 최종 (Pseudo-inverse + recall→classify 2단계)")
    print("="*55)

    print("\n[1단계] Train 데이터 로드")
    train_digit, train_hangul = load_and_split(TRAIN_PREP)

    print("\n[2단계] 홉필드 학습")
    hop_digit  = HopfieldNetwork(name='숫자')
    hop_hangul = HopfieldNetwork(name='한글')
    hop_digit.train(train_digit,  augment=False)
    hop_hangul.train(train_hangul, augment=False)
    hop_digit.save(os.path.join(OUTPUT_DIR, 'hopfield_digit.npy'))
    hop_hangul.save(os.path.join(OUTPUT_DIR, 'hopfield_hangul.npy'))

    print("\n[3단계] Train 평가")
    train_all = {**train_digit, **train_hangul}
    tr_bit, tr_exact, tr_recog = evaluate(hop_digit, hop_hangul, train_all, 'TRAIN')
    class_recog_train = get_class_recog(hop_digit, hop_hangul, train_all)

    print("\n[4단계] Valid 데이터 로드 & 평가")
    valid_digit, valid_hangul = load_and_split(VALID_PREP)
    valid_all = {**valid_digit, **valid_hangul}
    te_bit, te_exact, te_recog = evaluate(hop_digit, hop_hangul, valid_all, 'VALID')
    class_recog_valid = get_class_recog(hop_digit, hop_hangul, valid_all)

    summary_path = os.path.join(OUTPUT_DIR, 'summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("홉필드 최종 결과 (Pseudo-inverse + recall→classify 2단계)\n")
        f.write("="*50 + "\n")
        f.write(f"{'':12} {'비트정확도':>10} {'완전복원율':>10} {'클래스인식률':>12}\n")
        f.write("-"*48 + "\n")
        f.write(f"{'TRAIN':12} {tr_bit:>9.2f}% {tr_exact:>9.2f}% {tr_recog:>11.2f}%\n")
        f.write(f"{'VALID':12} {te_bit:>9.2f}% {te_exact:>9.2f}% {te_recog:>11.2f}%\n")
    print(f"\n  요약 저장: {summary_path}")

    print("\n[5단계] 결과 그래프")
    plot_results([tr_bit, tr_exact, tr_recog], [te_bit, te_exact, te_recog],
                 class_recog_train, class_recog_valid)

    print("\n[6단계] 혼동 행렬 생성")
    cm_d, cls_d = build_confusion_matrix(hop_digit, valid_digit, 'digit')
    plot_confusion_matrix(cm_d, cls_d, '숫자 혼동 행렬 (Valid, 노이즈 20%, recall→classify)',
                          os.path.join(OUTPUT_DIR, 'confusion_digit.png'))

    cm_h, cls_h = build_confusion_matrix(hop_hangul, valid_hangul, 'hangul')
    plot_confusion_matrix(cm_h, cls_h, '한글 혼동 행렬 (Valid, 노이즈 20%, recall→classify)',
                          os.path.join(OUTPUT_DIR, 'confusion_hangul.png'))

    print("\n  [취약 클래스 분석]")
    for cm, classes, name in [(cm_d, cls_d, '숫자'), (cm_h, cls_h, '한글')]:
        row_sum = cm.sum(axis=1, keepdims=True)
        cm_norm = np.where(row_sum > 0, cm / row_sum, 0)
        diag = np.diag(cm_norm)
        print(f"\n  {name} — 인식률 70% 미만 클래스:")
        weak_found = False
        for i, ch in enumerate(classes):
            acc = diag[i] * 100
            if acc < 70:
                row = cm_norm[i].copy()
                row[i] = 0
                confused_idx = int(np.argmax(row))
                confused_ch = classes[confused_idx]
                confused_rate = row[confused_idx] * 100
                print(f"    '{ch}': {acc:.0f}%  →  '{confused_ch}'로 {confused_rate:.0f}% 오인식")
                weak_found = True
        if not weak_found:
            print(f"    없음 (모든 클래스 70% 이상)")

    print("\n완료!")
    print("  → 다음 파일들을 확인하세요:")
    print(f"     {OUTPUT_DIR}\\final_result.png")
    print(f"     {OUTPUT_DIR}\\confusion_digit.png")
    print(f"     {OUTPUT_DIR}\\confusion_hangul.png")
    print(f"     {OUTPUT_DIR}\\summary.txt")

if __name__ == '__main__':
    main()