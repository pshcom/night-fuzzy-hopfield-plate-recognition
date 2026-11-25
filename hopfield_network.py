"""
hopfield_network.py (Pseudo-inverse 학습 + 클래스당 다중 대표패턴 + 노이즈 증강)
─────────────────────────────────────────────────────────────
[보완 1] 노이즈 증강 학습 (끌림 영역 확장)
  기존: medoid 패턴만 그대로 학습 → 끌림 영역이 좁아 완전복원율 낮음(20~25%)
  보완: 각 대표 패턴에 다양한 강도의 노이즈를 가해 증강 패턴을 생성하고 함께 학습.
       끌림 영역이 넓어져 완전복원율 향상 기대.

[보완 2] 데이터 불균형 클래스 샘플 수 하한 경고
  '사','자'(1개), '울'(7개) 처럼 극소 샘플 클래스를 detect해서 출력.

[보완 3] 획 구조 기반 Tie-break (도메인 지식 보정)
  홉필드 네트워크는 끌림 영역이 겹치는 혼동 쌍에서 오분류가 발생함.
  코사인 유사도 1위/2위 마진이 TIEBREAK_MARGIN 이하일 때만 발동:
  - 숫자: 5/6/8 혼동 → 상·하단 픽셀 밀도 비교로 보정
  - 한글: ㅏ/ㅓ 혼동 → 가로획 좌/우 위치, ㅗ/ㅜ 혼동 → 세로획 상/하 위치

[유지] Pseudo-inverse 학습 / 다중 medoid / classify_direct 방식
"""
import numpy as np
from sklearn.cluster import KMeans

CHAR_SIZE = 32
MAX_SWEEPS = 50
NOISE_RATIO = 0.2
N_REPS_PER_CLASS = 2      # 클래스당 서브클러스터(medoid) 수
AUGMENT_PER_REP = 1       # medoid 1개당 노이즈 증강 패턴 수
AUGMENT_NOISE_LEVELS = [0.05]  # 약한 노이즈만
MIN_SAMPLES_WARN = 10     # 이 수 미만이면 데이터 부족 경고

# =============================================
# Tie-break 설정
# =============================================
TIEBREAK_MARGIN_DIGIT  = 0.08   # 숫자: 1위/2위 마진 이하일 때 tie-break 발동
TIEBREAK_MARGIN_HANGUL = 0.03   # 한글: 더 확실할 때만 발동 (0.08은 역전 케이스 발생)

# 혼동 쌍 정의: {(1위후보, 2위후보): (1위가 맞는 조건 함수)}
# 조건 함수: pattern(32×32 int8, +1/-1) → True면 1위 유지, False면 2위로 교체
def _digit_tiebreak(pred, second, pattern_2d):
    """
    숫자 5/6/8 혼동 보정.
    32×32 패턴에서 상단(위 절반)과 하단(아래 절반) 흰 픽셀 비율로 판단.
      5: 상단 오른쪽이 열림 → 상단 우측 흰 픽셀 적음
      6: 상단이 말려 닫힘 + 하단 고리 → 상단 흰 픽셀 많고 하단도 많음
      8: 상·하단 모두 고리 → 상단 흰 픽셀 비율 ≈ 하단 (균형)
    반환: 최종 선택 문자
    """
    h, w = pattern_2d.shape
    top    = pattern_2d[:h//2, :]
    bottom = pattern_2d[h//2:, :]
    top_r  = pattern_2d[:h//2, w//2:]   # 상단 우측

    top_density    = np.mean(top    == 1)
    bottom_density = np.mean(bottom == 1)
    top_r_density  = np.mean(top_r  == 1)

    pair = tuple(sorted([pred, second]))

    if pair == ('5', '6'):
        # 5: 상단 우측 열림(밀도 낮음), 6: 상단 우측 닫힘(밀도 높음)
        return '5' if top_r_density < 0.30 else '6'

    if pair == ('5', '8'):
        # 8: 상·하단 균형, 5: 상단 밀도가 하단보다 낮음
        return '8' if abs(top_density - bottom_density) < 0.07 else '5'

    if pair == ('6', '8'):
        # 6: 하단 고리만, 8: 상·하단 모두 고리 → 상단 밀도로 구분
        return '8' if top_density > 0.25 else '6'

    if pair == ('5', '3'):
        # 3: 상·하단 오른쪽 돌출, 5: 상단 왼쪽 획
        return '3' if top_r_density > 0.30 else '5'

    # 그 외 쌍은 1위 유지
    return pred


def _hangul_tiebreak(pred, second, pattern_2d):
    """
    한글 모음 혼동 보정.
      ㅏ(가/나...): 세로획 오른쪽에 가로획 → 오른쪽 절반 흰 픽셀 많음
      ㅓ(거/너...): 세로획 왼쪽에 가로획 → 왼쪽 절반 흰 픽셀 많음
      ㅗ(고/노...): 가로획 아래에 세로획 → 하단 중앙에 획 집중
      ㅜ(구/누...): 가로획 위에 세로획 → 상단에 가로획
    반환: 최종 선택 문자
    """
    h, w = pattern_2d.shape
    left  = pattern_2d[:, :w//2]
    right = pattern_2d[:, w//2:]
    top   = pattern_2d[:h//2, :]
    bot   = pattern_2d[h//2:, :]

    # 한글 초성 추출 (마지막 자모 기준으로 모음 추정)
    # 초성이 같으면 모음만 다른 케이스 → 모음 비교
    # 실제로는 전체 글자 단위로 분류하므로 픽셀 분포로만 판단
    pair = tuple(sorted([pred, second]))

    # ㅏ 계열 vs ㅓ 계열 (가/거, 나/너, 다/더 ...)
    A_chars = set('가나다라마바사아자차카타파하')
    E_chars = set('거너더러머버서어저처커터퍼허')
    if pred in A_chars and second in E_chars:
        return pred if np.mean(right == 1) > np.mean(left == 1) else second
    if pred in E_chars and second in A_chars:
        return pred if np.mean(left == 1) > np.mean(right == 1) else second

    # ㅗ 계열 vs ㅜ 계열 (고/구, 노/누, 도/두 ...)
    O_chars = set('고노도로모보소오조초코토포호')
    U_chars = set('구누두루무부수우주추쿠투푸후')
    if pred in O_chars and second in U_chars:
        # ㅗ: 세로획이 가로획 아래 → 하단에 세로획 (하단 중앙 밀도 높음)
        bot_center = pattern_2d[h//2:, w//4:3*w//4]
        top_center = pattern_2d[:h//2, w//4:3*w//4]
        return pred if np.mean(bot_center == 1) > np.mean(top_center == 1) else second
    if pred in U_chars and second in O_chars:
        bot_center = pattern_2d[h//2:, w//4:3*w//4]
        top_center = pattern_2d[:h//2, w//4:3*w//4]
        return pred if np.mean(top_center == 1) > np.mean(bot_center == 1) else second

    return pred  # 그 외 쌍은 1위 유지

# =============================================
# 노이즈 증강
# =============================================
def augment_pattern(pattern, noise_ratio, rng):
    """이진 패턴에 노이즈를 추가해 증강 패턴 생성"""
    p = pattern.copy().astype(np.int8)
    n = len(p)
    n_flip = max(1, int(n * noise_ratio))
    idx = rng.choice(n, n_flip, replace=False)
    p[idx] *= -1
    return p

# =============================================
# 클래스당 다중 medoid 추출
# =============================================
def extract_multi_representative(patterns, n_reps=N_REPS_PER_CLASS, min_plus_ratio=0.03):
    N = len(patterns)
    plus_ratios = np.mean(patterns == 1, axis=1)
    valid_mask = plus_ratios >= min_plus_ratio
    if valid_mask.sum() >= max(3, N * 0.2):
        patterns = patterns[valid_mask]
        N = len(patterns)

    if N == 0:
        return [np.full(CHAR_SIZE * CHAR_SIZE, -1, dtype=np.int8)]
    if N <= n_reps:
        return [p.astype(np.int8) for p in patterns]

    if N > 150:
        idx = np.random.choice(N, 150, replace=False)
        patterns = patterns[idx]
        N = len(patterns)

    # ── [추가] 클래스 중심과 너무 다른 아웃라이어 제거 ──────────
    centroid = np.mean(patterns.astype(float), axis=0)
    centroid_bin = np.where(centroid >= 0, 1, -1).astype(np.int8)
    dists = np.sum(patterns != centroid_bin, axis=1)  # 해밍 거리
    threshold = np.percentile(dists, 80)  # 상위 20% 제거
    clean_mask = dists <= threshold
    if clean_mask.sum() >= max(n_reps, 3):
        patterns = patterns[clean_mask]
        N = len(patterns)
    # ─────────────────────────────────────────────────────────────

    k = min(n_reps, N)
    km = KMeans(n_clusters=k, n_init=5, random_state=42)
    km.fit(patterns.astype(float))

    reps = []
    for c in range(km.n_clusters):
        members = patterns[km.labels_ == c]
        if len(members) == 0:
            continue
        if len(members) == 1:
            reps.append(members[0].astype(np.int8))
            continue
        M = len(members)
        best_idx, best_score = 0, np.inf
        for i in range(M):
            dist = np.sum(members != members[i], axis=1).mean()
            if dist < best_score:
                best_score = dist
                best_idx = i
        reps.append(members[best_idx].astype(np.int8))

    return reps if reps else [patterns[0].astype(np.int8)]

# =============================================
# 홉필드 네트워크
# =============================================
class HopfieldNetwork:
    def __init__(self, name=''):
        self.name = name
        self.n = CHAR_SIZE * CHAR_SIZE
        self.W = np.zeros((self.n, self.n), dtype=np.float64)
        self.rep_pats = []
        self.rep_chs = []

    def train(self, class_arrays, n_reps_per_class=N_REPS_PER_CLASS,
              augment=True, augment_per_rep=AUGMENT_PER_REP):
        """
        augment=True: 각 medoid에 다양한 노이즈를 가한 증강 패턴도 같이 학습.
        끌림 영역(basin of attraction)이 넓어져 완전복원율 향상.
        """
        rng = np.random.default_rng(42)

        all_reps = []
        all_chs  = []
        low_sample_classes = []

        for ch, patterns in sorted(class_arrays.items()):
            n_samples = len(patterns)
            if n_samples < MIN_SAMPLES_WARN:
                low_sample_classes.append((ch, n_samples))

            medoids = extract_multi_representative(patterns, n_reps=n_reps_per_class)

            for med in medoids:
                # 원본 medoid 저장
                all_reps.append(med)
                all_chs.append(ch)

                # 노이즈 증강 패턴 추가 (augment=True일 때만)
                if augment:
                    for noise_ratio in AUGMENT_NOISE_LEVELS[:augment_per_rep]:
                        aug = augment_pattern(med, noise_ratio, rng)
                        all_reps.append(aug)
                        all_chs.append(ch)

        # 데이터 부족 클래스 경고
        if low_sample_classes:
            print(f"  [{self.name}] ⚠ 샘플 부족 클래스 ({MIN_SAMPLES_WARN}개 미만):")
            for ch, n in low_sample_classes:
                print(f"      '{ch}': {n}개 — 증강으로 보완했으나 인식률 낮을 수 있음")

        self.rep_pats = all_reps
        self.rep_chs  = all_chs

        X = np.stack(all_reps).astype(np.float64)
        K_total = len(all_reps)
        n_classes = len(class_arrays)

        # Pseudo-inverse(Projection) 학습 규칙
        XXt = X @ X.T
        XXt_inv = np.linalg.pinv(XXt)
        W = X.T @ XXt_inv @ X
        np.fill_diagonal(W, 0)
        self.W = W

        cap = int(self.n / (2 * np.log(self.n)))
        aug_note = f", 증강 포함" if augment else ""
        print(f"  [{self.name}] 클래스 {n_classes}개, 총 저장 패턴 {K_total}개"
              f"{aug_note} (Hebbian 이론용량 참고치: ~{cap}개)")

    def recall(self, pattern, max_sweeps=MAX_SWEEPS, seed=None):
        """비동기식 업데이트 — 시각화 보조용"""
        rng = np.random.default_rng(seed)
        state = np.where(np.asarray(pattern, dtype=np.float64).flatten() >= 0, 1.0, -1.0)

        for sweep in range(max_sweeps):
            order = rng.permutation(self.n)
            changed = False
            for idx in order:
                h = self.W[idx] @ state
                new_val = 1.0 if h >= 0 else -1.0
                if new_val != state[idx]:
                    state[idx] = new_val
                    changed = True
            if not changed:
                return state.astype(np.int8), sweep + 1
        return state.astype(np.int8), max_sweeps

    def classify(self, pattern, debug=False):
        """
        코사인 유사도 기반 분류 — 클래스당 여러 대표 중 최댓값.
        [보완 3] 1위/2위 마진이 TIEBREAK_MARGIN 이하이면 획 구조 tie-break 발동.
        debug=True: 1~3위 유사도와 tie-break 발동 여부를 출력.
        """
        p_f = np.asarray(pattern, dtype=np.float64).flatten()
        p_norm = np.linalg.norm(p_f)

        # 클래스별 최대 코사인 유사도 계산
        class_best = {}
        for rep, ch in zip(self.rep_pats, self.rep_chs):
            r_f = rep.astype(float)
            denom = p_norm * np.linalg.norm(r_f)
            sim = np.dot(p_f, r_f) / denom if denom > 0 else 0.0
            if ch not in class_best or sim > class_best[ch]:
                class_best[ch] = sim

        # 유사도 내림차순 정렬
        sorted_chs = sorted(class_best, key=class_best.get, reverse=True)
        best_ch  = sorted_chs[0]
        best_sim = class_best[best_ch]

        # Tie-break: 2위가 존재하고 마진이 임계값 이하일 때만 발동
        if len(sorted_chs) >= 2:
            second_ch  = sorted_chs[1]
            second_sim = class_best[second_ch]
            margin = best_sim - second_sim

            DIGITS = set('0123456789')
            is_digit_pair = best_ch in DIGITS and second_ch in DIGITS
            tiebreak_margin = TIEBREAK_MARGIN_DIGIT if is_digit_pair else TIEBREAK_MARGIN_HANGUL

            if debug:
                print(f"  [DEBUG] 1위={best_ch}({best_sim:.4f}) "
                      f"2위={second_ch}({second_sim:.4f}) "
                      f"margin={margin:.4f} "
                      f"{'→tie-break발동' if margin <= tiebreak_margin else '→tie-break미발동'}"
                      f"({'digit' if is_digit_pair else 'hangul'} threshold={tiebreak_margin})")

            if margin <= tiebreak_margin:
                pattern_2d = np.asarray(pattern, dtype=np.int8).reshape(CHAR_SIZE, CHAR_SIZE)

                # 숫자 혼동 쌍 (0~9 범위 문자)
                if is_digit_pair:
                    prev = best_ch
                    best_ch = _digit_tiebreak(best_ch, second_ch, pattern_2d)
                    if debug and best_ch != prev:
                        print(f"  [DEBUG] tie-break 교체: {prev}→{best_ch}")

                # 한글 혼동 쌍
                else:
                    prev = best_ch
                    best_ch = _hangul_tiebreak(best_ch, second_ch, pattern_2d)
                    if debug and best_ch != prev:
                        print(f"  [DEBUG] tie-break 교체: {prev}→{best_ch}")

                best_sim = class_best[best_ch]

        elif debug:
            print(f"  [DEBUG] 1위={best_ch}({best_sim:.4f}) 후보1개뿐")

        return best_ch, best_sim

    def classify_direct(self, pattern):
        """[핵심] recall 없이 직접 분류 (tie-break 포함)"""
        return self.classify(pattern)

    def save(self, path):
        data = {
            'W': self.W,
            'rep_pats': np.stack(self.rep_pats),
            'rep_chs': np.array(self.rep_chs, dtype=object)
        }
        np.save(path, data)

    def load(self, path):
        data = np.load(path, allow_pickle=True).item()
        self.W = data['W']
        self.rep_pats = list(data['rep_pats'])
        self.rep_chs = list(data['rep_chs'])