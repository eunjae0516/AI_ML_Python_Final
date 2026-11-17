"""
p1_3_compare_and_refine.py

Raw BlazePose 데이터 / 제공된 후처리(target) / 추가 후처리(refined)를 비교한다.

  1) Problem Identification : raw 데이터의 문제를 수치·그래프로 확인
  2) Results & Analysis     : 후처리 전후 비교 (metrics.csv + 그래프)
  3) Bonus                  : 추가 후처리 (누락 프레임 보간 + 저역통과 스무딩 + clamp 없는 지면 보정)

실행:
  python scripts/p1_3_compare_and_refine.py          # VIDEO_NAME 영상 분석
  python scripts/p1_3_compare_and_refine.py --vis    # + 제공 후처리(왼쪽) vs refined(오른쪽) 3D 재생
  python scripts/p1_3_compare_and_refine.py --all    # 모든 영상 분석

출력:
  results/<영상>_compare.png, results/summary_jitter.png, results/metrics.csv
  data/processed/<영상>_refined.npy, data/processed/<영상>_refined.csv (Unity용, 30fps)
"""

import os
import sys
import argparse
from glob import glob

import cv2
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 제공된 후처리 방법을 그대로 사용 (비교 기준)
from p1_2_post_processing import create_height_corrected_target, export_unity_csv

RAW_KEYPOINTS_DIR = os.path.join(ROOT, "data", "raw_keypoints")
RAW_VIDEO_DIR = os.path.join(ROOT, "data", "raw_videos")
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
RESULT_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

plt.rcParams["font.family"] = "Malgun Gothic"   # 한글 제목
plt.rcParams["axes.unicode_minus"] = False

FOOT_JOINTS = [29, 30, 31, 32]          # heel L/R, foot index L/R
L_HIP, R_HIP, R_WRIST = 23, 24, 16
BONES = [(11, 13), (13, 15), (12, 14), (14, 16),   # arms
         (23, 25), (25, 27), (24, 26), (26, 28),   # legs
         (11, 12), (23, 24)]                       # shoulders, hips
UNITY_FPS = 30

VIDEO_NAME = "L을_가져가_~__dance_trend"   # 분석 대상 영상 (data/raw_keypoints/<이름>.npz)


# ------------------------------------------------------------
# Loading
# ------------------------------------------------------------
def load_raw_with_frames(npz_path):
    """raw npz → (프레임 번호 배열, (T,33,3)). 사람이 검출되지 않은 프레임은 빠져 있음."""
    data = np.load(npz_path, allow_pickle=True)["data"].astype(float)
    frames = data[:, 0].astype(int)
    landmarks = data[:, 1].astype(int)
    uniq = np.unique(frames)
    seq = np.zeros((len(uniq), 33, 3))
    seq[np.searchsorted(uniq, frames), landmarks] = data[:, 2:5]
    return uniq, seq


def video_info(name, frames):
    cap = cv2.VideoCapture(os.path.join(RAW_VIDEO_DIR, f"{name}.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or int(frames[-1]) + 1
    cap.release()
    return fps, n


# ------------------------------------------------------------
# Additional post-processing (bonus)
# ------------------------------------------------------------
def fill_missing_frames(frames, seq):
    """검출 실패로 빠진 중간 프레임을 선형 보간 → 시간축 복원."""
    full = np.arange(frames[0], frames[-1] + 1)
    out = np.empty((len(full), 33, 3))
    for j in range(33):
        for k in range(3):
            out[:, j, k] = np.interp(full, frames, seq[:, j, k])
    interpolated = ~np.isin(full, frames)
    return full, out, interpolated


def lowpass(seq, fps, cutoff=6.0, order=4):
    """Zero-phase Butterworth 저역통과 필터 (위상 지연 없음)."""
    b, a = butter(order, cutoff / (fps / 2.0))
    return filtfilt(b, a, seq, axis=0)


def ground_per_frame(seq):
    """
    BlazePose world 좌표는 골반이 항상 원점 → 몸 전체의 상하 이동 정보가 없다.
    제공된 방법처럼 '매 프레임 가장 낮은 발이 지면에 닿는다'고 가정해 상하 이동을 복원하되,
      - 발가락(31,32)만이 아니라 뒤꿈치(29,30)까지 포함해 가장 낮은 점을 지면으로 사용
      - clamp(y<0 → 0)를 하지 않아 관절이 바닥에 납작하게 눌리는 왜곡을 없앰
      - 스무딩된 데이터에 적용하므로 골반 높이가 프레임마다 튀지 않음
    """
    out = seq.copy()
    out[:, :, 1] -= out[:, FOOT_JOINTS, 1].min(axis=1)[:, None]
    return out


def refine(frames, seq, fps, cutoff):
    full, filled, interp_mask = fill_missing_frames(frames, seq)
    smoothed = lowpass(filled, fps, cutoff)
    return full, ground_per_frame(smoothed), interp_mask


def to_unity_fps(seq, fps):
    step = max(1, int(round(fps / UNITY_FPS)))   # 60fps → 2프레임마다 1개 (스무딩이 anti-alias 역할)
    return seq[::step]


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------
def lowest_foot(seq):
    return seq[:, FOOT_JOINTS, 1].min(axis=1)


def pelvis_y(seq):
    return seq[:, [L_HIP, R_HIP], 1].mean(axis=1)


def metrics(seq, fps):
    accel = np.linalg.norm(np.diff(seq, 2, axis=0), axis=2) * fps ** 2           # m/s^2
    bone = np.stack([np.linalg.norm(seq[:, a] - seq[:, b], axis=1) for a, b in BONES], 1)
    foot = lowest_foot(seq)
    return {
        "jitter_mean_accel(m/s2)": accel.mean(),
        "bone_length_CV(%)": (bone.std(0) / bone.mean(0)).mean() * 100,
        "lowest_foot_mean(m)": foot.mean(),
        "lowest_foot_std(m)": foot.std(),
        "below_ground_frames(%)": (foot < -0.02).mean() * 100,
        "pelvis_height_range(m)": np.ptp(pelvis_y(seq)),
        # 지면 기준점(가장 낮은 관절 1개)을 제외하고 y=0에 '눌려 붙은' 관절 비율 → clamp 왜곡
        "joints_flattened_to_y0(%)": (np.clip((seq[:, :, 1] == 0).sum(1) - 1, 0, None) / 32).mean() * 100,
    }


# ------------------------------------------------------------
# Plots
# ------------------------------------------------------------
def shade(ax, t, mask):
    if mask.any():
        ax.fill_between(t, 0, 1, where=mask, transform=ax.get_xaxis_transform(),
                        color="grey", alpha=0.25, label="보간된 프레임 (검출 실패)")


def plot_compare(name, fps, frames, raw, target, full, refined, interp_mask):
    t_raw = frames / fps
    t_ref = full / fps
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    ax = axes[0]
    ax.plot(t_raw, lowest_foot(raw), color="tab:red", lw=1)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_title("① Raw: 가장 낮은 발의 높이 — 골반 원점 좌표라 지면(y=0) 아래에 있고 프레임마다 흔들림")
    ax.set_ylabel("y (m)")

    ax = axes[1]
    ax.plot(t_raw, pelvis_y(target), color="tab:orange", lw=1, label="제공 후처리 (프레임별 지면 + clamp)")
    ax.plot(t_ref, pelvis_y(refined), color="tab:green", lw=1.5, label="추가 후처리 (보간+스무딩+지면, clamp 없음)")
    shade(ax, t_ref, interp_mask)
    ax.set_title("② 골반 높이 — 두 방법 모두 상하 리듬을 복원, 제공 방법에는 raw의 잔떨림이 남음")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[2]
    v_raw = np.linalg.norm(np.diff(raw[:, R_WRIST], axis=0), axis=1) * fps
    v_ref = np.linalg.norm(np.diff(refined[:, R_WRIST], axis=0), axis=1) * fps
    ax.plot(t_raw[1:], v_raw, color="tab:red", lw=0.8, alpha=0.8, label="Raw")
    ax.plot(t_ref[1:], v_ref, color="tab:green", lw=1.5, label="추가 후처리")
    shade(ax, t_ref, interp_mask)
    ax.set_title("③ 오른손목 속도 — Raw의 뾰족한 값은 떨림(jitter), 스무딩 후 제거")
    ax.set_ylabel("speed (m/s)")
    ax.set_xlabel("time (s)")
    ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(f"{name}  ({fps:.0f} fps)", fontsize=13)
    fig.tight_layout()
    out = os.path.join(RESULT_DIR, f"{name}_compare.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_summary(df):
    pivot = df.pivot(index="video", columns="version", values="jitter_mean_accel(m/s2)")
    pivot = pivot[["raw", "provided", "refined"]]
    ax = pivot.plot.bar(figsize=(8, 4.5), color=["tab:red", "tab:orange", "tab:green"], rot=0)
    ax.set_ylabel("평균 가속도 크기 (m/s²) — 낮을수록 떨림 적음")
    ax.set_title("후처리 방법별 떨림(jitter) 비교")
    plt.tight_layout()
    out = os.path.join(RESULT_DIR, "summary_jitter.png")
    plt.savefig(out, dpi=130)
    plt.close()
    return out


# ------------------------------------------------------------
def process(name, cutoff):
    frames, raw = load_raw_with_frames(os.path.join(RAW_KEYPOINTS_DIR, f"{name}.npz"))
    fps, n_video = video_info(name, frames)
    target = create_height_corrected_target(raw)                  # 제공된 후처리
    full, refined, interp_mask = refine(frames, raw, fps, cutoff)  # 추가 후처리

    np.save(os.path.join(PROCESSED_DIR, f"{name}_refined.npy"), refined)
    export_unity_csv(to_unity_fps(refined, fps), os.path.join(PROCESSED_DIR, f"{name}_refined.csv"))
    fig = plot_compare(name, fps, frames, raw, target, full, refined, interp_mask)

    gaps = np.diff(frames) - 1
    info = {"video": name, "fps": round(fps), "video_frames": n_video,
            "detected_frames": len(frames), "missing_frames": n_video - len(frames),
            "longest_gap(frames)": int(gaps.max()) if len(gaps) else 0}
    rows = [{**info, "version": v, **metrics(s, fps)}
            for v, s in [("raw", raw), ("provided", target), ("refined", refined)]]
    print(f"✅ {name}: {fig}")
    return rows, (frames, target, full, refined, fps)


def visualize(frames, target, full, refined, fps):
    from utils.viser_test import PoseViser
    # 제공 후처리는 누락 프레임이 없으므로 직전 프레임을 유지해 같은 시간축에 맞춤 (끊김이 '멈춤'으로 보임)
    idx = np.searchsorted(frames, full, side="right") - 1
    vis = PoseViser(fps=fps)
    vis.play_two_sequences(target[idx], refined, offset=1.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", type=float, default=6.0, help="저역통과 차단 주파수(Hz)")
    parser.add_argument("--video", type=str, default=VIDEO_NAME, help="분석할 영상 이름")
    parser.add_argument("--all", action="store_true", help="raw_keypoints의 모든 영상 분석")
    parser.add_argument("--vis", action="store_true", help="제공 후처리 vs refined 3D 비교 재생")
    args = parser.parse_args()

    if args.all:
        names = [os.path.basename(p)[:-4] for p in sorted(glob(os.path.join(RAW_KEYPOINTS_DIR, "*.npz")))]
    else:
        names = [args.video]

    all_rows, cache = [], {}
    for name in names:
        rows, cache[name] = process(name, args.cutoff)
        all_rows += rows

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(RESULT_DIR, "metrics.csv"), index=False, encoding="utf-8-sig")
    plot_summary(df)

    cols = ["video", "version", "jitter_mean_accel(m/s2)", "bone_length_CV(%)",
            "lowest_foot_std(m)", "below_ground_frames(%)", "pelvis_height_range(m)",
            "joints_flattened_to_y0(%)"]
    with pd.option_context("display.width", 200, "display.max_columns", None,
                           "display.float_format", "{:.3f}".format):
        print(df[["video", "fps", "video_frames", "detected_frames", "missing_frames",
                  "longest_gap(frames)"]].drop_duplicates().to_string(index=False))
        print()
        print(df[cols].to_string(index=False))
    print(f"\n📁 결과 저장 → {RESULT_DIR}")

    if args.vis:
        visualize(*cache[names[0]])


if __name__ == "__main__":
    main()
