"""
Episode Labeler — frame-by-frame pickup/drop annotation tool.

Two cv2 trackbars let you scrub episodes and frames directly.
Frames load in a background thread so the UI stays responsive.
Auto-detected pickup/drop from the gripper signal are pre-loaded as suggestions.

Controls
--------
  Drag "Frame" trackbar    scrub to any frame
  Drag "Episode" trackbar  jump to any episode
  LEFT / RIGHT             ±1 frame
  , / .                    ±10 frames
  SPACE                    play / pause
  g                        jump to suggested pickup frame
  h                        jump to suggested drop frame
  1                        mark current frame as PICKUP
  2                        mark current frame as DROP
  c                        clear both marks for this episode
  s / Enter                save all labels
  q / Esc                  save and quit
"""

import threading
from pathlib import Path

import av
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── config ────────────────────────────────────────────────────────────────────
ORIG_ROOT = Path(
    r"C:\Users\calle\.cache\huggingface\hub"
    r"\datasets--nc8304--so101_combined_cubeONLY"
    r"\snapshots\acc242c231f60171a5b2833442d176cd793ea8c9"
)
VIDEO_KEY      = "observation.images.front"
FPS            = 30.0
GRIPPER_IDX    = 5
GRIP_THRESHOLD = 20.0
AUTO_PHASES    = Path(__file__).parent.parent / "outputs" / "episode_phases.parquet"
OUT_PATH       = Path(__file__).parent.parent / "outputs" / "episode_phases_manual.parquet"

FRAME_W, FRAME_H = 640, 480
PLOT_H           = 240
WIN_H            = FRAME_H + PLOT_H

C_PICKUP = (0,   220,  64)   # BGR green
C_DROP   = (60,  60,  230)   # BGR red/blue
C_CURR   = (0,   220, 220)   # BGR yellow-cyan


# ── background frame loader ───────────────────────────────────────────────────

class FrameLoader:
    """Decodes episode frames in a background thread."""

    def __init__(self):
        self._lock   = threading.Lock()
        self._frames : list[np.ndarray] = []
        self._active = False
        self._ep_tag = -1
        self._thread : threading.Thread | None = None

    def start(self, vid_path: Path, from_ts: float, n_frames: int, ep_tag: int):
        self._active = False                      # signals old thread to stop
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        with self._lock:
            self._frames = []
            self._ep_tag = ep_tag
        self._active = True
        self._thread = threading.Thread(
            target=self._run, args=(vid_path, from_ts, n_frames, ep_tag), daemon=True)
        self._thread.start()

    def _run(self, vid_path: Path, from_ts: float, n_frames: int, ep_tag: int):
        try:
            with av.open(str(vid_path)) as c:
                stream = c.streams.video[0]
                # seek: av.time_base = Fraction(1, 1_000_000)
                c.seek(int(from_ts * 1_000_000))
                count = 0
                for pkt in c.demux(stream):
                    if not self._active or self._ep_tag != ep_tag:
                        return
                    try:
                        for frame in pkt.decode():
                            if count >= n_frames:
                                break
                            img = frame.to_ndarray(format="bgr24")
                            if img.shape[:2] != (FRAME_H, FRAME_W):
                                img = cv2.resize(img, (FRAME_W, FRAME_H))
                            with self._lock:
                                self._frames.append(img)
                            count += 1
                    except av.error.InvalidDataError:
                        pass
                    if count >= n_frames:
                        break
        finally:
            self._active = False

    def get(self, idx: int) -> np.ndarray | None:
        with self._lock:
            if not self._frames:
                return None
            return self._frames[min(idx, len(self._frames) - 1)]

    @property
    def n_ready(self) -> int:
        with self._lock:
            return len(self._frames)

    @property
    def loading(self) -> bool:
        return self._active


# ── data helpers ──────────────────────────────────────────────────────────────

def load_meta():
    ep_df   = pd.read_parquet(ORIG_ROOT / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    data_df = pd.read_parquet(ORIG_ROOT / "data" / "chunk-000" / "file-000.parquet")
    return ep_df, data_df


def episode_info(ep_df: pd.DataFrame, ep_idx: int):
    row      = ep_df[ep_df["episode_index"] == ep_idx].iloc[0]
    file_idx = int(row[f"videos/{VIDEO_KEY}/file_index"])
    from_ts  = float(row[f"videos/{VIDEO_KEY}/from_timestamp"])
    to_ts    = float(row[f"videos/{VIDEO_KEY}/to_timestamp"])
    vid_path = ORIG_ROOT / "videos" / VIDEO_KEY / "chunk-000" / f"file-{file_idx:03d}.mp4"
    n_frames = max(1, int(round((to_ts - from_ts) * FPS)))
    return vid_path, from_ts, n_frames


def gripper_signal(data_df: pd.DataFrame, ep_idx: int) -> np.ndarray:
    ep = data_df[data_df["episode_index"] == ep_idx].sort_values("frame_index")
    return np.stack(ep["observation.state"].values)[:, GRIPPER_IDX]


def auto_detect(g: np.ndarray):
    """
    Enumerate all open→close→open cycles, return the one with the longest hold.
    Handles episodes where the robot makes multiple gripping attempts.
    """
    n    = len(g)
    thr  = GRIP_THRESHOLD
    ups  = [i for i in range(1, n) if g[i - 1] < thr <= g[i]]
    downs= [i for i in range(1, n) if g[i - 1] >= thr > g[i]]

    if not ups or not downs:
        return None, None

    best_pickup = best_drop = None
    best_hold   = -1

    for down in downs:
        if not any(u < down for u in ups):
            continue
        next_ups = [u for u in ups if u > down]
        if not next_ups:
            continue
        drop = next_ups[0]
        hold = drop - down
        if hold > best_hold:
            best_hold   = hold
            best_pickup = down
            best_drop   = drop

    return best_pickup, best_drop


# ── rendering ─────────────────────────────────────────────────────────────────

def build_base_plot(gripper: np.ndarray, pickup, drop, auto_pk, auto_dr):
    """
    Render the static gripper plot (no current-frame line).
    Returns (bgr_image, frame_to_x_fn) where frame_to_x_fn maps a local
    frame index to an x-pixel coordinate in the image.
    """
    dpi = 120
    fig, ax = plt.subplots(figsize=(FRAME_W / dpi, PLOT_H / dpi), dpi=dpi)
    t = np.arange(len(gripper)) / FPS

    ax.plot(t, gripper, color="steelblue", linewidth=1.2, zorder=2, label="gripper°")
    ax.axhline(GRIP_THRESHOLD, color="orange", linestyle="--", linewidth=1.0,
               alpha=0.8, label=f"threshold={GRIP_THRESHOLD}")

    # auto suggestions — dashed
    if auto_pk is not None:
        ax.axvline(auto_pk / FPS, color="#22cc22", linewidth=1.5, linestyle="--", alpha=0.8,
                   label=f"auto pickup f{auto_pk}")
    if auto_dr is not None:
        ax.axvline(auto_dr / FPS, color="#cc2222", linewidth=1.5, linestyle="--", alpha=0.8,
                   label=f"auto drop f{auto_dr}")

    # manual marks — solid, thick
    if pickup is not None:
        ax.axvline(pickup / FPS, color="#00ff44", linewidth=2.5, zorder=3,
                   label=f"PICKUP = {pickup}")
    if drop is not None:
        ax.axvline(drop / FPS, color="#4466ff", linewidth=2.5, zorder=3,
                   label=f"DROP = {drop}")

    ax.set_xlim(0, max(1.0, (len(gripper) - 1) / FPS))
    ax.set_ylim(max(0.0, float(gripper.min()) - 2), float(gripper.max()) + 3)
    ax.set_xlabel("time (s)", fontsize=8)
    ax.set_ylabel("gripper°", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.85)
    ax.grid(True, alpha=0.25)
    fig.tight_layout(pad=0.4)
    fig.canvas.draw()

    # data→pixel transform for the time axis
    x0 = int(ax.transData.transform((0, 0))[0])
    x1 = int(ax.transData.transform(((len(gripper) - 1) / FPS, 0))[0])

    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    bgr = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
    plt.close(fig)

    if bgr.shape != (PLOT_H, FRAME_W, 3):
        scale = FRAME_W / bgr.shape[1]
        x0 = int(x0 * scale)
        x1 = int(x1 * scale)
        bgr = cv2.resize(bgr, (FRAME_W, PLOT_H))

    n = len(gripper)
    def frame_to_x(f: int) -> int:
        frac = f / max(n - 1, 1)
        return int(np.clip(x0 + (x1 - x0) * frac, 1, FRAME_W - 2))

    return bgr, frame_to_x


def overlay_video(frame: np.ndarray, ep_idx: int, n_ep: int,
                  local_frame: int, n_frames: int, n_ready: int,
                  pickup, drop, playing: bool) -> np.ndarray:
    img = frame.copy()
    h, w = img.shape[:2]

    # coloured border when at a marked frame
    if local_frame == pickup:
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), C_PICKUP, 6)
    elif local_frame == drop:
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), C_DROP, 6)

    # header bar
    cv2.rectangle(img, (0, 0), (w, 54), (0, 0, 0), -1)

    # episode + frame info
    loading_str = f"  (loading {n_ready}/{n_frames})" if n_ready < n_frames else ""
    cv2.putText(img,
                f"Ep {ep_idx} / {n_ep - 1}{loading_str}   "
                f"frame {local_frame} / {n_frames - 1}  ({local_frame / FPS:.2f}s)"
                f"{'  ▶ PLAYING' if playing else ''}",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    pk_str = f"PICKUP = {pickup}" if pickup is not None else "PICKUP = --"
    dr_str = f"DROP = {drop}"     if drop   is not None else "DROP = --"
    cv2.putText(img, pk_str, (8,   42), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                C_PICKUP if pickup is not None else (100, 100, 100), 1, cv2.LINE_AA)
    cv2.putText(img, dr_str, (210, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                C_DROP   if drop   is not None else (100, 100, 100), 1, cv2.LINE_AA)

    # footer bar
    cv2.rectangle(img, (0, h - 22), (w, h), (0, 0, 0), -1)
    cv2.putText(img,
                "1=PICKUP  2=DROP  g/h=jump  SPACE=play  c=clear  s=save  q=quit",
                (6, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (170, 170, 170), 1, cv2.LINE_AA)
    return img


def loading_placeholder(n_ready: int, n_frames: int, ep_idx: int) -> np.ndarray:
    img = np.full((FRAME_H, FRAME_W, 3), 30, dtype=np.uint8)
    pct = n_ready / max(n_frames, 1)
    bar_w = int((FRAME_W - 60) * pct)
    cv2.rectangle(img, (30, FRAME_H // 2 - 8), (30 + bar_w, FRAME_H // 2 + 8), (80, 160, 80), -1)
    cv2.rectangle(img, (30, FRAME_H // 2 - 8), (FRAME_W - 30, FRAME_H // 2 + 8), (120, 120, 120), 1)
    cv2.putText(img, f"Loading ep {ep_idx}: {n_ready}/{n_frames} frames ({int(pct*100)}%)",
                (30, FRAME_H // 2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    return img


# ── save ──────────────────────────────────────────────────────────────────────

def save_labels(labels: dict):
    rows = []
    for ep_idx, d in sorted(labels.items()):
        pf, df_ = d.get("pickup"), d.get("drop")
        rows.append(dict(
            episode_index = ep_idx,
            pickup_frame  = pf,
            drop_frame    = df_,
            pickup_time_s = round(pf  / FPS, 3) if pf  is not None else None,
            drop_time_s   = round(df_ / FPS, 3) if df_ is not None else None,
        ))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(OUT_PATH, index=False)
    done = sum(1 for r in rows if r["pickup_frame"] is not None and r["drop_frame"] is not None)
    print(f"Saved — {done}/{len(rows)} episodes fully labelled → {OUT_PATH}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading dataset metadata...")
    ep_df, data_df = load_meta()
    episodes = sorted(ep_df["episode_index"].unique().tolist())
    n_ep = len(episodes)

    # auto-suggestions
    auto: dict[int, tuple] = {}
    if AUTO_PHASES.exists():
        ap = pd.read_parquet(AUTO_PHASES)
        for _, row in ap.iterrows():
            pf  = None if pd.isna(row["pickup_frame"]) else int(row["pickup_frame"])
            df_ = None if pd.isna(row["drop_frame"])   else int(row["drop_frame"])
            auto[int(row["episode_index"])] = (pf, df_)
    else:
        print("Running auto-detection from gripper signal...")
        for ei in episodes:
            auto[ei] = auto_detect(gripper_signal(data_df, ei))

    # labels — start from auto
    labels: dict[int, dict] = {
        ei: {"pickup": auto.get(ei, (None, None))[0],
             "drop":   auto.get(ei, (None, None))[1]}
        for ei in episodes
    }

    # ── window + trackbars ────────────────────────────────────────────────────
    WIN = "Episode Labeler"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, FRAME_W, WIN_H + 60)   # +60 for two trackbars

    # trackbar state (written by callbacks, read in main loop)
    tb_state = {"ep": 0, "frame": 0}

    def on_ep_tb(val):
        tb_state["ep"] = val
        tb_state["frame"] = 0          # reset frame when episode changes

    def on_frame_tb(val):
        tb_state["frame"] = val

    cv2.createTrackbar("Episode", WIN, 0, n_ep - 1, on_ep_tb)
    cv2.createTrackbar("Frame",   WIN, 0, 1,         on_frame_tb)  # max updated per episode

    # ── episode state ─────────────────────────────────────────────────────────
    loader     = FrameLoader()
    gripper    = np.zeros(1)
    base_plot  = np.zeros((PLOT_H, FRAME_W, 3), dtype=np.uint8)
    frame_to_x = lambda f: FRAME_W // 2

    cur_ep_pos  = -1      # index into episodes[], -1 forces initial load
    n_frames    = 1
    local_frame = 0
    playing     = False

    def load_episode(pos: int, jump_to_pickup: bool = True):
        nonlocal gripper, base_plot, frame_to_x, n_frames, local_frame, playing
        playing = False
        ei = episodes[pos]
        vid_path, from_ts, nf = episode_info(ep_df, ei)
        n_frames = nf

        # update frame trackbar range
        cv2.setTrackbarMax("Frame", WIN, max(nf - 1, 1))

        # gripper + base plot (fast — no video needed)
        gripper = gripper_signal(data_df, ei)
        pk, dr  = labels[ei]["pickup"], labels[ei]["drop"]
        apk, adr = auto.get(ei, (None, None))
        base_plot, frame_to_x = build_base_plot(gripper, pk, dr, apk, adr)

        # start background video load
        loader.start(vid_path, from_ts, nf, ei)

        # jump to pickup frame (or 0)
        if jump_to_pickup and pk is not None:
            local_frame = pk
        else:
            local_frame = 0
        cv2.setTrackbarPos("Frame", WIN, local_frame)

    def refresh_plot():
        nonlocal base_plot, frame_to_x
        ei = episodes[cur_ep_pos]
        pk, dr   = labels[ei]["pickup"], labels[ei]["drop"]
        apk, adr = auto.get(ei, (None, None))
        base_plot, frame_to_x = build_base_plot(gripper, pk, dr, apk, adr)

    # ── main loop ─────────────────────────────────────────────────────────────
    while True:
        # ── detect episode change (trackbar or keyboard) ──────────────────────
        tb_ep = tb_state["ep"]
        if tb_ep != cur_ep_pos:
            cur_ep_pos = tb_ep
            load_episode(cur_ep_pos)

        ei       = episodes[cur_ep_pos]
        n_ready  = loader.n_ready

        # ── sync frame trackbar → local_frame (trackbar is authoritative when idle) ──
        if not playing:
            tb_frame = tb_state["frame"]
            if tb_frame != local_frame:
                local_frame = tb_frame

        local_frame = int(np.clip(local_frame, 0, n_frames - 1))

        # ── build composite image ─────────────────────────────────────────────
        raw = loader.get(local_frame)
        if raw is None:
            vid_img = loading_placeholder(n_ready, n_frames, ei)
        else:
            vid_img = overlay_video(raw, ei, n_ep, local_frame, n_frames, n_ready,
                                    labels[ei]["pickup"], labels[ei]["drop"], playing)

        # gripper plot: copy base + current-frame vertical line
        plot_img = base_plot.copy()
        cx = frame_to_x(local_frame)
        cv2.line(plot_img, (cx, 0), (cx, PLOT_H), C_CURR, 1)

        cv2.imshow(WIN, np.vstack([vid_img, plot_img]))

        # ── playback ──────────────────────────────────────────────────────────
        wait_ms = 33 if playing else 15
        key = cv2.waitKeyEx(wait_ms)

        if playing:
            next_f = local_frame + 1
            if next_f >= n_ready:          # wait for more frames or stop at end
                if not loader.loading:
                    playing = False
            else:
                local_frame = next_f
                cv2.setTrackbarPos("Frame", WIN, local_frame)
            if key == ord(" "):
                playing = False
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break
            continue

        if key == -1:
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break
            continue

        # ── keyboard handling ─────────────────────────────────────────────────

        # navigation
        if key in (2424832, 65361):           # LEFT
            local_frame = max(local_frame - 1, 0)
            cv2.setTrackbarPos("Frame", WIN, local_frame)
        elif key in (2555904, 65363):         # RIGHT
            local_frame = min(local_frame + 1, n_frames - 1)
            cv2.setTrackbarPos("Frame", WIN, local_frame)
        elif key == ord(","):
            local_frame = max(local_frame - 10, 0)
            cv2.setTrackbarPos("Frame", WIN, local_frame)
        elif key == ord("."):
            local_frame = min(local_frame + 10, n_frames - 1)
            cv2.setTrackbarPos("Frame", WIN, local_frame)
        elif key == ord(" "):
            playing = True

        # jump to suggestions
        elif key == ord("g"):
            apk, _ = auto.get(ei, (None, None))
            if apk is not None:
                local_frame = apk
                cv2.setTrackbarPos("Frame", WIN, local_frame)
        elif key == ord("h"):
            _, adr = auto.get(ei, (None, None))
            if adr is not None:
                local_frame = adr
                cv2.setTrackbarPos("Frame", WIN, local_frame)

        # marking
        elif key == ord("1"):
            labels[ei]["pickup"] = local_frame
            refresh_plot()
            print(f"  ep {ei}: PICKUP = {local_frame}")
        elif key == ord("2"):
            labels[ei]["drop"] = local_frame
            refresh_plot()
            print(f"  ep {ei}: DROP   = {local_frame}")
        elif key == ord("c"):
            labels[ei] = {"pickup": None, "drop": None}
            refresh_plot()
            print(f"  ep {ei}: cleared")

        # save
        elif key in (ord("s"), 13):           # s or Enter
            save_labels(labels)

        # quit
        elif key in (ord("q"), 27):           # q or Esc
            break

    cv2.destroyAllWindows()
    save_labels(labels)
    print("Done.")


if __name__ == "__main__":
    main()
