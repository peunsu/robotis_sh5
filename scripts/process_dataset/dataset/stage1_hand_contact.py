"""1단계(hand pretrain) 롤아웃 → 32링크 접촉 맵 (hand_contact_stage1.npz).  [stage1-contact-map]

사람 접촉 맵(parahome_hand_contact.py)과 같은 방식·같은 코어(frame_contacts)로 만든다:
  물체 정점마다 가장 가까운 손 표면 점을 찾아 gamma 안이면 접촉 → FPS 로 num_contacts 개 이하 → 링크별 평균 위치·평균 법선.
차이는 손 표면 점의 출처만이다. SMPL-X 정점 대신 Shadow 링크 시각 메시(URDF <visual>, origin 적용) 정점을,
롤아웃에 기록된 시뮬레이션 링크 자세(link_pos/link_quat; rollout.py --dump_hand_traj)로 월드에 놓는다.
FK 를 하지 않으므로 시뮬레이션과 링크 자세가 정확히 같다. 물체 자세는 손이 실제로 닿은 1단계 물체(obj_pos/obj_quat).

출력(기본: 입력과 같은 폴더의 hand_contact_stage1.npz; 스테이지 2 env 의 hand_pretrain_contact_map 이 읽는 스키마):
  link_names (L,), mask (T,L), normal (T,L,3) 물체 로컬 바깥 표면 법선, target (T,L,3) 물체 로컬 (coord="object"),
  target_world (T,L,3) 참고용, frame (T,) 롤아웃 프레임, control_fps, normal_source, obj_pose_source="stage1",
  gamma, num_contacts, source_hand_traj, urdf.

검증 출력:
  (1) pad 검사: 롤아웃의 손끝 pad 위치(ft_pos, 시뮬레이션)가 배치된 distal 메시의 표면에서 몇 mm 인지.
      URDF 링크 프레임 = 시뮬레이션 바디 프레임인지 확인하는 검사이며, 중앙값이 --pad-check-max-mm 를 넘으면 실패(exit 3).
  (2) 링크별 접촉 프레임 수와 1단계 접촉력(>1 N) 대비 IoU.
  (3) 사람 맵(hand_contact.npz)이 있으면 손끝 링크 IoU 와 전이 횟수 비교.

env_isaaclab 파이썬으로 실행 (trimesh, scipy; GPU 불필요, 500 프레임 ≈ 1~2 분):
  python scripts/process_dataset/dataset/stage1_hand_contact.py --hand_traj <clip>/0/hand_traj_best.npz
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parahome_hand_contact import (  # noqa: E402  같은 코어·같은 상수
    _OBJ_ANGVEL_TH, _OBJ_LINVEL_TH, _PROC, _RAW_SCAN, _quat2R, frame_contacts,
)

_ROOT = Path(__file__).resolve().parents[3]
_URDF = _ROOT / "source" / "robotis_sh5" / "data" / "robots" / "G1" / "urdf_pyroki" / "g1_shadow_nomimic.urdf"


def _rpy2R(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return Rz @ Ry @ Rx


def load_link_meshes(urdf_path: str, link_names: list[str]) -> dict[str, np.ndarray]:
    """URDF <visual> 메시 정점을 링크 로컬 좌표로 (scale 과 visual origin 적용). {link: (H,3) float64}"""
    root = ET.parse(urdf_path).getroot()
    links = {l.get("name"): l for l in root.iter("link")}
    out = {}
    for n in link_names:
        if n not in links:
            raise KeyError(f"URDF 에 링크 {n} 이 없습니다: {urdf_path}")
        pts = []
        for v in links[n].findall("visual"):
            m = v.find("geometry/mesh")
            if m is None:
                continue
            fn = m.get("filename")
            fn = fn[len("file://"):] if fn.startswith("file://") else fn
            if not os.path.isabs(fn):
                fn = str(Path(urdf_path).parent / fn)
            mesh = trimesh.load(fn, process=False, force="mesh")
            mesh.merge_vertices()                      # OBJ 의 면별 중복 정점 병합 (표면 동일, 점 수 ~6배 감소 → 속도)
            V = np.asarray(mesh.vertices, np.float64)
            if m.get("scale"):
                V = V * np.array([float(x) for x in m.get("scale").split()])
            o = v.find("origin")
            if o is not None:
                xyz = np.array([float(x) for x in (o.get("xyz") or "0 0 0").split()])
                rpy = [float(x) for x in (o.get("rpy") or "0 0 0").split()]
                V = V @ _rpy2R(*rpy).T + xyz
            pts.append(V)
        if not pts:
            raise ValueError(f"링크 {n} 에 시각 메시가 없습니다")
        out[n] = np.concatenate(pts, 0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hand_traj", required=True, help="1단계 롤아웃 hand_traj_best.npz (link_pos/link_quat 포함)")
    ap.add_argument("--out", default=None, help="출력 경로 (기본: 입력 폴더/hand_contact_stage1.npz)")
    ap.add_argument("--urdf", default=str(_URDF))
    ap.add_argument("--gamma", type=float, default=0.015, help="object-vertex→nearest-hand-vertex contact dist (m) — 사람 맵과 동일")
    ap.add_argument("--num-contacts", type=int, default=50, help="farthest-point-subsample cap per frame — 사람 맵과 동일")
    ap.add_argument("--normal-source", choices=("surface", "to-hand"), default="surface")
    ap.add_argument("--use-velocity-gate", action="store_true", help="사람 스크립트와 같은 물체 속도 게이트 (기본 끔; env 가 런타임에 건다)")
    ap.add_argument("--pad-check-max-mm", type=float, default=10.0, help="pad↔distal 메시 표면 거리 중앙값 상한 (초과 시 exit 3)")
    ap.add_argument("--skip-pad-check", action="store_true")
    ap.add_argument("--force-thresh", type=float, default=1.0, help="검증용: 1단계 접촉력 임계 (N)")
    args = ap.parse_args()

    hd = np.load(args.hand_traj, allow_pickle=True)
    need = ("link_pos", "link_quat", "obj_pos", "obj_quat", "frame", "link_contact_names", "object_name", "control_fps")
    missing = [k for k in need if k not in hd.files]
    if missing:
        print(f"[stage1-contact] {args.hand_traj} 에 {missing} 가 없습니다. link_pos/link_quat 는 갱신된 rollout.py 의 "
              f"--dump_hand_traj 가 기록합니다 → evaluate_sequences_hand_pretrain.sh 를 FORCE=1 로 다시 돌려 덤프하세요.")
        sys.exit(2)
    link_names = [str(n) for n in hd["link_contact_names"]]
    L = len(link_names)
    obj = str(hd["object_name"])
    fps = float(hd["control_fps"])
    fr = np.asarray(hd["frame"])
    T = len(fr)
    lp = np.asarray(hd["link_pos"], np.float64)      # (T,L,3) env-local
    lq = np.asarray(hd["link_quat"], np.float64)     # (T,L,4) wxyz
    op = np.asarray(hd["obj_pos"], np.float64)       # (T,3)   env-local (같은 프레임)
    oq = np.asarray(hd["obj_quat"], np.float64)      # (T,4)   wxyz
    assert lp.shape == (T, L, 3) and lq.shape == (T, L, 4) and op.shape == (T, 3), (lp.shape, lq.shape, op.shape)

    meshes = load_link_meshes(args.urdf, link_names)
    hand_local = [meshes[n] for n in link_names]
    hand_link = np.concatenate([np.full(len(v), i, np.int64) for i, v in enumerate(hand_local)])
    H = int(hand_link.shape[0])
    obj_mesh = trimesh.load(str(_RAW_SCAN / obj / "simplified" / "base.obj"), process=False, force="mesh")
    V = np.asarray(obj_mesh.vertices, np.float64)
    VN = np.asarray(obj_mesh.vertex_normals, np.float64)           # object-LOCAL outward normals (사람 맵과 동일)
    print(f"[stage1-contact] {args.hand_traj}\n    롤아웃 {T} 행 @ {fps:.0f} Hz, 물체 {obj} ({len(V)} 정점), "
          f"손 표면 점 {H} (링크 {L}, URDF {os.path.basename(args.urdf)}), gamma {args.gamma*100:.1f} cm, FPS 상한 {args.num_contacts}")

    def hand_world(t: int) -> np.ndarray:
        return np.concatenate([v @ _quat2R(lq[t, i]).T + lp[t, i] for i, v in enumerate(hand_local)], 0)

    # (1) pad 검사: URDF 링크 프레임이 시뮬레이션 바디 프레임과 같은지. 시뮬레이션이 기록한 pad 점은 distal 표면 위에 있어야 한다.
    if not args.skip_pad_check and "ft_pos" in hd.files and "fingertip_body_names" in hd.files:
        ft = np.asarray(hd["ft_pos"], np.float64)
        ftn = [str(n) for n in hd["fingertip_body_names"]]
        rows = []
        ts = list(range(0, T, max(1, T // 40)))
        for i, n in enumerate(ftn):
            if n not in link_names:
                rows.append((n, float("nan")))
                continue
            li = link_names.index(n)
            ds = [float(np.min(np.linalg.norm(hand_local[li] @ _quat2R(lq[t, li]).T + lp[t, li] - ft[t, i], axis=1))) for t in ts]
            rows.append((n, float(np.median(ds)) * 1000.0))
        vals = np.array([r[1] for r in rows])
        print("    pad 검사 (pad↔distal 메시 최근접 정점 거리 중앙값, mm): "
              + ", ".join(f"{n.replace('robot0_', '')} {d:.1f}" for n, d in rows))
        if np.nanmax(vals) > args.pad_check_max_mm:
            print(f"[stage1-contact] 실패: pad 거리 중앙값 최대 {np.nanmax(vals):.1f} mm > {args.pad_check_max_mm} mm. "
                  f"URDF 링크 프레임과 시뮬레이션 바디 프레임이 어긋난 것으로 보입니다 (--skip-pad-check 로 강행 가능).")
            sys.exit(3)

    # 속도 게이트 (사람 스크립트와 같은 임계, 롤아웃 fps). 기본은 끔 — env 가 런타임에 자체 게이트를 곱한다.
    vel = np.ones(T, bool)
    if args.use_velocity_gate:
        lv = np.zeros_like(op); lv[:-1] = (op[1:] - op[:-1]) * fps
        dotq = np.abs((oq[:-1] * oq[1:]).sum(-1)).clip(0, 1)
        ang = np.zeros(T); ang[:-1] = 2 * np.arccos(dotq) * fps
        vel = (np.linalg.norm(lv, axis=-1) > _OBJ_LINVEL_TH) | (ang > _OBJ_ANGVEL_TH)

    mask = np.zeros((T, L), np.float32)
    tgt_w = np.zeros((T, L, 3), np.float32)
    nrm = np.zeros((T, L, 3), np.float32)
    ncon = np.zeros(T, np.int32)
    t0 = time.time()
    for t in range(T):
        if not vel[t]:
            continue
        hw = hand_world(t)
        R = _quat2R(oq[t])
        # 손 주변 상자(gamma 여유) 밖의 물체 정점은 어떤 손 점과도 gamma 안에 있을 수 없다 → 미리 걸러 속도만 높인다 (결과 동일).
        Vw_all = V @ R.T + op[t]
        lo, hi = hw.min(0) - args.gamma, hw.max(0) + args.gamma
        sub = np.where(np.all((Vw_all >= lo) & (Vw_all <= hi), axis=1))[0]
        if sub.size == 0:
            continue
        fc = frame_contacts(hw, hand_link, V[sub], VN[sub], R, op[t], L, args.gamma, args.num_contacts, args.normal_source)
        if fc is None:
            continue
        mask[t], tgt_w[t], nrm[t], ncon[t] = fc
    # 월드 → 1단계 물체 로컬 (env 로더 규약: t_local = (t_world - obj_pos) @ R_obj). 비접촉 행은 0 유지.
    tgt_l = np.zeros_like(tgt_w)
    for t in range(T):
        sel = mask[t] > 0.5
        if sel.any():
            tgt_l[t, sel] = (tgt_w[t, sel].astype(np.float64) - op[t]) @ _quat2R(oq[t])
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.hand_traj)), "hand_contact_stage1.npz")
    np.savez(out, link_names=np.array(link_names), mask=mask, normal=nrm.astype(np.float32),
             target=tgt_l.astype(np.float32), target_world=tgt_w.astype(np.float32), frame=fr,
             coord=np.array("object"), normal_source=np.array(args.normal_source), control_fps=np.array(fps),
             obj_pose_source=np.array("stage1"), gamma=np.array(args.gamma), num_contacts=np.array(args.num_contacts),
             velocity_gate=np.array(bool(args.use_velocity_gate)),
             source_hand_traj=np.array(os.path.abspath(args.hand_traj)), urdf=np.array(os.path.abspath(args.urdf)))
    m = mask > 0.5
    print(f"    → {out}\n    {time.time() - t0:.0f} s; 접촉 프레임 {int(m.any(1).sum())}/{T}, 프레임당 접촉 링크 {m.sum(1).mean():.1f}, "
          f"프레임당 접촉 정점(FPS 후) 평균 {ncon[ncon > 0].mean() if (ncon > 0).any() else 0:.0f}")

    # (2) 1단계 접촉력 대비
    if "contact_force_w" in hd.files:
        cf = np.linalg.norm(np.asarray(hd["contact_force_w"], np.float64), axis=-1) > args.force_thresh   # (T,L)
        rows = []
        for li, n in enumerate(link_names):
            a, b = m[:, li], cf[:, li]
            u = int((a | b).sum())
            if int(a.sum()) + int(b.sum()) >= 5:
                rows.append((n.replace("robot0_", ""), int(a.sum()), int(b.sum()), (int((a & b).sum()) / u) if u else float("nan")))
        print(f"    1단계 접촉력(>{args.force_thresh:.0f} N) 대비: 접촉 프레임 맵 {int(m.any(1).sum())} / 힘 {int(cf.any(1).sum())}; "
              f"링크별 (맵 프레임 / 힘 프레임 / IoU):")
        for i in range(0, len(rows), 4):
            print("      " + " | ".join(f"{n} {a}/{b}/{iou:.2f}" for n, a, b, iou in rows[i:i + 4]))
    # (3) 사람 맵 대비 (손끝 링크)
    clip = str(hd["clip"]) if "clip" in hd.files else ""
    cls = str(hd["clip_class"]) if "clip_class" in hd.files else ""
    hp = _PROC / "smplx" / cls / clip / "0" / "hand_contact.npz"
    if clip and hp.exists():
        hc = np.load(hp, allow_pickle=True)
        hn = [str(n) for n in hc["link_names"]]
        hm = np.asarray(hc["mask"]) > 0.5
        t30 = np.clip(np.rint(fr.astype(np.float64) * 30.0 / fps).astype(int), 0, hm.shape[0] - 1)
        hm50 = hm[t30]
        rows = []
        for li, n in enumerate(link_names):
            if not n.endswith("distal") or n not in hn:
                continue
            a, b = m[:, li], hm50[:, hn.index(n)]
            u = int((a | b).sum())
            if u:
                rows.append((n.replace("robot0_", ""), int(a.sum()), int(b.sum()), int((a & b).sum()) / u))
        tr_new = float(np.abs(np.diff(m.astype(int), axis=0)).sum(0).mean())
        tr_hum = float(np.abs(np.diff(hm50.astype(int), axis=0)).sum(0).mean())
        print(f"    사람 맵 대비 손끝 (맵 프레임 / 사람 프레임 / IoU): "
              + " | ".join(f"{n} {a}/{b}/{iou:.2f}" for n, a, b, iou in rows))
        print(f"    링크당 평균 전이 횟수: 1단계 맵 {tr_new:.1f}, 사람 맵(50 Hz 정합) {tr_hum:.1f}")


if __name__ == "__main__":
    main()
