"""GRAB 벤치마크 클립 후보를 측정으로 고른다 (train/evaluate_sequences_*.sh 의 CLIPS 목록 근거).

ParaHome 스크립트와 같은 기준 + G1(키 ~1.3 m) 도달성 기준:
  single-hand intent   - pass(남에게 건네기)·offhand(다른 손으로 옮기기) 제외
  standing             - 골반이 낮은 쪽 발(SMPL-X 10/11)보다 계속 0.85 m 이상 위
  actually grasped     - 어떤 손끝 pad 가 물체 표면 2 cm 안 (물체 좌표계 KD-트리, 정점 간격 ~mm)
  manipulated          - 물체가 첫 위치에서 0.10 m 넘게 이동
  in place             - 골반 수평 이동 < 0.5 m
  trainable length     - 30 fps 로 120-320 프레임
  G1 reach             - 첫 프레임 테이블 윗면 <= 0.95 m, 물체 중심 최고 높이 <= 1.35 m
고르기: 기준을 모두 통과한 클립 중 물체마다 하나, 아직 안 쓴 피험자를 먼저, 잡은 프레임이 가장 긴 것.
물체는 PREFERRED 순서(일상 물체 우선)로 --n 개.

    python scripts/benchmark/grab/select_clips.py [--n 8]

측정 대상은 지금의 processed/grab 이다. 벤치마크 스크립트의 8개는 grab.py 가 접촉 구간으로 자르기 전에 고른 고정 목록이라
(자른 뒤 154-213 프레임), 지금 다시 돌리면 다른 목록이 나올 수 있다.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.append(str(Path(__file__).resolve().parents[2] / "process_dataset" / "dataset"))
import dataset_paths  # noqa: E402

PREFERRED = ["mug", "cup", "bowl", "hammer", "waterbottle", "banana", "knife", "fryingpan", "teapot",
             "stapler", "flashlight", "apple", "wineglass", "camera", "toothpaste", "scissors"]
EXCLUDED_INTENTS = {"pass", "offhand"}
TH = dict(stand=0.85, grasp=0.02, travel=0.10, in_place=0.5, fmin=120, fmax=320, table=0.95, obj_top=1.35)


def measure(clip_dir: Path, meshes: dict) -> dict:
    ti = json.load(open(clip_dir / "task_info.json"))
    t = np.load(clip_dir / "0" / "trajectory.npz")
    obj = ti["manip_objects"][0]["name"]
    j = t["smplx_joints"]
    ob = t[f"obj__{obj}__base"]
    F = len(j)
    stand = float((j[:, 0, 2] - j[:, [10, 11], 2].min(1)).min())
    travel = float(np.linalg.norm(ob[:, :3] - ob[0, :3], axis=1).max())
    in_place = float(np.linalg.norm(j[:, 0, :2] - j[0, 0, :2], axis=1).max())
    tb = t["ctx__table__base"]
    table_top = float((Rotation.from_quat(tb[0, [4, 5, 6, 3]]).apply(meshes["table"][0]) + tb[0, :3])[:, 2].max())
    # 손끝 pad 를 물체 좌표계로: p_obj = R^T (p - t)
    R = Rotation.from_quat(ob[:, [4, 5, 6, 3]])
    pads = t["fingertip_pad_pos"]                                     # (F,10,3)
    local = np.stack([R[f].inv().apply(pads[f] - ob[f, :3]) for f in range(F)])
    dist = meshes[obj][1].query(local.reshape(-1, 3))[0].reshape(F, 10).min(1)
    return dict(clip=clip_dir.name, obj=obj, subject=ti["subject"], intent=ti["motion_intent"], split=ti["split"],
                frames=F, stand=stand, travel=travel, in_place=in_place, table_top=table_top,
                obj_top=float(ob[:, 2].max()), grasp_frames=int((dist < TH["grasp"]).sum()))


def passes(m: dict) -> list[str]:
    fail = []
    if m["intent"] in EXCLUDED_INTENTS: fail.append("intent")
    if m["stand"] < TH["stand"]: fail.append("stand")
    if m["grasp_frames"] == 0: fail.append("grasp")
    if m["travel"] <= TH["travel"]: fail.append("travel")
    if m["in_place"] >= TH["in_place"]: fail.append("in_place")
    if not TH["fmin"] <= m["frames"] <= TH["fmax"]: fail.append("length")
    if m["table_top"] > TH["table"]: fail.append("table")
    if m["obj_top"] > TH["obj_top"]: fail.append("obj_top")
    return fail


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()
    root = dataset_paths.processed_root("grab") / "smplx" / "single_rigid"
    meshes = {}
    for d in sorted((dataset_paths.processed_root("grab") / "assets" / "objects").iterdir()):
        V = np.asarray(trimesh.load(str(d / "mesh" / f"{d.name}.obj"), process=False, force="mesh").vertices)
        meshes[d.name] = (V, cKDTree(V))
    rows = [measure(c, meshes) for c in sorted(root.iterdir())]
    fails = defaultdict(int)
    ok = []
    for m in rows:
        f = passes(m)
        for k in f: fails[k] += 1
        if not f: ok.append(m)
    print(f"클립 {len(rows)}개 중 모든 기준 통과 {len(ok)}개. 기준별 탈락 (중복 포함): {dict(fails)}")
    by_obj = defaultdict(list)
    for m in ok:
        by_obj[m["obj"]].append(m)
    print("통과 클립 수 (물체별):", {o: len(v) for o, v in sorted(by_obj.items(), key=lambda kv: -len(kv[1]))})
    used, picks = set(), []
    for obj in PREFERRED:
        if obj not in by_obj or len(picks) >= args.n:
            continue
        cand = sorted(by_obj[obj], key=lambda m: (m["subject"] in used, -m["grasp_frames"]))
        picks.append(cand[0]); used.add(cand[0]["subject"])
    print(f"\n선택 {len(picks)}개:")
    for m in picks:
        print(f"  {m['clip']:32s} {m['intent']:8s} {m['split']:5s} {m['frames']:3d}f  이동 {m['travel']:.2f} m  "
              f"잡은 프레임 {m['grasp_frames']:3d}  테이블 {m['table_top']:.2f} m  물체 최고 {m['obj_top']:.2f} m")


if __name__ == "__main__":
    main()
