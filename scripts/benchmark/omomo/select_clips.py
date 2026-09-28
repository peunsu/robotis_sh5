"""OMOMO 벤치마크 클립 후보를 측정으로 고른다 (train/evaluate_sequences_*.sh 의 CLIPS 목록 근거).

GRAB select_clips.py 와 같은 틀에, OMOMO(큰 물체를 들고 걷기)와 G1(키 ~1.3 m)에 맞춘 기준:
  rigid only        - single_rigid 만 (mop/vacuum 은 두 부품이라 single_articulated, env 미지원)
  standing          - 골반이 낮은 쪽 발(SMPL-X 10/11)보다 계속 0.75 m 이상 위 (GRAB 0.85: 바닥에서 들어 올리느라
                      허리를 굽혀 OMOMO 중앙값이 0.825 — 깊게 쪼그려 앉는 클립만 뺀다)
  actually grasped  - 어떤 손끝 pad 가 물체 표면 2 cm 안 (물체 좌표계 KD-트리)
  carried           - 물체의 가장 낮은 점이 0프레임보다 5 cm 넘게 올라감 (밀기·끌기·차기 제외)
  walking range     - 골반 수평 이동 <= 2.0 m (GRAB 은 제자리 < 0.5 m. OMOMO 는 중앙값 1.2 m 를 걷는다)
  trainable length  - 30 fps 로 120-320 프레임
  G1 reach          - 물체에 닿은 pad 의 최고 높이 <= 1.35 m
  rests on floor    - 0프레임 물체 최저점이 바닥 -3.5 ~ +2 cm. OMOMO 자세의 바닥 관통·뜸은 물체마다 거의 일정하다
                      (largebox 는 377 클립 모두 -2.6~-3.0 cm, suitcase 는 90% 가 +3.7~+4.4 cm). 관통은 env 의
                      spawn-declear 가 스폰 때 들어 올리지만(상한 5 cm) 바닥은 내릴 수 없어 물체가 놓인 프레임에서
                      그만큼 물체 위치 보상이 깎인다. 뜬 물체는 스폰 때 떨어진다
  steady scale      - 프레임별 스케일이 omomo.py 의 물체 스케일과 2% 안 (메시 크기 오차)
  retarget          - RETARGET_REJECTED 가 아님: 뽑힌 클립을 리타게팅(retarget_g1_pyroki → export_wrist_ref →
                      export_wrist_dof6)해서, 손바닥이 한 프레임에 45° 넘게 돌거나(뒤집힘) 한 손의 det(J) < 0.2
                      프레임이 15% 를 넘으면(YZX 특이점) 넣고 다시 고른다 (2026-09-28 측정)
고르기: 기준을 모두 통과한 클립 중 물체마다 하나, 아직 안 쓴 피험자를 먼저, 잡은 프레임이 가장 긴 것.
기본은 rigid 물체 13개 전부. PREFERRED 는 고르는 순서(= 어느 물체가 안 쓴 피험자를 먼저 가져가는지)이고,
--n 을 줄이면 앞에서부터 그만큼만 고른다.

    python scripts/benchmark/omomo/select_clips.py [--n 13]
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

PREFERRED = ["smallbox", "largebox", "plasticbox", "trashcan", "suitcase", "monitor", "woodchair", "smalltable",
             "whitechair", "largetable", "floorlamp", "tripod", "clothesstand"]
TH = dict(stand=0.75, grasp=0.02, lift=0.05, walk=2.0, fmin=120, fmax=320, reach=1.35, floor_lo=-0.035,
          floor_hi=0.02, scale=0.02)
# 리타게팅 결과로 뺀 클립. 뺀 뒤 다시 돌린 선택이 벤치마크 스크립트의 8개다.
RETARGET_REJECTED = {
    "sub6_trashcan_012": "palm 91 deg in one frame at frames 135/162/167, in contact; left det(J) < 0.2 on 16% of frames",
    "sub6_trashcan_017": "palm 145 deg in one frame; arm joints jump 117 deg",
    "sub11_trashcan_014": "right det(J) < 0.2 on 16% of frames (wrist at the YZX singularity)",
    "sub16_whitechair_019": "palm 131 deg in one frame; arm joints jump 70 deg",
    "sub1_largetable_016": "left det(J) < 0.2 on 19% of frames",
}
N_LIFT_VERTS = 2000   # 물체 최저점 계산에 쓰는 정점 수 (무작위, 고정 시드)


def measure(clip_dir: Path, meshes: dict) -> dict:
    ti = json.load(open(clip_dir / "task_info.json"))
    t = np.load(clip_dir / "0" / "trajectory.npz")
    obj = ti["manip_objects"][0]["name"]
    j = t["smplx_joints"]
    ob = t[f"obj__{obj}__base"]
    F = len(j)
    R = Rotation.from_quat(ob[:, [4, 5, 6, 3]]).as_matrix()
    # 손끝 pad 를 물체 좌표계로: p_obj = R^T (p - t)
    pads = t["fingertip_pad_pos"]                                                  # (F,10,3)
    local = np.einsum("fji,fkj->fki", R, pads - ob[:, None, :3])
    grasped = meshes[obj][1].query(local.reshape(-1, 3))[0].reshape(F, 10) < TH["grasp"]
    low = (np.einsum("fij,nj->fni", R, meshes[obj][2])[..., 2] + ob[:, None, 2]).min(1)   # (F,) 물체 최저점
    return dict(clip=clip_dir.name, obj=obj, subject=ti["subject"], split=ti["split"], frames=F,
                stand=float((j[:, 0, 2] - j[:, [10, 11], 2].min(1)).min()),
                walk=float(np.linalg.norm(j[:, 0, :2] - j[0, 0, :2], axis=1).max()),
                lift=float((low - low[0]).max()),
                reach=float(pads[..., 2][grasped].max()) if grasped.any() else 0.0,
                floor0=float(ti["object_min_z_frame0"]),
                scale_dev=float(max(ti["object_scale_max_rel_dev"].values())),
                grasp_frames=int(grasped.any(1).sum()), text=ti["action_text"])


def passes(m: dict) -> list[str]:
    fail = []
    if m["stand"] < TH["stand"]: fail.append("stand")
    if m["grasp_frames"] == 0: fail.append("grasp")
    if m["lift"] <= TH["lift"]: fail.append("lift")
    if m["walk"] > TH["walk"]: fail.append("walk")
    if not TH["fmin"] <= m["frames"] <= TH["fmax"]: fail.append("length")
    if m["reach"] > TH["reach"]: fail.append("reach")
    if not TH["floor_lo"] <= m["floor0"] <= TH["floor_hi"]: fail.append("floor")
    if m["scale_dev"] > TH["scale"]: fail.append("scale")
    if m["clip"] in RETARGET_REJECTED: fail.append("retarget")
    return fail


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=len(PREFERRED))
    args = ap.parse_args()
    root = dataset_paths.processed_root("omomo") / "smplx" / "single_rigid"
    rng = np.random.default_rng(0)
    meshes = {}
    for d in sorted((dataset_paths.processed_root("omomo") / "assets" / "objects").iterdir()):
        if d.name not in PREFERRED:
            continue
        V = np.asarray(trimesh.load(str(d / "mesh" / f"{d.name}.obj"), process=False, force="mesh").vertices)
        meshes[d.name] = (V, cKDTree(V), V[rng.choice(len(V), min(N_LIFT_VERTS, len(V)), replace=False)])
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
        print(f"  {m['clip']:24s} {m['split']:5s} {m['frames']:3d}f  골반 이동 {m['walk']:.2f} m  "
              f"들어 올림 {m['lift']:.2f} m  잡은 프레임 {m['grasp_frames']:3d}  닿은 최고 {m['reach']:.2f} m  "
              f"바닥 {m['floor0'] * 100:+.1f} cm  | {m['text']}")


if __name__ == "__main__":
    main()
