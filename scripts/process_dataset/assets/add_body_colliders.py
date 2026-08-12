#!/usr/bin/env python3
"""G1_shadow.usd 의 충돌체 없는 강체에 볼록 껍질(convex hull) 충돌체를 붙입니다.

    /home/peunsu/anaconda3/envs/env_isaaclab/bin/python \
        scripts/process_dataset/assets/add_body_colliders.py [--groups arm waist leg] [--dry-run]

왜 필요한가. 이 자산은 강체 78개 중 48개에만 충돌체가 있습니다. 팔은 14개 중 2개(손목 yaw)뿐이라
어깨·팔꿈치·손목이 조리대와 싱크대를 그냥 통과합니다. 정책이 그 통로를 학습하면 실제 로봇으로
옮길 수 없는 동작이 됩니다.

어떻게. 팔 링크의 기하는 `visuals` 하위에 INSTANCEABLE 로 들어 있어서 그 안의 메시에 직접
스키마를 붙일 수 없습니다(인스턴스는 읽기 전용이고, 프로토타입에 붙이면 그 프로토타입을 쓰는
모든 링크가 함께 바뀝니다). 그래서 시각 메시의 점을 링크 좌표계로 모아 볼록 껍질을 미리 계산하고,
그 결과를 `<링크>/collisions/hull` 이라는 새 메시로 authoring 합니다. 껍질은 정점 수십 개라
가볍고, 무엇이 충돌체인지 USD 만 열어봐도 보입니다.

왜 볼록 껍질인가. 팔 링크는 원래 볼록에 가깝고 접촉 생성이 싸고 안정적입니다. 오목한 형상이
중요한 손과 물체에는 이미 볼록 분해가 적용돼 있습니다.

안전한가. 자기충돌이 꺼져 있으므로(cfg enabled_self_collisions=False, USD 도 False) 팔 충돌체는
외부 물체와의 접촉에만 관여합니다. 레퍼런스 자세에서 팔이 컨텍스트를 뚫는 정도도 미리 쟀습니다 —
칼 클립에서 싱크대만 35% 프레임, 평균 2.9 mm(왼쪽 팔꿈치)로 손(6.4 mm)보다 작습니다.

멱등입니다 — 이미 충돌체가 있는 링크는 건너뜁니다. 원본은 처음 실행할 때 .bak 으로 백업합니다.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import trimesh
from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt

_USD = Path("/home/peunsu/workspace/robotis_sh5/source/robotis_sh5/data/robots/G1/G1_shadow.usd")
_GROUPS = {
    "arm": ("shoulder", "elbow", "wrist"),
    "waist": ("waist", "torso"),
    "leg": ("hip", "knee", "ankle"),
}


def _group_of(name: str) -> str | None:
    if name.startswith("robot0_"):
        return "hand"
    for g, keys in _GROUPS.items():
        if any(k in name for k in keys):
            return g
    return None


def _hull_in_link(stage, link, xc):
    """링크 하위 시각 메시들을 링크 좌표계로 모아 볼록 껍질을 계산합니다. (정점, 면) 또는 None."""
    inv = xc.GetLocalToWorldTransform(link).GetInverse()
    pts = []
    for q in Usd.PrimRange(link, Usd.TraverseInstanceProxies()):
        if not q.IsA(UsdGeom.Mesh):
            continue
        p = UsdGeom.Mesh(q).GetPointsAttr().Get()
        if not p:
            continue
        M = xc.GetLocalToWorldTransform(q) * inv          # 메시 -> 링크
        arr = np.asarray([[v[0], v[1], v[2]] for v in p], np.float64)
        R = np.asarray([[M[i][j] for j in range(3)] for i in range(3)], np.float64)
        t = np.asarray([M[3][0], M[3][1], M[3][2]], np.float64)
        pts.append(arr @ R + t)                            # USD 는 행벡터 규약
    if not pts:
        return None
    hull = trimesh.Trimesh(np.concatenate(pts), process=False).convex_hull
    return np.asarray(hull.vertices, np.float64), np.asarray(hull.faces, np.int32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", default=["arm"], choices=sorted(_GROUPS))
    ap.add_argument("--usd", default=str(_USD))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    stage = Usd.Stage.Open(args.usd)
    xc = UsdGeom.XformCache()
    todo = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI) or _group_of(prim.GetName()) not in args.groups:
            continue
        if any(q.HasAPI(UsdPhysics.CollisionAPI)
               for q in Usd.PrimRange(prim, Usd.TraverseInstanceProxies())):
            continue                                       # 멱등: 이미 충돌체가 있음
        h = _hull_in_link(stage, prim, xc)
        if h is None:
            print(f"  [건너뜀] {prim.GetName()}: 시각 메시가 없습니다")
            continue
        todo.append((prim, h))

    print(f"충돌체를 붙일 링크 {len(todo)}개 (그룹: {', '.join(args.groups)})")
    for prim, (v, f) in todo:
        ext = v.max(0) - v.min(0)
        print(f"  {prim.GetName():<28} 껍질 정점 {len(v):>4}  크기 {np.round(ext * 1000, 0).astype(int).tolist()} mm")
    if args.dry_run or not todo:
        print("(dry-run — 아무것도 쓰지 않았습니다)" if args.dry_run else "할 일 없음")
        return

    bak = Path(args.usd).with_suffix(".usd.bak")
    if not bak.exists():
        shutil.copy(args.usd, bak)
        print(f"원본 백업 -> {bak.name}")

    for prim, (v, f) in todo:
        scope = UsdGeom.Xform.Define(stage, prim.GetPath().AppendChild("collisions"))
        mesh = UsdGeom.Mesh.Define(stage, scope.GetPath().AppendChild("hull"))
        mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in v]))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([int(i) for i in f.reshape(-1)]))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(f)))
        mesh.CreatePurposeAttr(UsdGeom.Tokens.guide)       # 렌더에는 안 보이게
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr().Set(
            UsdPhysics.Tokens.convexHull)
    stage.GetRootLayer().Save()
    print(f"볼록 껍질 충돌체 추가 완료: 링크 {len(todo)}개")


if __name__ == "__main__":
    main()
