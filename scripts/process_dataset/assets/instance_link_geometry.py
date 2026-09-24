"""로봇 USD 의 링크 형상 그룹(visuals/collisions)을 instanceable 로 바꾼 시뮬레이션용 사본을 만든다.

G1 몸 링크는 이미 instanced 인데, 붙인 Shadow 손(visuals 메시 3.9 MB)과 팔 hull 충돌체는 파일에 직접
저작돼 있어 env 마다 따로 합성된다. 각 그룹의 내용을 같은 파일 안 `/Instance_Prototypes/<링크>__<그룹>`
(`over`, 스테이지에 정의되지 않음)으로 옮기고, 그룹 자리에는 그 경로로의 내부 참조 + instanceable 을 둔다.
G1 몸의 `Flattened_Prototype_*` 와 같은 방식이고, prim 경로는 인스턴스 프록시로 그대로 남는다.

원본은 그대로 둔다 — 오프라인 도구(리타깃·URDF 추출·build_shadow_floating_usd 등)가 `Traverse()` 로
손 메시를 읽기 때문이다. 원본을 다시 빌드하면 이 스크립트도 다시 돌려야 한다(env cfg 가 확인한다).

쓰고 나면 원본과 합성 결과를 비교한다: 모든 prim(인스턴스 프록시 포함)의 타입·스키마·속성값·관계·월드
변환이 같아야 하고, 다른 건 그룹 prim 의 instanceable/참조뿐이어야 한다.

    python scripts/process_dataset/assets/instance_link_geometry.py            # 기본 3개 파일
    python scripts/process_dataset/assets/instance_link_geometry.py <src.usd> ...
"""

import argparse
from pathlib import Path

from pxr import Sdf, Usd, UsdGeom, UsdPhysics

_ROBOTS = Path(__file__).resolve().parents[3] / "source" / "robotis_sh5" / "data" / "robots" / "G1"
_DEFAULT_SRCS = [_ROBOTS / "G1_shadow.usd", _ROBOTS / "shadow_float6_l.usd", _ROBOTS / "shadow_float6_r.usd"]
_PROTO_ROOT = Sdf.Path("/Instance_Prototypes")


def instanced_path(src: Path) -> Path:
    return src.with_name(f"{src.stem}_inst{src.suffix}")


def _geometry_groups(stage: Usd.Stage) -> list[Sdf.Path]:
    """인스턴스 밖에 직접 저작된 형상(Gprim)을 담은, 강체 링크 바로 아래 그룹 prim 들."""
    groups: list[Sdf.Path] = []
    for p in Usd.PrimRange(stage.GetDefaultPrim()):
        if not p.IsA(UsdGeom.Gprim):
            continue
        q = p
        while q.GetParent() and not q.GetParent().HasAPI(UsdPhysics.RigidBodyAPI):
            q = q.GetParent()
        if not q.GetParent() or q.GetName() not in ("visuals", "collisions"):
            raise SystemExit(f"{p.GetPath()}: 강체 링크 아래 visuals/collisions 그룹에 있지 않은 형상")
        if q.GetPath() not in groups:
            groups.append(q.GetPath())
    return groups


def make_instanced(src: Path, dst: Path) -> int:
    stage = Usd.Stage.Open(str(src))
    groups = _geometry_groups(stage)
    layer = Sdf.Layer.CreateNew(str(dst))
    layer.TransferContent(stage.GetRootLayer())
    if layer.GetPrimAtPath(_PROTO_ROOT):
        raise SystemExit(f"{src} 에 이미 {_PROTO_ROOT} 가 있다")
    for g in groups:
        spec = layer.GetPrimAtPath(g)
        if spec.properties or spec.referenceList.GetAddedOrExplicitItems() or spec.instanceable:
            raise SystemExit(f"{g}: 그룹 prim 에 속성/참조가 있어 옮기지 않는다")
        proto = _PROTO_ROOT.AppendChild(f"{g.GetParentPath().name}__{g.name}")
        Sdf.CreatePrimInLayer(layer, proto)     # 조상은 over 로 만들어진다 → 스테이지에 정의되지 않음
        if not Sdf.CopySpec(layer, g, layer, proto):
            raise SystemExit(f"{g} → {proto} 복사 실패")
        for child in list(spec.nameChildren):
            del spec.nameChildren[child.name]
        spec.referenceList.Prepend(Sdf.Reference(primPath=proto))
        spec.instanceable = True
    layer.Save()
    return len(groups)


def _prim_signature(p: Usd.Prim, xf: UsdGeom.XformCache) -> tuple:
    attrs = tuple((a.GetName(), repr(a.Get())) for a in p.GetAuthoredAttributes())
    rels = tuple((r.GetName(), tuple(str(t) for t in r.GetTargets())) for r in p.GetAuthoredRelationships())
    mat = tuple(round(v, 12) for row in xf.GetLocalToWorldTransform(p) for v in row) if p.IsA(UsdGeom.Xformable) else ()
    return (p.GetTypeName(), tuple(p.GetAppliedSchemas()), p.IsActive(), attrs, rels, mat)


def check_equivalent(src: Path, dst: Path) -> tuple[int, int]:
    """(합성 prim 수, 인스턴스 수). 다르면 SystemExit."""
    pred = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
    sigs = []
    for path in (src, dst):
        st = Usd.Stage.Open(str(path))
        xf = UsdGeom.XformCache()
        sigs.append({str(p.GetPath()): _prim_signature(p, xf)
                     for p in Usd.PrimRange(st.GetPseudoRoot(), pred)})
        n_inst = sum(1 for p in st.Traverse() if p.IsInstance())
    a, b = sigs
    if a.keys() != b.keys():
        raise SystemExit(f"prim 집합이 다르다: 원본에만 {sorted(a.keys() - b.keys())[:5]}, "
                         f"사본에만 {sorted(b.keys() - a.keys())[:5]}")
    diff = [k for k in a if a[k] != b[k]]
    if diff:
        raise SystemExit(f"{len(diff)} 개 prim 의 합성 결과가 다르다: {diff[:5]}")
    return len(a), n_inst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("srcs", nargs="*", type=Path, default=_DEFAULT_SRCS)
    args = ap.parse_args()
    for src in args.srcs:
        dst = instanced_path(src)
        n = make_instanced(src, dst)
        n_prims, n_inst = check_equivalent(src, dst)
        print(f"{src.name} → {dst.name}: 그룹 {n}개를 instanceable 로, 인스턴스 {n_inst}개, "
              f"합성 prim {n_prims}개 모두 원본과 같음")


if __name__ == "__main__":
    main()
