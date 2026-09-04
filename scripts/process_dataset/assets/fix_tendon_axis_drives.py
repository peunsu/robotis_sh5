"""텐던 축 관절(J0)의 USD 조인트 드라이브를 제거해 고정 텐던이 실제로 작동하게 합니다.

    /home/peunsu/anaconda3/envs/env_isaaclab/bin/python \
        scripts/process_dataset/assets/fix_tendon_axis_drives.py [--dry-run] [--usd <path>]

증상. Shadow 손의 말단 마디(J0)가 중간 마디(J1)를 따라 감기지 않습니다. 시뮬레이터 실측:
J1 을 0.99 rad 굽혀도 J0 8개의 가동 범위가 0.0000~0.0004 rad. 감싸 쥐기가 안 되고 밀어붙이기만
가능합니다. cfg 에 fixed_tendons_props 를 넣으면 결합이 서기는커녕 손가락 J1 이 관절 한계를 넘어
10 rad 로 발산합니다(limit_stiffness 0.05~100, spring stiffness 2e3~1e5 전 구간).

원인. 우리 자산의 J0 에는 PhysicsDriveAPI(stiffness=1.0, damping=0.1, target=0, maxForce=inf)가
붙어 있습니다. 액추에이터 정규식(J[1-3])이 J0 를 잡지 않으므로 이 USD 드라이브가 그대로 살아서
J0 를 0 에 붙들고, 텐던이 그것과 싸우다 터집니다. 강성 크기와 무관하게 발산한 이유가 이것입니다 —
힘의 크기가 아니라 두 제약의 충돌이기 때문입니다.

NVIDIA 원본(Isaac 5.1 shadow_hand_instanceable.usd)과 TJ 자산(wrist_free_hand_mount.usd)의 J0 에는
드라이브가 아예 없습니다. 텐던만 작용합니다. Isaac Lab 도 같은 전제로 동작합니다 —
sim/schemas/schemas.py:638 의 modify_joint_drive_properties 는

    if prim.HasAPI(PhysxTendonAxisAPI) and not prim.HasAPI(PhysxTendonAxisRootAPI):
        return False        # 텐던 축 관절의 드라이브는 건드리지 않음

로 축 관절을 건너뜁니다. 즉 "축 관절에는 드라이브가 없다"가 전제인데, 우리 자산만 그 전제를
어기고 있었습니다(G1 + Shadow 병합 과정에서 모든 관절에 드라이브가 일괄로 붙은 것으로 보입니다).

검증(사본으로 나란히 띄워 측정, limit_stiffness=30 / damping=0.2):
    원본                     발산 step 67 (robot0_{l,r}_RFJ1 = 10.26 rad)
    J0 드라이브 제거          정상. J1 0.556 -> J0 0.543, 상관 0.909, 기울기 0.913
    J0 드라이브 제거 + 텐던끔  정상. J0 범위 0.000  <- 결합이 텐던에서 온다는 대조군
  기울기 0.913 은 TJ 자산 실측(0.912)과 사실상 동일합니다.

대상 선정은 Isaac Lab 과 같은 기준을 씁니다: PhysxTendonAxisAPI 를 갖고 PhysxTendonAxisRootAPI 는
갖지 않는 관절. 이름 규칙(J0)으로 고르면 텐던과 무관한 robot0_*_WRJ0 까지 잡힙니다.

멱등입니다 — 이미 드라이브가 없는 관절은 건너뜁니다. 원본은 .pre_tendon.bak 로 백업합니다.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from isaaclab.app import AppLauncher

_USD = "/home/peunsu/workspace/robotis_sh5/source/robotis_sh5/data/robots/G1/G1_shadow.usd"

parser = argparse.ArgumentParser()
parser.add_argument("--usd", default=_USD)
parser.add_argument("--dry-run", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(headless=True)
sim_app = app_launcher.app

from pxr import Usd, UsdPhysics  # noqa: E402


def _is_tendon_axis_only(prim) -> bool:
    """Isaac Lab schemas.py:638 과 같은 기준: 축 API 는 있고 루트 API 는 없는 관절."""
    md = prim.GetMetadata("apiSchemas")
    items = list(md.explicitItems) if md else []
    has_axis = any(s.startswith("PhysxTendonAxisAPI") for s in items)
    has_root = any(s.startswith("PhysxTendonAxisRootAPI") for s in items)
    return has_axis and not has_root


def main() -> None:
    stage = Usd.Stage.Open(args.usd)
    todo = []
    for prim in stage.Traverse():
        if not _is_tendon_axis_only(prim):
            continue
        drv = [a.GetName() for a in prim.GetAttributes() if a.GetName().startswith("drive:")]
        md = prim.GetMetadata("apiSchemas")
        items = list(md.explicitItems) if md else []
        has_drive_api = any("DriveAPI" in s for s in items)
        if not drv and not has_drive_api:
            print(f"  [멱등] {prim.GetName()}: 이미 드라이브 없음")
            continue
        todo.append((prim, drv))

    print(f"\n드라이브를 제거할 텐던 축 관절 {len(todo)}개")
    for prim, drv in todo:
        st = prim.GetAttribute("drive:angular:physics:stiffness")
        print(f"  {prim.GetName():<20} 속성 {len(drv)}개  stiffness={st.Get() if st else None}")
    if args.dry_run or not todo:
        print("(dry-run — 아무것도 쓰지 않았습니다)" if args.dry_run else "할 일 없음")
        sim_app.close()
        return

    bak = Path(args.usd).with_suffix(".usd.pre_tendon.bak")
    if not bak.exists():
        shutil.copy(args.usd, bak)
        print(f"원본 백업 -> {bak.name}")

    for prim, drv in todo:
        for name in drv:
            prim.RemoveProperty(name)
        prim.RemoveAPI(UsdPhysics.DriveAPI, "angular")
    stage.GetRootLayer().Save()
    print(f"\n텐던 축 관절 드라이브 제거 완료: {len(todo)}개")
    sim_app.close()


main()
