# 가장 먼저 SimulationApp을 실행해야 해!
from isaacsim import SimulationApp

# 가볍게 변환만 할 거니까 headless 모드로 설정
simulation_app = SimulationApp({"headless": True})

import asyncio
import omni.kit.asset_converter as converter
from pxr import UsdPhysics, UsdGeom, Usd
import omni.usd
import os

async def convert_obj_to_usd_with_physics(input_obj_path, output_usd_path):
    if not os.path.exists(input_obj_path):
        print(f"에러: 파일이 없어! {input_obj_path}")
        return

    task_manager = converter.get_instance()
    options = converter.AssetConverterContext()
    
    # 변환 실행
    task = task_manager.create_converter_task(
        input_obj_path, output_usd_path, None, options
    )
    success = await task.wait_until_finished()
    
    if success:
        stage = Usd.Stage.Open(output_usd_path)
        root_prim = stage.GetDefaultPrim()
        
        if not root_prim:
            # Default Prim이 설정 안 되어 있다면 첫 번째 자식을 사용
            root_prim = next(stage.Traverse())
            stage.SetDefaultPrim(root_prim)
        
        # Rigid Body 및 Mass 설정 (0.5kg)
        UsdPhysics.RigidBodyAPI.Apply(root_prim)
        mass_api = UsdPhysics.MassAPI.Apply(root_prim)
        mass_api.CreateMassAttr(0.5)

        # Collider 설정
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(prim)
                mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
                mesh_collision_api.CreateApproximationAttr("convexDecomposition")

        stage.GetRootLayer().Save()
        print(f"변환 성공: {output_usd_path}")
    else:
        print("변환 실패")

# 실행부
input_path = "/home/peunsu/workspace/robotis_sh5/retargeting/DexYCB/models/006_mustard_bottle/textured.obj" # 절대경로 추천
output_path = "/home/peunsu/workspace/robotis_sh5/retargeting/DexYCB/models/006_mustard_bottle/textured.usd"

asyncio.ensure_future(convert_obj_to_usd_with_physics(input_path, output_path))

# 작업 완료를 위해 잠시 대기 후 종료
for _ in range(100):
    simulation_app.update()

simulation_app.close()