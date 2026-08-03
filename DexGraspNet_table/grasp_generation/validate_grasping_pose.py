import argparse
import glob
import json
import math
import os
import time
import uuid  # 引入 uuid 防止文件名冲突

import numpy as np
import pybullet as p
import pybullet_data
import transforms3d
import trimesh


def validate_grasp_pose(
        hand_mesh_path,
        obj_mesh_path,
        obj_data_path,
        inflation=0.002,
        debug=False,
        shake_force=5.0  # <--- [修复] 加回参数
):
    """
    验证抓取稳定性。
    """

    # --- 1. 初始化 PyBullet ---
    connection_mode = p.GUI if debug else p.DIRECT

    if p.isConnected():
        p.disconnect()
    p.connect(connection_mode)

    # [修复] 强制重置仿真并关闭文件缓存 (解决 Batch 模式 Scale 错误问题)
    p.resetSimulation()
    p.setPhysicsEngineParameter(enableFileCaching=0)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # [优化] 初始重力设为 0 (为了 Settle 阶段)
    p.setGravity(0, 0, 0)
    p.setTimeStep(1. / 240.)

    # --- 2. 准备数据 ---
    obj_data = json.load(open(obj_data_path, 'r'))
    obj_scale = np.array(obj_data['scale'])
    obj_pose_mat = np.array(obj_data['pose'])

    # --- 3. 加载手 ---
    # [优化] 使用 uuid 生成唯一文件名，彻底避免多进程或系统缓存冲突
    temp_hand_path = f"temp_hand_inflated_{uuid.uuid4().hex}.obj"

    mesh_hand = trimesh.load(hand_mesh_path, force='mesh', process=False)
    mesh_hand.fix_normals()
    mesh_hand.vertices += mesh_hand.vertex_normals * inflation
    mesh_hand.export(temp_hand_path)

    hand_col = p.createCollisionShape(p.GEOM_MESH, fileName=temp_hand_path, flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
    hand_body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=hand_col, basePosition=[0, 0, 0])
    p.changeDynamics(hand_body, -1, lateralFriction=2.0, spinningFriction=0.1)

    # --- 4. 加载物体 ---
    start_pos = obj_pose_mat[:3, 3]
    quat_wxyz = transforms3d.quaternions.mat2quat(obj_pose_mat[:3, :3])
    start_quat = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]

    obj_col = p.createCollisionShape(p.GEOM_MESH, fileName=obj_mesh_path, meshScale=obj_scale)
    obj_body = p.createMultiBody(baseMass=0.1, baseCollisionShapeIndex=obj_col, basePosition=start_pos, baseOrientation=start_quat)

    p.changeDynamics(
        obj_body,
        -1,
        lateralFriction=1.0,
        rollingFriction=0.001,
        spinningFriction=0.001,
        linearDamping=0.05,
    )

    # --- 5. Debug 视图 ---
    if debug:
        p.changeVisualShape(obj_body, -1, rgbaColor=[0.2, 0.8, 0.2, 1])
        p.changeVisualShape(hand_body, -1, rgbaColor=[0.8, 0.2, 0.2, 0.6])
        p.resetDebugVisualizerCamera(cameraDistance=0.3, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=start_pos)
        print(f"Debug Mode: Press 'S' to start.")
        while p.isConnected():
            keys = p.getKeyboardEvents()
            if ord('s') in keys and keys[ord('s')] & p.KEY_WAS_TRIGGERED:
                break
            p.getMouseEvents()

    # ==========================================
    # Phase 1: Settle (无重力微调，解决穿模)
    # ==========================================
    # 先跑 50 步无重力，让膨胀的手把物体轻轻推到表面
    for _ in range(50):
        p.stepSimulation()
        if debug:
            time.sleep(1. / 1000.)

    # ==========================================
    # Phase 2: Test (开启重力 + 扰动)
    # ==========================================
    p.setGravity(0, 0, -9.8)  # [关键] 此时才开启重力

    # 重新获取基准位置 (因为 Settle 阶段物体可能微动了)
    curr_pos_settled, _ = p.getBasePositionAndOrientation(obj_body)
    start_pos_check = np.array(curr_pos_settled)

    sim_steps = 200
    success = True

    debug_text_id = -1
    if debug:
        text_pos = np.array(start_pos) + np.array([0, -0.1, 0.15])
        debug_text_id = p.addUserDebugText("Start", text_pos, [1, 1, 0], textSize=1.5)

    for i in range(sim_steps):
        # [Shake Test]
        # 在第 20 步后开始施加力 (给重力一点时间先作用)
        if i > 50:
            alpha = (i - 50) * 0.1
            # [修复] 使用传入的 shake_force 参数
            force_x = math.sin(alpha) * shake_force
            force_y = math.cos(alpha) * shake_force

            curr_pos, _ = p.getBasePositionAndOrientation(obj_body)
            p.applyExternalForce(obj_body, -1, [force_x, force_y, 0], curr_pos, p.WORLD_FRAME)

            if debug and i % 10 == 0:
                p.addUserDebugLine(curr_pos, [curr_pos[0] + force_x * 0.01, curr_pos[1] + force_y * 0.01, curr_pos[2]], [1, 0, 0], lifeTime=0.1)

        p.stepSimulation()

        curr_pos, _ = p.getBasePositionAndOrientation(obj_body)
        dist = np.linalg.norm(np.array(curr_pos) - start_pos_check)

        if debug:
            p.addUserDebugText(f"Step: {i}\nDist: {dist:.3f}m", text_pos, [1, 1, 0], replaceItemUniqueId=debug_text_id)
            time.sleep(1. / 240.)

        if dist > 0.1:
            if debug:
                print(f"Failed at step {i}: Object fell ({dist:.3f}m).")
            success = False
            break

    # --- Cleanup ---
    if debug and p.isConnected():
        print("\nSimulation finished. Close window.")
        while p.isConnected():
            try:
                p.getMouseEvents()
                time.sleep(0.01)
            except:
                break

    if p.isConnected():
        p.disconnect()

    if os.path.exists(temp_hand_path):
        try:
            os.remove(temp_hand_path)
        except:
            pass

    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hand_path', type=str, default='grasp_data/hand_mesh_00031.obj')
    parser.add_argument('--obj_mesh_path', type=str, default='decomposed.obj')
    parser.add_argument('--obj_data_path', type=str, default='grasp_data/obj_data_00031.json')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--batch', action='store_true')
    # [新增] 命令行控制力度
    parser.add_argument('--shake', type=float, default=5.0, help="Shake force in Newtons")

    args = parser.parse_args()

    if args.batch:
        print(f"--- Running Batch Validation (Shake Force: {args.shake}N) ---")
        hand_files = sorted(glob.glob("grasp_data/hand_mesh_*.obj"))

        success_count = 0
        for i, hand_path in enumerate(hand_files):
            case_id = os.path.basename(hand_path).split('_')[-1].split('.')[0]
            obj_data_path = f"grasp_data/obj_data_{case_id}.json"

            if not os.path.exists(obj_data_path):
                continue

            is_valid = validate_grasp_pose(
                hand_path,
                'decomposed.obj',
                obj_data_path,
                debug=False,
                shake_force=args.shake  # [修复] 传入参数
            )

            if is_valid:
                print(f"[{i+1}/{len(hand_files)}] {case_id}... -> SUCCESS")
                success_count += 1
            # else:
            # print(" -> FAILURE")

        print(f"\nResult: {success_count}/{len(hand_files)} passed.")

    else:
        # Single Run
        validate_grasp_pose(
            args.hand_path,
            args.obj_mesh_path,
            args.obj_data_path,
            debug=args.debug,
            shake_force=args.shake  # [修复] 传入参数
        )
