from scipy.spatial.transform import Rotation as R
import numpy as np

# --- 1. 基本となる「正面」の姿勢 ---
roll_base = -1.5708   # X軸周りの回転
pitch_base = -1.5708  # Y軸周りの回転
yaw_base = 0.0        # Z軸周りの回転

# ZYX (Yaw-Pitch-Roll) 順でクォータニオンを計算
r_base = R.from_euler('zyx', [yaw_base, pitch_base, roll_base], degrees=False)
q_base = r_base.as_quat() # (x, y, z, w)
print(f"Base Quaternion (x,y,z,w): {q_base}")

rpy_check = r_base.as_euler('zyx', degrees=False)
print(f"Base RPY (zyx) check: {rpy_check}")


pitch_additional_deg = -2.5
pitch_additional_rad = np.deg2rad(pitch_additional_deg) # 1度をラジアンに変換

r_additional = R.from_euler('y', [pitch_additional_rad], degrees=False)
q_additional = r_additional.as_quat()
print(f"Additional Rotation Quaternion (y-axis only): {q_additional}")

# --- 3. 2つのクォータニオンを合成し、最終RPYに変換 ---
r_final = r_additional * r_base
q_final = r_final.as_quat()
print(f"Final Quaternion (x,y,z,w): {q_final}")

# 最終RPYをZ-Y-X (Yaw-Pitch-Roll) 順で取得
final_rpy_zyx = r_final.as_euler('zyx', degrees=False)
print(f"Final RPY (yaw, pitch, roll) for URDF: {final_rpy_zyx}")

urdf_rpy_format = [final_rpy_zyx[0][2], final_rpy_zyx[0][1], final_rpy_zyx[0][0]]
print(f"Final RPY for URDF (roll, pitch, yaw): {urdf_rpy_format}")
