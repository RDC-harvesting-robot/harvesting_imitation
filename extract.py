#!/usr/bin/env python3
import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import sqlite3
import os
import pandas as pd
import sys
from scipy.spatial.transform import Rotation
from cv_bridge import CvBridge
import cv2


def extract_topic_data(bag_path, topic_name, image_save_dir=None):
    """
    ROS2バッグから指定されたトピックのデータを抽出し、DataFrameとして返す。
    Imageトピックの場合は、画像を保存しファイルパスをDataFrameに記録する。
    """
    db_files = [f for f in os.listdir(bag_path) if f.endswith('.db3')]
    if not db_files:
        raise FileNotFoundError(f"No .db3 files found in the specified bag path: {bag_path}")

    db_file_path = os.path.join(bag_path, db_files[0])
    print(f"Using DB file: {db_file_path}")

    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()

    cursor.execute("SELECT id, name, type FROM topics")
    topics = {name: (id, type_) for id, name, type_ in cursor.fetchall()}

    if topic_name not in topics:
        print(f"⚠️ Topic '{topic_name}' not found in bag.")
        print("Available topics:")
        for t in topics.keys():
            print(f"  - {t}")
        return None

    topic_id, type_name = topics[topic_name]
    msg_type = get_message(type_name)

    cursor.execute(f"SELECT timestamp, data FROM messages WHERE topic_id = {topic_id}")
    rows = cursor.fetchall()
    conn.close() # データベース接続を閉じる

    if not rows:
        print(f"⚠️ No messages found for topic '{topic_name}'.")
        return None

    rclpy.init(args=None)
    data_list = []
    bridge = CvBridge()
    os.makedirs(image_save_dir, exist_ok=True) if image_save_dir else None

    print(f"Extracting {len(rows)} messages from '{topic_name}'...")

    for i, (timestamp, data) in enumerate(rows):
        msg = deserialize_message(data, msg_type)
        time_sec = timestamp * 1e-9  # UNIX秒

        if type_name == 'sensor_msgs/msg/JointState':
            joint_data = {'time': time_sec}
            for name, position in zip(msg.name, msg.position):
                joint_data[name] = position
            # (オプション) 速度やトルクも必要ならここに追加
            # for name, velocity in zip(msg.name, msg.velocity):
            #     joint_data[f"{name}_velocity"] = velocity
            data_list.append(joint_data)

        elif type_name == 'geometry_msgs/msg/Pose':
            pose = msg
            data_list.append({
                'time': time_sec,
                'position_x': pose.position.x,
                'position_y': pose.position.y,
                'position_z': pose.position.z,
                'orientation_x': pose.orientation.x,
                'orientation_y': pose.orientation.y,
                'orientation_z': pose.orientation.z,
                'orientation_w': pose.orientation.w,
            })

        elif type_name == 'sensor_msgs/msg/Image':
            if not image_save_dir:
                print("⚠️ 'image_save_dir' must be provided for Image topics.")
                continue
                
            # ★変更点: "bgr8" から "passthrough" に変更
            # これにより、depth (16UC1など) も color (bgr8など) も
            # 元のビット深度のまま正しくPNGとして保存される
            try:
                cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            except Exception as e:
                print(f"Error converting image: {e}")
                continue
                
            filename = f"image_{i:06d}.png"
            filepath = os.path.join(image_save_dir, filename)
            cv2.imwrite(filepath, cv_img)

            data_list.append({
                'time': time_sec,
                'width': msg.width,
                'height': msg.height,
                'encoding': msg.encoding,
                'image_file': filepath
            })

    rclpy.shutdown()
    
    if not data_list:
        print(f"⚠️ No data could be extracted for '{topic_name}'.")
        return None

    df = pd.DataFrame(data_list)

    # Poseならオイラー角を追加
    if 'orientation_x' in df.columns:
        def quat_to_euler(row):
            q = [row['orientation_x'], row['orientation_y'], row['orientation_z'], row['orientation_w']]
            r = Rotation.from_quat(q)
            roll, pitch, yaw = r.as_euler('xyz', degrees=False)
            return pd.Series({'roll': roll, 'pitch': pitch, 'yaw': yaw})

        euler_angles = df.apply(quat_to_euler, axis=1)
        df = pd.concat([df, euler_angles], axis=1)

    return df


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract_bag_data.py <path_to_rosbag_folder>")
        sys.exit(1)

    bag_path = sys.argv[1] 
    
    # --- 1. Joint States ---
    topic_joints = "/joint_states"
    print(f"Extracting {topic_joints} ...")
    joint_df = extract_topic_data(bag_path, topic_joints)
    if joint_df is None or joint_df.empty:
        print(f"⚠️ Failed to extract data from {topic_joints}. Exiting.")
        sys.exit(1)

    # --- 2. Color Image ---
    topic_color = "/devices/ee_camera/realsense_node/color/image_raw"
    print(f"Extracting {topic_color} ...")
    image_save_dir_color = os.path.join(bag_path, "extracted_color_images")
    color_df = extract_topic_data(bag_path, topic_color, image_save_dir_color)
    if color_df is None or color_df.empty:
        print(f"⚠️ Failed to extract data from {topic_color}. Exiting.")
        sys.exit(1)

    # --- 3. Depth Image ---
    topic_depth = "/devices/ee_camera/realsense_node/depth/image_rect_raw"
    print(f"Extracting {topic_depth} ...")
    image_save_dir_depth = os.path.join(bag_path, "extracted_depth_images")
    depth_df = extract_topic_data(bag_path, topic_depth, image_save_dir_depth)
    if depth_df is None or depth_df.empty:
        print(f"⚠️ Failed to extract data from {topic_depth}. Exiting.")
        sys.exit(1)

    # --- マージ処理 ---
    
    data_frames_list = [joint_df, color_df, depth_df]
    
    # 基準となる最小時間を全DFから計算
    try:
        all_min_time = min(df['time'].min() for df in data_frames_list)
    except ValueError:
        print("⚠️ Could not find minimum time from dataframes. Check if bags are empty.")
        sys.exit(1)

    # 全てのDFで共通の基準時間から relative_time を計算
    for df in data_frames_list:
        df.sort_values("time", inplace=True)
        df['relative_time'] = df['time'] - all_min_time

    # 3つのデータを time でマージ (joint_df をベースにする)
    print("Merging dataframes...")
    
    # 1. joint_df をベースに color_df をマージ
    merged_df = pd.merge_asof(
        joint_df,
        color_df,
        on="time",
        direction="nearest",
        tolerance=0.02,  # 20ms (同期ズレの許容範囲: 必要に応じて調整)
        suffixes=('_joint', '_color') # 衝突する列名 (relative_timeなど)
    )
    
    # 2. 1の結果に depth_df をマージ
    merged_df = pd.merge_asof(
        merged_df,
        depth_df,
        on="time",
        direction="nearest",
        tolerance=0.02,  # 20ms (同期ズレの許容範囲: 必要に応じて調整)
        suffixes=('', '_depth') 
        # (例: color_df由来の 'width' は 'width', depth_df由来の 'width' は 'width_depth' になる)
    )
    
    merged_df.dropna(inplace=True) # マージできなかった行を削除

    if merged_df is not None and not merged_df.empty:
        # 'time' (joint_df の time) を基準に 'relative_time' を再計算・上書き
        merged_df['relative_time'] = merged_df['time'] - all_min_time
        merged_df = merged_df.drop(columns=["time"])

        # マージによって生じた不要な relative_time_* 列を削除
        other_rel_times = [c for c in merged_df.columns if c.startswith('relative_time_')]
        merged_df = merged_df.drop(columns=other_rel_times)

        # relative_time を先頭に
        merged_df = merged_df[['relative_time'] + [c for c in merged_df.columns if c != 'relative_time']]

        output_csv_path = os.path.join(bag_path, "joint_color_depth_merged.csv")
        merged_df.to_csv(output_csv_path, index=False)
        print("---")
        print(f"✅ Merged data saved: {output_csv_path}")
        print(f"📸 Color images saved under: {image_save_dir_color}")
        print(f"📸 Depth images saved under: {image_save_dir_depth}")
    else:
        print("⚠️ Could not merge data. Check tolerance or bag contents.")