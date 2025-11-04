#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import numpy as np
import cv2
from cv_bridge import CvBridge

# ROS 2 メッセージ
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray # ★コントローラに合わせて変更

# 同期用
import message_filters

# ==========================================================
# ★★★ 設定 (train_bc.py と全く同じにする) ★★★
# ==========================================================
IMG_SIZE = (224, 224)  # (高さ, 幅)
MODEL_PATH = "best_bc_model.pth" # 学習済みモデルのパス

# ★★★ ロボットの関節可動域 (正規化/逆正規化に必須) ★★★
JOINT_LIMITS = {
    'shoulder_pan_joint': [-3.14, 3.14],
    'shoulder_lift_joint': [-3.14, 3.14],
    'elbow_joint': [-3.14, 3.14],
    'wrist_1_joint': [-3.14, 3.14],
    'wrist_2_joint': [-3.14, 3.14],
    'wrist_3_joint': [-3.14, 3.14]
}

JOINT_NAMES = list(JOINT_LIMITS.keys())
JOINT_DIM = len(JOINT_NAMES)

# --- ROS 2 トピック名 ---
COLOR_IMAGE_TOPIC = "/devices/ee_camera/realsense_node/color/image_raw"
JOINT_STATE_TOPIC = "/joint_states"

# ★★★ 出力先コントローラのトピック ★★★
# (ros2_control の forward_position_controller を想定)
JOINT_COMMAND_TOPIC = "/forward_position_controller/commands"

# ==========================================================
# 2. BCモデル (train_bc.py からコピー)
# ==========================================================
# モデルをロードするために、学習時と全く同じクラス定義が必要
class BCModel(nn.Module):
    def __init__(self, joint_dim=JOINT_DIM, action_dim=JOINT_DIM):
        super().__init__()
        self.joint_dim = joint_dim
        self.action_dim = action_dim
        
        resnet = models.resnet18() # pretrained=FalseでOK (重みはロードするため)
        self.vision_encoder = nn.Sequential(*list(resnet.children())[:-1])
        vision_feature_dim = 512
        
        self.joint_encoder = nn.Sequential(
            nn.Linear(self.joint_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )
        joint_feature_dim = 64
        
        combined_feature_dim = vision_feature_dim + joint_feature_dim
        
        self.head = nn.Sequential(
            nn.Linear(combined_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2), # eval()モードでは自動で無効になる
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim),
            nn.Tanh()
        )

    def forward(self, obs):
        color_obs = obs['color']
        joint_obs = obs['joint']
        
        vision_features = self.vision_encoder(color_obs)
        vision_features = vision_features.view(vision_features.size(0), -1)
        joint_features = self.joint_encoder(joint_obs)
        combined_features = torch.cat([vision_features, joint_features], dim=1)
        predicted_action = self.head(combined_features)
        
        return predicted_action

# ==========================================================
# 3. 推論ノード (InferenceNode) - ★同期しないバージョン★
# ==========================================================

class InferenceNode(Node):
    def __init__(self):
        super().__init__('bc_inference_node')
        self.get_logger().info("BC Inference Node starting...")

        # --- 1. デバイス設定 ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")

        # --- 2. モデルのロード ---
        self.model = self._load_model(MODEL_PATH)
        
        # --- 3. CV Bridge ---
        self.bridge = CvBridge()
        
        # --- 4. 最新データを保持する変数 ---
        self.latest_image_msg = None
        self.latest_joint_msg = None
        
        # --- 5. パブリッシャー (アーム制御用) ---
        self.action_pub = self.create_publisher(
            Float64MultiArray, 
            JOINT_COMMAND_TOPIC, 
            10)

        # --- 6. サブスクライバー (同期なし) ---
        # 2つのコールバックを別々に作成
        
        # 画像が来たら、最新の画像を self.latest_image_msg に保存するだけ
        self.image_sub = self.create_subscription(
            Image, 
            COLOR_IMAGE_TOPIC, 
            self._image_callback, # ★コールバック関数を変更
            10 # QoS
        )
        
        # 関節状態が来たら、推論を実行する
        self.joint_sub = self.create_subscription(
            JointState, 
            JOINT_STATE_TOPIC, 
            self._joint_callback, # ★コールバック関数を変更
            10 # QoS
        )

        self.get_logger().info("Node initialized. Waiting for topics...")

    def _load_model(self, path):
        # ... (この関数は変更なし) ...
        try:
            model = BCModel(joint_dim=JOINT_DIM, action_dim=JOINT_DIM).to(self.device)
            model.load_state_dict(torch.load(path, map_location=self.device))
            model.eval() 
            self.get_logger().info(f"Successfully loaded model from {path}")
            return model
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            rclpy.shutdown()

    def _image_callback(self, msg):
        """画像トピックのコールバック。最新の画像を保存するだけ。"""
        self.latest_image_msg = msg
        # self.get_logger().info("Image received.", throttle_duration_sec=5.0)

    def _joint_callback(self, msg):
        """関節状態のコールバック。ここでAI推論をトリガーする。"""
        # self.get_logger().info("Joint state received.", throttle_duration_sec=5.0)
        
        # 最新の関節状態を保存
        self.latest_joint_msg = msg
        
        # --- まだ両方のデータが揃っていない場合は何もしない ---
        if self.latest_image_msg is None or self.latest_joint_msg is None:
            self.get_logger().warn("Waiting for both topics to arrive...", throttle_duration_sec=5.0)
            return

        # --- 両方揃ったら、推論を実行 ---
        
        # --- 1. 画像の前処理 ---
        obs_color = self._preprocess_image(self.latest_image_msg)
        
        # --- 2. 関節状態の前処理 ---
        obs_joint = self._preprocess_joints(self.latest_joint_msg)
        
        if obs_color is None or obs_joint is None:
            self.get_logger().warn("Skipping frame due to preprocessing error.", throttle_duration_sec=2.0)
            return

        # --- 3. モデル推論 (GPU/CPUで実行) ---
        with torch.no_grad():
            obs_dict = {
                'color': obs_color.unsqueeze(0).to(self.device),
                'joint': obs_joint.unsqueeze(0).to(self.device)
            }
            normalized_action = self.model(obs_dict)
            normalized_action_cpu = normalized_action.squeeze(0).cpu().numpy()

        # --- 4. 後処理 (逆正規化) ---
        denormalized_action = self._denormalize_joints(normalized_action_cpu)

        # --- 5. コマンド送信 ---
        action_msg = Float64MultiArray()
        action_msg.data = denormalized_action.tolist()
        self.action_pub.publish(action_msg)
        
        # self.get_logger().info(f"Published action.", throttle_duration_sec=2.0)

    def _preprocess_image(self, msg):
        # ... (この関数は変更なし) ...
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            img_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (IMG_SIZE[1], IMG_SIZE[0]), interpolation=cv2.INTER_AREA)
            img_normalized = img_resized.astype(np.float32) / 255.0
            img_tensor = torch.tensor(img_normalized).permute(2, 0, 1)
            return img_tensor
        except Exception as e:
            self.get_logger().error(f"Image preprocessing failed: {e}")
            return None

    def _preprocess_joints(self, msg):
        # ... (この関数は変更なし) ...
        try:
            joint_angles = [0.0] * JOINT_DIM
            joint_map = dict(zip(msg.name, msg.position))
            for i, name in enumerate(JOINT_NAMES):
                if name not in joint_map:
                    self.get_logger().warn(f"Joint '{name}' not found in /joint_states message.")
                    return None
                joint_angles[i] = joint_map[name]
            normalized_angles = self._normalize_joints(np.array(joint_angles))
            return torch.tensor(normalized_angles, dtype=torch.float32)
        except Exception as e:
            self.get_logger().error(f"Joint preprocessing failed: {e}")
            return None

    def _normalize_joints(self, angles):
        # ... (この関数は変更なし) ...
        normalized = np.zeros_like(angles)
        for i, angle in enumerate(angles):
            joint_name = JOINT_NAMES[i]
            min_val, max_val = JOINT_LIMITS[joint_name]
            normalized[i] = 2 * ((angle - min_val) / (max_val - min_val)) - 1
        return normalized

    def _denormalize_joints(self, normalized_angles):
        # ... (この関数は変更なし) ...
        denormalized = np.zeros_like(normalized_angles)
        for i, norm_angle in enumerate(normalized_angles):
            joint_name = JOINT_NAMES[i]
            min_val, max_val = JOINT_LIMITS[joint_name]
            denormalized[i] = (norm_angle + 1) / 2 * (max_val - min_val) + min_val
        return denormalized

# ==========================================================
# 5. メイン実行部
# ==========================================================

def main(args=None):
    rclpy.init(args=args)
    
    node = InferenceNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()