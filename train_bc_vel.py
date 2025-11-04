#!/usr/bin/env python3
import os
import pandas as pd # ★ pandas が main で必要
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
from tqdm import tqdm
import argparse
import random
import math

# ==========================================================
# ★★★ ユーザー設定項目 (重要) ★★★
# ==========================================================
IMG_SIZE = (224, 224)  # (高さ, 幅)

# --- 関節の可動域 (入力の正規化用) ---
JOINT_LIMITS = {
    'shoulder_pan_joint':  [math.radians(-20.0), math.radians(110.0)],
    'shoulder_lift_joint': [math.radians(-150.0), math.radians(20.0)],
    'elbow_joint':         [math.radians(-20.0), math.radians(180.0)],
    'wrist_1_joint':       [math.radians(-300.0), math.radians(20.0)],
    'wrist_2_joint':       [math.radians(-150.0), math.radians(80.0)],
    'wrist_3_joint':       [math.radians(-50.0), math.radians(300.0)]
}
JOINT_NAMES = list(JOINT_LIMITS.keys())
JOINT_DIM = len(JOINT_NAMES)

# --- 関節の「最大速度」 (出力の正規化用) ---
# (これは例です。あなたのロボットの安全な最大速度 [rad/s] に合わせてください)
MAX_VELOCITY = 2.0  # 全関節共通の最大速度 [rad/s] と仮定

# --- データ拡張の定義 ---
train_transforms = transforms.Compose([
    # (Sim-to-Realギャップ対策)
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0),
])
val_transforms = transforms.Compose([]) # 検証用に拡張はしない

# ==========================================================
# 1. データローダー (BCDataset) - ★速度制御バージョン★
# ==========================================================

def normalize_joint_angles(angles, limits):
    """ 関節角度(rad) を -1 から 1 の範囲に正規化する """
    normalized = []
    for i, angle in enumerate(angles):
        joint_name = JOINT_NAMES[i]
        min_val, max_val = limits[joint_name]
        # ゼロ除算を避ける
        if max_val - min_val == 0:
            norm_angle = 0.0
        else:
            norm_angle = 2 * ((angle - min_val) / (max_val - min_val)) - 1
        normalized.append(norm_angle)
    return np.array(normalized, dtype=np.float32)

def normalize_joint_velocity(velocities, max_vel):
    """ 関節速度(rad/s) を -1 から 1 の範囲に正規化する """
    if max_vel == 0:
        return np.zeros_like(velocities, dtype=np.float32)
    normalized = velocities / max_vel
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)

def preprocess_image(img_path, target_size, transform):
    """ カラー画像を読み込み、正規化し、拡張を適用する """
    if not os.path.exists(img_path):
        return torch.zeros((3, target_size[0], target_size[1]), dtype=torch.float32)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        return torch.zeros((3, target_size[0], target_size[1]), dtype=torch.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_tensor = torch.tensor(img_normalized).permute(2, 0, 1) # (H, W, C) -> (C, H, W)
    img_tensor = transform(img_tensor)
    return img_tensor

class BCDataset(Dataset):
    def __init__(self, demo_dirs, transform, is_train=True):
        self.transform = transform
        self.is_train = is_train
        self.data = self._load_demos(demo_dirs)

    def _load_demos(self, demo_dirs):
        all_dfs = []
        print(f"Loading {len(demo_dirs)} demos and calculating velocities...")
        for demo_dir in tqdm(demo_dirs):
            # (DepthなしバージョンのCSV名)
            csv_path = os.path.join(demo_dir, 'joint_color_merged.csv') 
            if not os.path.exists(csv_path):
                print(f"Warning: CSV 'joint_color_merged.csv' not found in {demo_dir}")
                continue
                
            df = pd.read_csv(csv_path)
            
            # --- 速度の計算 ---
            df.sort_values("relative_time", inplace=True)
            df['delta_time'] = df['relative_time'].diff()
            
            skip_demo = False
            for joint in JOINT_NAMES:
                if joint not in df.columns:
                    print(f"Warning: Joint '{joint}' not in {csv_path}. Skipping demo.")
                    skip_demo = True
                    break
                df[f'delta_{joint}'] = df[joint].diff()
                df[f'vel_{joint}'] = df[f'delta_{joint}'] / df['delta_time']
            
            if skip_demo:
                continue
                
            # 最初の行 (NaN) と、時間が飛んだ行を削除
            df.dropna(inplace=True) 
            df = df[df['delta_time'] < 1.0] # 1秒以上のジャンプは削除
            # --- 速度計算ここまで ---

            # (画像パスの修正)
            def fix_path(p):
                try:
                    parts = p.replace('\\', '/').split('/')
                    relative_part = os.path.join(parts[-2], parts[-1])
                except Exception:
                    relative_part = os.path.basename(p)
                return os.path.join(demo_dir, relative_part)
            
            if 'image_file' not in df.columns:
                 print(f"Warning: 'image_file' column not in {csv_path}. Skipping.")
                 continue
            df['image_file'] = df['image_file'].apply(fix_path)
            
            all_dfs.append(df)
            
        if not all_dfs:
            raise ValueError("No valid data found or velocities could be calculated.")

        full_data = pd.concat(all_dfs, ignore_index=True)
        print(f"Total valid data points (with velocity): {len(full_data)}")
        return full_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # --- 入力 (Observation) ---
        # 1. 画像
        color_path = row['image_file']
        color_tensor = preprocess_image(color_path, IMG_SIZE, self.transform)
        
        # 2. 現在の関節角度 (ノイズあり) - 「揺れ」対策
        clean_joint_angles = row[JOINT_NAMES].values.astype(np.float32)
        obs_joint_angles = clean_joint_angles.copy()
        if self.is_train:
            noise = np.random.uniform(-0.01, 0.01, size=obs_joint_angles.shape).astype(np.float32)
            obs_joint_angles += noise
        
        norm_obs_joint = normalize_joint_angles(obs_joint_angles, JOINT_LIMITS)

        # --- 出力 (Action) - 速度 ---
        velocity_names = [f'vel_{joint}' for joint in JOINT_NAMES]
        joint_velocities = row[velocity_names].values.astype(np.float32)
        norm_action_velocity = normalize_joint_velocity(joint_velocities, MAX_VELOCITY)
        
        obs = {
            'color': color_tensor,
            'joint': torch.tensor(norm_obs_joint, dtype=torch.float32) # 入力 (位置)
        }
        
        action_tensor = torch.tensor(norm_action_velocity, dtype=torch.float32) # 出力 (速度)
        
        return obs, action_tensor

# ==========================================================
# 2. BCモデル (BCModel)
# ==========================================================
class BCModel(nn.Module):
    def __init__(self, joint_dim=JOINT_DIM, action_dim=JOINT_DIM):
        super().__init__()
        self.joint_dim = joint_dim
        self.action_dim = action_dim
        
        # 1. ビジョンエンコーダ (ImageNet学習済みResNet-18)
        resnet = models.resnet18(pretrained=True)
        self.vision_encoder = nn.Sequential(*list(resnet.children())[:-1])
        vision_feature_dim = 512
        
        # 2. 関節角度エンコーダ (MLP)
        self.joint_encoder = nn.Sequential(
            nn.Linear(self.joint_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )
        joint_feature_dim = 64
        
        # 3. 結合ヘッド (Fusion Head)
        combined_feature_dim = vision_feature_dim + joint_feature_dim
        
        self.head = nn.Sequential(
            nn.Linear(combined_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim),
            nn.Tanh() # ★速度制御のキモ: 出力を-1〜1に正規化
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
# 3. 学習・検証ループ
# ==========================================================

def train_one_epoch(model, data_loader, optimizer, loss_fn, device):
    model.train() # 学習モード
    total_loss = 0.0
    for obs, action in tqdm(data_loader, desc="Training"):
        # データをGPUへ
        obs_gpu = {'color': obs['color'].to(device), 'joint': obs['joint'].to(device)}
        action_gpu = action.to(device)

        # 予測
        predicted_action = model(obs_gpu)
        loss = loss_fn(predicted_action, action_gpu)

        # 逆伝播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(data_loader)

def validate_one_epoch(model, data_loader, loss_fn, device):
    model.eval() # 評価モード (Dropoutなどが無効になる)
    total_loss = 0.0
    with torch.no_grad(): # 勾配計算を無効化
        for obs, action in tqdm(data_loader, desc="Validating"):
            obs_gpu = {'color': obs['color'].to(device), 'joint': obs['joint'].to(device)}
            action_gpu = action.to(device)
            predicted_action = model(obs_gpu)
            loss = loss_fn(predicted_action, action_gpu)
            total_loss += loss.item()
            
    return total_loss / len(data_loader)

# ==========================================================
# 4. メイン実行部
# ==========================================================

def main(args):
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. デモフォルダのリストアップと分割 ---
    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
        
    # 'data_dir' フォルダ直下の数字のフォルダを探す
    all_demo_names = [d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d)) and d.isdigit()]
    if len(all_demo_names) < 2:
        # (ファインチューニング用にデモが1つでも動くように < 1 に変更)
        if len(all_demo_names) < 1:
            raise ValueError(f"Need at least 1 demo folder, but found {len(all_demo_names)} in {args.data_dir}")
        
    all_demo_paths = [os.path.join(args.data_dir, name) for name in all_demo_names]
    random.shuffle(all_demo_paths) # シャッフル

    # 90% を学習用、10% を検証用 (最低1個)
    val_size = max(1, int(len(all_demo_paths) * 0.1))
    
    # (デモが1つしかない場合は、それを学習と検証の両方に使う)
    if len(all_demo_paths) == 1:
        train_dirs = all_demo_paths
        val_dirs = all_demo_paths
    else:
        train_dirs = all_demo_paths[val_size:]
        val_dirs = all_demo_paths[:val_size]

    print(f"Total demos found: {len(all_demo_paths)}")
    print(f"Training demos: {len(train_dirs)}")
    print(f"Validation demos: {len(val_dirs)}")

    # --- 2. データセットとデータローダーの作成 ---
    train_dataset = BCDataset(demo_dirs=train_dirs, transform=train_transforms, is_train=True)
    val_dataset = BCDataset(demo_dirs=val_dirs, transform=val_transforms, is_train=False)

    train_loader = DataLoader(train_dataset, 
                              batch_size=args.batch_size, 
                              shuffle=True, 
                              num_workers=4, 
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, 
                            batch_size=args.batch_size, 
                            shuffle=False, 
                            num_workers=4, 
                            pin_memory=True)

    # --- 3. モデル、オプティマイザ、損失関数の準備 ---
    model = BCModel(joint_dim=JOINT_DIM, action_dim=JOINT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss() # 平均二乗誤差

    # --- 4. ログ保存用のリストを初期化 ---
    training_log = []
    best_val_loss = float('inf')
    
    # (ファインチューニング用: 事前学習モデルの読み込み)
    if args.load_model:
        if os.path.exists(args.load_model):
            try:
                model.load_state_dict(torch.load(args.load_model, map_location=device))
                print(f"Successfully loaded pre-trained model from {args.load_model}")
            except Exception as e:
                print(f"Warning: Failed to load pre-trained model. Starting from scratch. Error: {e}")
        else:
            print(f"Warning: --load_model path '{args.load_model}' not found. Starting from scratch.")

    # --- 5. 学習ループ ---
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        print(f"Epoch {epoch+1} Training Loss: {train_loss:.6f}")
        
        val_loss = validate_one_epoch(model, val_loader, loss_fn, device)
        print(f"Epoch {epoch+1} Validation Loss: {val_loss:.6f}")
        
        # ログリストにエポックごとの結果を追加
        training_log.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss
        })

        # ベストモデルの保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(args.save_dir, args.save_name) 
            torch.save(model.state_dict(), save_path)
            print(f"✨ New best model saved to {save_path} (Val Loss: {best_val_loss:.6f})")

    print("\n--- Training Complete ---")
    print(f"Best validation loss: {best_val_loss:.6f}")
    
    # --- 6. ログをCSVファイルに保存 ---
    log_df = pd.DataFrame(training_log)
    log_path = os.path.join(args.save_dir, f"{args.save_name}.log.csv") # ★ログファイル名もsave_nameに紐付け
    try:
        log_df.to_csv(log_path, index=False)
        print(f"✅ Training log saved to: {log_path}")
    except Exception as e:
        print(f"Error saving training log: {e}")


if __name__ == "__main__":
    # --- 引数の設定 ---
    parser = argparse.ArgumentParser(description="Train a BC (Velocity) model")
    
    parser.add_argument('--data_dir', 
                        type=str, 
                        required=True, 
                        help="Path to the base directory containing demo folders (e.g., './fine_tune' or './ext')")
    
    parser.add_argument('--save_dir', 
                        type=str, 
                        default=".", 
                        help="Directory to save the model and logs (default: current directory)")
    
    parser.add_argument('--save_name', 
                        type=str, 
                        default="best_bc_velocity_model.pth", 
                        help="Filename for the best model (e.g., 'naive_model.pth' or 'finetune_model.pth')")

    parser.add_argument('--load_model', 
                        type=str, 
                        default=None, 
                        help="Path to a pre-trained model to load for fine-tuning (e.g., './model_A_naive/naive_model.pth')")

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size (adjust to your GPU VRAM)")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate (use a smaller value like 1e-5 for fine-tuning)")

    args = parser.parse_args()
    
    # 保存先ディレクトリがなければ作成
    os.makedirs(args.save_dir, exist_ok=True)
    
    main(args)