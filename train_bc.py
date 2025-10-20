#!/usr/bin/env python3
import os
import pandas as pd
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
IMG_SIZE = (224, 224)  # (高さ, 幅) - ResNetが期待する標準サイズ

# ★★★ ロボットの関節可動域 (正規化に必須) ★★★
# あなたの 'joint_color_depth_merged.csv' の列名と、
# ロボットの可動域 [最小値, 最大値] (rad) を正確に指定してください。
JOINT_LIMITS = {
    'shoulder_pan_joint':  [math.radians(-20.0), math.radians(110.0)],
    'shoulder_lift_joint': [math.radians(-150.0), math.radians(20.0)],
    'elbow_joint':         [math.radians(-20.0), math.radians(180.0)],
    'wrist_1_joint':       [math.radians(-300.0), math.radians(20.0)],
    'wrist_2_joint':       [math.radians(-150.0), math.radians(80.0)],
    'wrist_3_joint':       [math.radians(-50.0), math.radians(300.0)]
}
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

JOINT_NAMES = list(JOINT_LIMITS.keys())
JOINT_DIM = len(JOINT_NAMES)

# --- データ拡張（Augmentation）の定義 ---
# 学習用データにのみ適用
train_transforms = transforms.Compose([
    # (C, H, W) のTensorに適用
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
    # RandomErasing (三脚が万一映っていても、それに依存するのを防ぐ)
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0),
])

# 検証用データ（正規化のみ）
val_transforms = transforms.Compose([
    # 何も変換しない (もしImageNet正規化するならここに追加)
])

# ==========================================================
# 1. データローダー (BCDataset)
# ==========================================================

def normalize_joint_angles(angles, limits):
    """ 関節角度を -1 から 1 の範囲に正規化する """
    normalized = []
    for i, angle in enumerate(angles):
        joint_name = JOINT_NAMES[i]
        min_val, max_val = limits[joint_name]
        norm_angle = 2 * ((angle - min_val) / (max_val - min_val)) - 1
        normalized.append(norm_angle)
    return np.array(normalized, dtype=np.float32)

def preprocess_image(img_path, target_size, transform):
    """ カラー画像を読み込み、正規化し、拡張を適用する """
    if not os.path.exists(img_path):
        return torch.zeros((3, target_size[0], target_size[1]), dtype=torch.float32)

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        return torch.zeros((3, target_size[0], target_size[1]), dtype=torch.float32)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR -> RGB
    img_resized = cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
    img_normalized = img_resized.astype(np.float32) / 255.0 # 0-255 -> 0.0-1.0
    img_tensor = torch.tensor(img_normalized).permute(2, 0, 1) # (H, W, C) -> (C, H, W)
    
    # データ拡張を適用
    img_tensor = transform(img_tensor)
    
    return img_tensor

class BCDataset(Dataset):
    def __init__(self, demo_dirs, transform):
        """
        demo_dirs: 使用するデモフォルダ (例: ['.../1', '.../3', ...]) のリスト
        transform: 適用する画像変換 (train_transforms or val_transforms)
        """
        self.transform = transform
        self.data = self._load_demos(demo_dirs)

    def _load_demos(self, demo_dirs):
        all_dfs = []
        print(f"Loading {len(demo_dirs)} demos...")
        for demo_dir in tqdm(demo_dirs):
            csv_path = os.path.join(demo_dir, 'joint_color_depth_merged.csv')
            if not os.path.exists(csv_path):
                print(f"Warning: CSV not found in {demo_dir}")
                continue
                
            df = pd.read_csv(csv_path)
            
            # --- ▼ここからが修正▼ ---
            # CSVに書かれたパスが絶対パスか相対パスかに関わらず、
            # 強制的に「親フォルダ名/ファイル名」だけを抜き出す
            def fix_path(p):
                # p = "/home/sho/.../extracted_color_images/image_000.png" (古いPCの絶対パス)
                try:
                    parts = p.replace('\\', '/').split('/')
                    # 最後の2要素 (例: "extracted_color_images", "image_000.png") を取得
                    relative_part = os.path.join(parts[-2], parts[-1])
                except Exception:
                    # パスが変な場合 (例: "image_000.png" のみ)
                    relative_part = os.path.basename(p)

                # "demo_dir" (例: /home/gpu-server/.../1) と結合して
                # "今いるPC" での正しい絶対パスを再構築する
                return os.path.join(demo_dir, relative_part)
            
            # 'fix_path' 関数を color_image 列に適用
            df['image_file_color'] = df['image_file_color'].apply(fix_path)
            # --- ▲ここまでが修正▲ ---
            
            # 必要な列が存在するかチェック
            if not all(col in df.columns for col in JOINT_NAMES):
                print(f"Warning: Missing joint columns in {csv_path}. Skipping.")
                continue

            all_dfs.append(df)
            
        if not all_dfs:
            raise ValueError("No valid data found in the provided directories.")

        full_data = pd.concat(all_dfs, ignore_index=True)
        print(f"Total data points loaded: {len(full_data)}")
        return full_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # --- 入力 (Observation) ---
        color_path = row['image_file']
        color_tensor = preprocess_image(color_path, IMG_SIZE, self.transform)
        
        joint_angles = row[JOINT_NAMES].values.astype(np.float32)
        joint_tensor = normalize_joint_angles(joint_angles, JOINT_LIMITS)
        
        # --- 出力 (Action) ---
        action_tensor = torch.tensor(joint_tensor, dtype=torch.float32)
        
        obs = {
            'color': color_tensor,
            'joint': torch.tensor(joint_tensor, dtype=torch.float32)
        }
        
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
        
        # (オプション) 学習初期はResNetの重みを固定 (ファインチューニング)
        # for param in self.vision_encoder.parameters():
        #     param.requires_grad = False
        
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
            nn.Tanh() # 出力を-1〜1に正規化
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
        
    # '1015_imitation' フォルダ直下の '1', '2', ... '53' などのフォルダを探す
    all_demo_names = [d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d)) and d.isdigit()]
    if len(all_demo_names) < 2:
        raise ValueError(f"Need at least 2 demo folders, but found {len(all_demo_names)} in {args.data_dir}")
        
    all_demo_paths = [os.path.join(args.data_dir, name) for name in all_demo_names]
    random.shuffle(all_demo_paths) # シャッフル

    # 90% を学習用、10% を検証用 (最低1個)
    val_size = max(1, int(len(all_demo_paths) * 0.1))
    train_dirs = all_demo_paths[val_size:]
    val_dirs = all_demo_paths[:val_size]

    print(f"Total demos: {len(all_demo_paths)}")
    print(f"Training demos: {len(train_dirs)}")
    print(f"Validation demos: {len(val_dirs)}")

    # --- 2. データセットとデータローダーの作成 ---
    train_dataset = BCDataset(demo_dirs=train_dirs, transform=train_transforms)
    val_dataset = BCDataset(demo_dirs=val_dirs, transform=val_transforms)

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

    # --- 4. 学習ループ ---
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        print(f"Epoch {epoch+1} Training Loss: {train_loss:.6f}")
        
        val_loss = validate_one_epoch(model, val_loader, loss_fn, device)
        print(f"Epoch {epoch+1} Validation Loss: {val_loss:.6f}")
        
        # ベストモデルの保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = "best_bc_model.pth"
            torch.save(model.state_dict(), save_path)
            print(f"✨ New best model saved to {save_path} (Val Loss: {best_val_loss:.6f})")

    print("\n--- Training Complete ---")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best model saved to best_bc_model.pth")


if __name__ == "__main__":
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description="Train a BC model")
    
    # ★★★ '1015_imitation' フォルダへのパスを指定する ★★★
    parser.add_argument('--data_dir', 
                        type=str, 
                        required=True, 
                        help="Path to the base directory containing demo folders (e.g., '1015_imitation')")
    
    parser.add_argument('--epochs', 
                        type=int, 
                        default=50, 
                        help="Number of training epochs")
    
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=16, 
                        help="Batch size (reduce if you get CUDA OOM error)")
    
    parser.add_argument('--lr', 
                        type=float, 
                        default=1e-4, 
                        help="Learning rate")

    args = parser.parse_args()
    main(args)