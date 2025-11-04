# ピーマン収穫ロボットのためのBC（模倣学習）モデル

## 実行環境

- Ubuntu / ROS 2
- Python 3.x
- PyTorch
- 必要なライブラリは `requirements.txt` を参照

## 実行
データはharvesting_imitation/extの下においてください

```
./train_bc.py --data_dir ext --batch_size 16
```

```
# 1. 結果保存用フォルダを作成
mkdir -p model_A_naive

# 2. 学習を実行
python3 train_bc_vel.py \
    --data_dir ./fine_tune \
    --save_dir ./model_A_naive \
    --save_name naive_model.pth \
    --lr 1e-4 \
    --epochs 100 \
    --batch_size 64
```
