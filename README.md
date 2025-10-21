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
