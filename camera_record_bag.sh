arg_one=$1
echo "記録するディレクトリ名"
echo $arg_one

ros2 bag record -o /media/sho/ssdkun/250825_bags/$arg_one \
    /bbox_image \
    /crop_cordinate \
    /devices/ee_camera/realsense_node/rgbd \
    /devices/ee_camera/realsense_node/color/camera_info \
    /camera_info \
    /devices/ee_camera/realsense_node/depth/color/points \