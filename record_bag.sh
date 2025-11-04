arg_one=$1
echo "記録するディレクトリ名"
echo $arg_one

ros2 bag record -o /home/sho/ur_ws/bags/fine/$arg_one \
    /joint_states \
    /devices/ee_camera/realsense_node/color/image_raw \
    /devices/ee_camera/realsense_node/depth/image_rect_raw

