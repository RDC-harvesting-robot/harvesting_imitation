u   arg_one=$1
echo "再生するディレクトリ名"
echo $arg_one
ros2 bag play $arg_one --clock --topics /bbox_image /clock /crop_cordinate /devices/ee_camera/realsense_node/color/camera_info /devices/ee_camera/realsense_node/depth/color/points /devices/ee_camera/realsense_node/rgbd /events/read_split /output /parameter_events