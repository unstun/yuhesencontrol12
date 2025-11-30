#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>

int main(int argc, char** argv){
  ros::init(argc, argv, "Trans_TF_2d");

  ros::NodeHandle node;

  tf::TransformListener listener;
  tf::TransformBroadcaster broadcaster;
  tf::Transform transform_broadcaster;
  ros::Duration(1.0).sleep();  // 等一下 TF

  ros::Rate rate(1000);
  while (node.ok()){
    tf::StampedTransform transform_listener;
    
    try{
      // 监听 map -> body 的 3D 位姿（这是 FAST_LIO 给你的）
      listener.lookupTransform("map", "body",  
                               ros::Time(0), transform_listener);
    }
    catch (tf::TransformException ex){
      ROS_ERROR("%s",ex.what());
      ros::Duration(1.0).sleep();
      continue;   // 注意：这里最好 continue 一下，避免下面用到未定义数据
    }

    float robot_oriation_z=transform_listener.getRotation().getZ();
    float robot_oriation_w=transform_listener.getRotation().getW();

    // 只取 x、y，z 强制为 0（投到地面）
    transform_broadcaster.setOrigin(
        tf::Vector3(transform_listener.getOrigin().x(),
                    transform_listener.getOrigin().y(),
                    0.0));

    // 只保留 z 轴的旋转，把 pitch/roll 设为 0
    transform_broadcaster.setRotation(
        tf::Quaternion(0, 0, robot_oriation_z, robot_oriation_w));

    // 发布 map -> body_2d
    broadcaster.sendTransform(tf::StampedTransform(
        transform_broadcaster, ros::Time::now(), "map", "body_2d"));

    rate.sleep();
  }
  return 0;
}
