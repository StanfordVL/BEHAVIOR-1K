cmake_minimum_required(VERSION 2.8.3)

project(turtlebot_description)
find_package(catkin REQUIRED COMPONENTS urdf xacro)

catkin_package(
   CATKIN_DEPENDS urdf xacro
)

install(DIRECTORY robots
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
install(DIRECTORY meshes
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
install(DIRECTORY test
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
install(DIRECTORY urdf
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

catkin_add_gtest(test_urdf test/test_urdf.cpp)

