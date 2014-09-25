
cmake_minimum_required(VERSION 2.8)

project(IBBDemo)

set(CMAKE_CXX_FLAGS "${CMAKE_CSS_FLAGS} -std=c++11 -stdlib=libc++")

# set(Boost_NO_SYSTEM_PATHS ON)
# set(Boost_NO_BOOST_CMAKE ON)
# set(BOOST_ROOT "/Users/llz/Packages/boost_1_55_0/build-boost155/")

set(BOOST_DIR "D:/dev-packs/boost_1_56_0")
### Under Windows, boost libraries are compiled with the name of many more extensions than just being the 'libboost_filesystem',
### but rather being 'libboost_filesystem-vc120-mt-1_56.lib'
### Therefore even specify the correct path of where the libaries are saved to BOOST_LIBRARYDIR, it still complains - STUPID!!!
set(BOOST_ROOT "D:/dev-packs/boost_1_56_0")
set(BOOST_LIBRARYDIR "D:/dev-packs/boost_1_56_0/")

# set(OpenCV_CONFIG_PATH "/Users/llz/Packages/opencv-2.4.8/opencv-build/")
# set(OpenCV_DIR "/Users/llz/Packages/opencv-2.4.8/opencv-build/")
set(OpenCV_CONFIG_PATH "D:/dev-packs/OpenCV-2.4.8/")
set(OpenCV_DIR "D:/dev-packs/OpenCV-2.4.8/")

find_package(OpenCV REQUIRED)

### for Windows, specify filesystem component won't work, maybe because of customised installation path
find_package(Boost REQUIRED)
# find_package(Boost REQUIRED COMPONENTS system filesystem)

### Under Windows (virgil W8.1), I must 'installed' opencv 2.4.9 before so find_package insists on 2.4.9 settings
### Have to manually overwrite it with 2.4.8 settings
set(OpenCV_INCLUDE_DIRS "D:\\dev-packs\\OpenCV-2.4.8\\opencv\\build\\include\\opencv" "D:/dev-packs/OpenCV-2.4.8/opencv/build/include")
set(OpenCV_LIBS "D:\\dev-packs\\OpenCV-2.4.8\\opencv\\build\\x64\\vc12\\lib\\opencv_videostab248.lib" \\
"D:\\Dev-packs\\OpenCV-2.4.8\\opencv\\build\\x64\\vc12\\lib\\opencv_video248.lib" \\
"D:\\Dev-packs\\OpenCV-2.4.8\\opencv\\build\\x64\\vc12\\lib\\opencv_contrib248.lib" \\
"D:\\Dev-packs\\OpenCV-2.4.8\\opencv\\build\\x64\\vc12\\lib\\opencv_ts248.lib" \\
"D:\\Dev-packs\\OpenCV-2.4.8\\opencv\\build\\x64\\vc12\\lib\\opencv_superres248.lib"  \\
"D:\\Dev-packs\\OpenCV-2.4.8\\opencv\\build\\x64\\vc12\\lib\\opencv_stitching248.lib" \\
"D:\\Dev-packs\\OpenCV-2.4.8\\opencv\\build\\x64\\vc12\\lib\\opencv_photo248.lib" \\
"D:\\Dev-packs\\OpenCV-2.4.8\\opencv\\build\\x64\\vc12\\lib\\opencv_objdetect248.lib" \\
"D:\\Dev-packs\\OpenCV-2.4.8\\opencv\\build\\x64\\vc12\\lib\\opencv_ml248.lib" \\
"D:\\Dev-packs\\OpenCV-2.4.8\\opencv\\build\\x64\\vc12\\lib\\opencv_imgproc248.lib" \\
"D:\\Dev-packs\\OpenCV-2.4.8\\opencv\\build\\x64\\vc12\\lib\\opencv_highgui248.lib" \\
"D:\\Dev-packs\\OpenCV-2.4.8\\opencv\\build\\x64\\vc12\\lib\\opencv_flann248.lib" \\
"D:\\Dev-packs\\OpenCV-2.4.8\\opencv\\build\\x64\\vc12\\lib\\opencv_features2d248.lib" \\
"D:\\Dev-packs\\OpenCV-2.4.8\\opencv\\build\\x64\\vc12\\lib\\opencv_core248.lib" \\
"D:\\Dev-packs\\OpenCV-2.4.8\\opencv\\build\\x64\\vc12\\lib\\opencv_calib3d248.lib" )

set(Boost_LIBRARY_DIR "D:/dev-packs/boost_1_56_0/stage/lib")

message (STATUS "LLZ: BOOST_DIR: ${BOOST_DIR}")
message (STATUS "LLZ: BOOST_LIBRARYDIR: ${BOOST_LIBRARYDIR}")
message (WARNING "LLZ: Boost_LIBRARY_DIR: ${Boost_LIBRARY_DIR}")
message (WARNING "LLZ: Boost_LIBRARY_DIRS: ${Boost_LIBRARY_DIRS}")
message (WARNING "LLZ: Boost_INCLUDE_DIR: ${Boost_INCLUDE_DIR}")
message (WARNING "LLZ: Boost_LIBRARIES: ${Boost_LIBRARIES}")
message (WARNING "LLZ: Boost_FOUND: ${Boost_FOUND}")
message (WARNING "LLZ: OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message (WARNING "LLZ: OpenCV_LIBS: ${OpenCV_LIBS}")

if(${OpenCV_VERSION} VERSION_LESS 2.3.1)
  message (FATAL_ERROR "OpenCV version is not compatible: ${OpenCV_VERSION}")
endif()

file(GLOB sources FrameProcessor.cpp PreProcessor.cpp VideoAnalysis.cpp  VideoCapture.cpp DynamicTimeWarping.cpp)
file(GLOB main Main.cpp)

file(GLOB_RECURSE bgs package_bgs/*.cpp)
file(GLOB_RECURSE bgs_include package_bgs/*.h)

# GMG is not available in older OpenCV versions
if(${OpenCV_VERSION} VERSION_LESS 2.4.3)
  file(GLOB gmg package_bgs/GMG.cpp)
  list(REMOVE_ITEM bgs ${gmg})
endif()

include_directories(${CMAKE_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIR})

add_library(ibb SHARED ${sources} ${bgs})
target_link_libraries(ibb  ${OpenCV_LIBS} ${Boost_LIBRARIES})
set_property(TARGET ibb PROPERTY PUBLIC_HEADER ${bgs_include})

add_executable(ibb_bin ${main})
target_link_libraries(ibb_bin ${OpenCV_LIBS} ${Boost_LIBRARIES} ibb)
set_target_properties(ibb_bin PROPERTIES OUTPUT_NAME ibb)

INSTALL(TARGETS ibb
	ibb_bin
  RUNTIME DESTINATION bin COMPONENT app
  LIBRARY DESTINATION lib COMPONENT runtime
  ARCHIVE DESTINATION lib COMPONENT runtime
  PUBLIC_HEADER DESTINATION include/package_bgs COMPONENT dev
  FRAMEWORK DESTINATION "/Library/Frameworks"
)
