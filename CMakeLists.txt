
cmake_minimum_required(VERSION 2.8)

project(IBBDemo)

set(CMAKE_CXX_FLAGS "${CMAKE_CSS_FLAGS} -std=c++11 -stdlib=libc++")

set(Boost_NO_SYSTEM_PATHS "True")
set(Boost_NO_BOOST_CMAKE "True")
set(BOOST_ROOT "/Users/llz/Packages/boost_1_55_0/build-boost155/")
# set(BOOST_ROOT "D:/dev-packs/boost_1_56_0")

set(OpenCV_CONFIG_PATH "/Users/llz/Packages/opencv-2.4.8/opencv-build/")
set(OpenCV_DIR "/Users/llz/Packages/opencv-2.4.8/opencv-build/")

find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem)

message (WARNING "LLZ: Boost_INCLUDE_DIR: ${Boost_INCLUDE_DIR}")
message (WARNING "LLZ: Boost_LIBRARIES: ${Boost_LIBRARIES}")
message (WARNING "LLZ: Boost_FOUND: ${Boost_FOUND}")
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

include_directories(${CMAKE_SOURCE_DIR} ${Boost_INCLUDE_DIR})

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
