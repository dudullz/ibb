
cmake_minimum_required(VERSION 2.8)

project(IBBDemo)

set(CMAKE_CXX_FLAGS "${CMAKE_CSS_FLAGS} -std=c++11 -stdlib=libc++")

# set(BOOST_ROOT "/Users/llz/Packages/boost_1_55_0/")
set(BOOST_ROOT "D:/dev-packs/boost_1_56_0")

find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED)

message (WARNING "LLZ: Boost_INCLUDES: ${Boost_INCLUDES}")
message (WARNING "LLZ: Boost_LIBS: ${Boost_LIBS}")
message (WARNING "LLZ: Boost_FOUND: ${Boost_FOUND}")

if(${OpenCV_VERSION} VERSION_LESS 2.3.1)
  message (FATAL_ERROR "OpenCV version is not compatible: ${OpenCV_VERSION}")
endif()

file(GLOB sources FrameProcessor.cpp PreProcessor.cpp VideoAnalysis.cpp  VideoCapture.cpp)
file(GLOB main Main.cpp)

file(GLOB_RECURSE bgs package_bgs/*.cpp)
file(GLOB_RECURSE bgs_include package_bgs/*.h)

# GMG is not available in older OpenCV versions
if(${OpenCV_VERSION} VERSION_LESS 2.4.3)
  file(GLOB gmg package_bgs/GMG.cpp)
  list(REMOVE_ITEM bgs ${gmg})
endif()

include_directories(${CMAKE_SOURCE_DIR})

add_library(ibb SHARED ${sources} ${bgs})
target_link_libraries(ibb  ${OpenCV_LIBS})
set_property(TARGET ibb PROPERTY PUBLIC_HEADER ${bgs_include})

add_executable(ibb_bin ${main})
target_link_libraries(ibb_bin ${OpenCV_LIBS} ibb)
set_target_properties(ibb_bin PROPERTIES OUTPUT_NAME ibb)

INSTALL(TARGETS ibb
	ibb_bin
  RUNTIME DESTINATION bin COMPONENT app
  LIBRARY DESTINATION lib COMPONENT runtime
  ARCHIVE DESTINATION lib COMPONENT runtime
  PUBLIC_HEADER DESTINATION include/package_bgs COMPONENT dev
  FRAMEWORK DESTINATION "/Library/Frameworks"
)
