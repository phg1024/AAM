cmake_minimum_required(VERSION 2.8.11)

project(MultilinearReconstruction)

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC ON)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

# Boost
find_package(Boost COMPONENTS filesystem timer program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(/usr/local/Cellar/boost/1.60.0_2/lib)
link_libraries(${Boost_LIBRARIES} -lboost_filesystem -lboost_system)

# OpenCV
find_package( OpenCV REQUIRED )
if (OpenCV_FOUND)
    message("-- Found OpenCV")
    message("-- Linking with OpenCV libraries: ${OpenCV_LIBS}")
    link_libraries(${OpenCV_LIBS})
else (OpenCV_FOUND)
    message("-- OpenCV not found.")
endif()

# OpenGL
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS} )
link_libraries(${OPENGL_LIBRARIES} ${GLUT_LIBRARY})

find_package(GLEW REQUIRED)
if (GLEW_FOUND)
    include_directories(${GLEW_INCLUDE_DIRS})
    link_libraries(${GLEW_LIBRARIES})
endif()

# Eigen
find_package(Eigen3)
include_directories(${EIGEN_INCLUDE_DIR})

# Qt5
find_package(Qt5Core)
find_package(Qt5Widgets)
find_package(Qt5OpenGL)
find_package(Qt5Test)

# Targets
add_library(ioutils ioutils.cpp)
target_link_libraries(ioutils
        Qt5::Core
        Qt5::Widgets
        Qt5::OpenGL
        Qt5::Test)

add_library(aammodel aammodel.cpp)
target_link_libraries(aammodel
        ioutils
        Qt5::Core
        Qt5::Widgets
        Qt5::OpenGL
        Qt5::Test)

add_library(fpevaluater fpevaluater.cpp features/vl_hog.cpp)
target_link_libraries(fpevaluater
        ioutils
        Qt5::Core
        Qt5::Widgets
        Qt5::OpenGL
        Qt5::Test)

# Single image reconstruction program
add_executable(AAMFilter aamfilter.cpp common.h ioutils.h utils.h)
target_link_libraries(AAMFilter
        ioutils
        aammodel
        Qt5::Core
        Qt5::Widgets
        Qt5::OpenGL
        Qt5::Test)

# Single image reconstruction program
add_executable(FPFilter fpfilter.cpp common.h ioutils.h utils.h)
target_link_libraries(FPFilter
        ioutils
        fpevaluater
        Qt5::Core
        Qt5::Widgets
        Qt5::OpenGL
        Qt5::Test)

add_subdirectory(tests)
#add_subdirectory(superviseddescent)
