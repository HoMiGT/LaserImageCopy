cmake_minimum_required(VERSION 4.1)

set(VCPKG_ROOT "E:/vcpkg" CACHE STRING "vcpkg root")

list(APPEND CMAKE_PREFIX_PATH "E:/3rdPackages/TensorRT-10.1.0.27")
list(APPEND CMAKE_PREFIX_PATH "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5")
list(APPEND CMAKE_PREFIX_PATH "${VCPKG_INSTALLED_DIR}/x64-windows/")
set(CMAKE_TOOLCHAIN_FILE "${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "vcpkg toolchain file")
set(Protobuf_PROTOC_EXECUTABLE "${VCPKG_INSTALLED_DIR}/x64-windows/tools/protobuf/protoc.exe" CACHE STRING "Protobuf compiler")


project(LaserImageCopy VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TensorRT_INCLUDE_DIR "E:/3rdPackages/TensorRT-10.1.0.27/include")
set(TensorRT_DIR "E:/3rdPackages/TensorRT-10.1.0.27/lib")
list(APPEND TensorRT_LIBS
		"${TensorRT_DIR}/nvinfer_10.lib"
		"${TensorRT_DIR}/nvinfer_dispatch_10.lib"
		"${TensorRT_DIR}/nvinfer_lean_10.lib"
		"${TensorRT_DIR}/nvinfer_plugin_10.lib"
		"${TensorRT_DIR}/nvinfer_vc_plugin_10.lib"
		"${TensorRT_DIR}/nvonnxparser_10.lib"
)

set(CUDA_INCLUDE_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/include")
set(CUDA_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/lib/x64")
file(GLOB CUDA_LIBS "${CUDA_DIR}/*.lib")

find_package(nlohmann_json CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
set(OpenCV_ROOT "${VCPKG_INSTALLED_DIR}/share/opencv4")
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(Protobuf REQUIRED protobuf)

message(STATUS "vcpkg installed dir: ${VCPKG_INSTALLED_DIR}")
set(ZBAR_INCLUDE_DIR "${VCPKG_INSTALLED_DIR}/x64-windows/include")
set(ZBAR_LIB_DIR "${VCPKG_INSTALLED_DIR}/x64-windows/lib")
set(ZBAR_LIBRARIES "${ZBAR_LIB_DIR}/zbar.lib")

set(Indicators_ZIP "file://D:/Downloads/indicators-master.zip")
include(FetchContent)
FetchContent_Declare(
		indicators
		URL ${Indicators_ZIP}
)
FetchContent_MakeAvailable(indicators)

set(PROJECT_SOURCES
	main.cpp
)

add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})

target_sources(${PROJECT_NAME} PRIVATE
		FILE_SET cxx_modules
		TYPE CXX_MODULES
		BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
		FILES
			logger.cppm
		    copier.cppm
			detector.cppm
)
target_include_directories(${PROJECT_NAME} PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}
		${OpenCV_INCLUDE_DIRS}
		${ZBAR_INCLUDE_DIRS}
		${TensorRT_INCLUDE_DIR}
		${CUDA_INCLUDE_DIR}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
		nlohmann_json::nlohmann_json
		spdlog::spdlog
		indicators::indicators
		${OpenCV_LIBS}
		${ZBAR_LIBRARIES}
		${TensorRT_LIBS}
		${CUDA_LIBS}
)

target_compile_options(${PROJECT_NAME} PRIVATE
	${ZBAR_CFLAGS_OTHER}
)

if (WIN32)
	add_custom_target(run
			COMMAND ${CMAKE_COMMAND} -E echo "Launching in new terminal..."
			COMMAND cmd /c start "" cmd /k "$<TARGET_FILE:${PROJECT_NAME}>"
			DEPENDS ${PROJECT_NAME}
			WORKING_DIRECTORY $<TARGET_FILE_DIR:${PROJECT_NAME}>
 	)
endif()