# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/alientek/example/32_rknn_retinaface_demo_video/rknn_retinaface_demo

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/alientek/example/32_rknn_retinaface_demo_video/rknn_retinaface_demo/build/build_linux_aarch64

# Include any dependencies generated for this target.
include CMakeFiles/rknn_retinaface_demo.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/rknn_retinaface_demo.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/rknn_retinaface_demo.dir/flags.make

CMakeFiles/rknn_retinaface_demo.dir/src/main.cc.o: CMakeFiles/rknn_retinaface_demo.dir/flags.make
CMakeFiles/rknn_retinaface_demo.dir/src/main.cc.o: ../../src/main.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/alientek/example/32_rknn_retinaface_demo_video/rknn_retinaface_demo/build/build_linux_aarch64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/rknn_retinaface_demo.dir/src/main.cc.o"
	/opt/atk-dlrk356x-toolchain/usr/bin/aarch64-buildroot-linux-gnu-c++ --sysroot=/opt/atk-dlrk356x-toolchain/aarch64-buildroot-linux-gnu/sysroot  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/rknn_retinaface_demo.dir/src/main.cc.o -c /home/alientek/example/32_rknn_retinaface_demo_video/rknn_retinaface_demo/src/main.cc

CMakeFiles/rknn_retinaface_demo.dir/src/main.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/rknn_retinaface_demo.dir/src/main.cc.i"
	/opt/atk-dlrk356x-toolchain/usr/bin/aarch64-buildroot-linux-gnu-c++ --sysroot=/opt/atk-dlrk356x-toolchain/aarch64-buildroot-linux-gnu/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/alientek/example/32_rknn_retinaface_demo_video/rknn_retinaface_demo/src/main.cc > CMakeFiles/rknn_retinaface_demo.dir/src/main.cc.i

CMakeFiles/rknn_retinaface_demo.dir/src/main.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/rknn_retinaface_demo.dir/src/main.cc.s"
	/opt/atk-dlrk356x-toolchain/usr/bin/aarch64-buildroot-linux-gnu-c++ --sysroot=/opt/atk-dlrk356x-toolchain/aarch64-buildroot-linux-gnu/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/alientek/example/32_rknn_retinaface_demo_video/rknn_retinaface_demo/src/main.cc -o CMakeFiles/rknn_retinaface_demo.dir/src/main.cc.s

CMakeFiles/rknn_retinaface_demo.dir/src/retinaface.cc.o: CMakeFiles/rknn_retinaface_demo.dir/flags.make
CMakeFiles/rknn_retinaface_demo.dir/src/retinaface.cc.o: ../../src/retinaface.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/alientek/example/32_rknn_retinaface_demo_video/rknn_retinaface_demo/build/build_linux_aarch64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/rknn_retinaface_demo.dir/src/retinaface.cc.o"
	/opt/atk-dlrk356x-toolchain/usr/bin/aarch64-buildroot-linux-gnu-c++ --sysroot=/opt/atk-dlrk356x-toolchain/aarch64-buildroot-linux-gnu/sysroot  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/rknn_retinaface_demo.dir/src/retinaface.cc.o -c /home/alientek/example/32_rknn_retinaface_demo_video/rknn_retinaface_demo/src/retinaface.cc

CMakeFiles/rknn_retinaface_demo.dir/src/retinaface.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/rknn_retinaface_demo.dir/src/retinaface.cc.i"
	/opt/atk-dlrk356x-toolchain/usr/bin/aarch64-buildroot-linux-gnu-c++ --sysroot=/opt/atk-dlrk356x-toolchain/aarch64-buildroot-linux-gnu/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/alientek/example/32_rknn_retinaface_demo_video/rknn_retinaface_demo/src/retinaface.cc > CMakeFiles/rknn_retinaface_demo.dir/src/retinaface.cc.i

CMakeFiles/rknn_retinaface_demo.dir/src/retinaface.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/rknn_retinaface_demo.dir/src/retinaface.cc.s"
	/opt/atk-dlrk356x-toolchain/usr/bin/aarch64-buildroot-linux-gnu-c++ --sysroot=/opt/atk-dlrk356x-toolchain/aarch64-buildroot-linux-gnu/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/alientek/example/32_rknn_retinaface_demo_video/rknn_retinaface_demo/src/retinaface.cc -o CMakeFiles/rknn_retinaface_demo.dir/src/retinaface.cc.s

CMakeFiles/rknn_retinaface_demo.dir/src/file_utils.cc.o: CMakeFiles/rknn_retinaface_demo.dir/flags.make
CMakeFiles/rknn_retinaface_demo.dir/src/file_utils.cc.o: ../../src/file_utils.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/alientek/example/32_rknn_retinaface_demo_video/rknn_retinaface_demo/build/build_linux_aarch64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/rknn_retinaface_demo.dir/src/file_utils.cc.o"
	/opt/atk-dlrk356x-toolchain/usr/bin/aarch64-buildroot-linux-gnu-c++ --sysroot=/opt/atk-dlrk356x-toolchain/aarch64-buildroot-linux-gnu/sysroot  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/rknn_retinaface_demo.dir/src/file_utils.cc.o -c /home/alientek/example/32_rknn_retinaface_demo_video/rknn_retinaface_demo/src/file_utils.cc

CMakeFiles/rknn_retinaface_demo.dir/src/file_utils.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/rknn_retinaface_demo.dir/src/file_utils.cc.i"
	/opt/atk-dlrk356x-toolchain/usr/bin/aarch64-buildroot-linux-gnu-c++ --sysroot=/opt/atk-dlrk356x-toolchain/aarch64-buildroot-linux-gnu/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/alientek/example/32_rknn_retinaface_demo_video/rknn_retinaface_demo/src/file_utils.cc > CMakeFiles/rknn_retinaface_demo.dir/src/file_utils.cc.i

CMakeFiles/rknn_retinaface_demo.dir/src/file_utils.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/rknn_retinaface_demo.dir/src/file_utils.cc.s"
	/opt/atk-dlrk356x-toolchain/usr/bin/aarch64-buildroot-linux-gnu-c++ --sysroot=/opt/atk-dlrk356x-toolchain/aarch64-buildroot-linux-gnu/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/alientek/example/32_rknn_retinaface_demo_video/rknn_retinaface_demo/src/file_utils.cc -o CMakeFiles/rknn_retinaface_demo.dir/src/file_utils.cc.s

CMakeFiles/rknn_retinaface_demo.dir/src/image_drawing.cc.o: CMakeFiles/rknn_retinaface_demo.dir/flags.make
CMakeFiles/rknn_retinaface_demo.dir/src/image_drawing.cc.o: ../../src/image_drawing.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/alientek/example/32_rknn_retinaface_demo_video/rknn_retinaface_demo/build/build_linux_aarch64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/rknn_retinaface_demo.dir/src/image_drawing.cc.o"
	/opt/atk-dlrk356x-toolchain/usr/bin/aarch64-buildroot-linux-gnu-c++ --sysroot=/opt/atk-dlrk356x-toolchain/aarch64-buildroot-linux-gnu/sysroot  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/rknn_retinaface_demo.dir/src/image_drawing.cc.o -c /home/alientek/example/32_rknn_retinaface_demo_video/rknn_retinaface_demo/src/image_drawing.cc

CMakeFiles/rknn_retinaface_demo.dir/src/image_drawing.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/rknn_retinaface_demo.dir/src/image_drawing.cc.i"
	/opt/atk-dlrk356x-toolchain/usr/bin/aarch64-buildroot-linux-gnu-c++ --sysroot=/opt/atk-dlrk356x-toolchain/aarch64-buildroot-linux-gnu/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/alientek/example/32_rknn_retinaface_demo_video/rknn_retinaface_demo/src/image_drawing.cc > CMakeFiles/rknn_retinaface_demo.dir/src/image_drawing.cc.i

CMakeFiles/rknn_retinaface_demo.dir/src/image_drawing.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/rknn_retinaface_demo.dir/src/image_drawing.cc.s"
	/opt/atk-dlrk356x-toolchain/usr/bin/aarch64-buildroot-linux-gnu-c++ --sysroot=/opt/atk-dlrk356x-toolchain/aarch64-buildroot-linux-gnu/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/alientek/example/32_rknn_retinaface_demo_video/rknn_retinaface_demo/src/image_drawing.cc -o CMakeFiles/rknn_retinaface_demo.dir/src/image_drawing.cc.s

CMakeFiles/rknn_retinaface_demo.dir/src/image_utils.cc.o: CMakeFiles/rknn_retinaface_demo.dir/flags.make
CMakeFiles/rknn_retinaface_demo.dir/src/image_utils.cc.o: ../../src/image_utils.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/alientek/example/32_rknn_retinaface_demo_video/rknn_retinaface_demo/build/build_linux_aarch64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/rknn_retinaface_demo.dir/src/image_utils.cc.o"
	/opt/atk-dlrk356x-toolchain/usr/bin/aarch64-buildroot-linux-gnu-c++ --sysroot=/opt/atk-dlrk356x-toolchain/aarch64-buildroot-linux-gnu/sysroot  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/rknn_retinaface_demo.dir/src/image_utils.cc.o -c /home/alientek/example/32_rknn_retinaface_demo_video/rknn_retinaface_demo/src/image_utils.cc

CMakeFiles/rknn_retinaface_demo.dir/src/image_utils.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/rknn_retinaface_demo.dir/src/image_utils.cc.i"
	/opt/atk-dlrk356x-toolchain/usr/bin/aarch64-buildroot-linux-gnu-c++ --sysroot=/opt/atk-dlrk356x-toolchain/aarch64-buildroot-linux-gnu/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/alientek/example/32_rknn_retinaface_demo_video/rknn_retinaface_demo/src/image_utils.cc > CMakeFiles/rknn_retinaface_demo.dir/src/image_utils.cc.i

CMakeFiles/rknn_retinaface_demo.dir/src/image_utils.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/rknn_retinaface_demo.dir/src/image_utils.cc.s"
	/opt/atk-dlrk356x-toolchain/usr/bin/aarch64-buildroot-linux-gnu-c++ --sysroot=/opt/atk-dlrk356x-toolchain/aarch64-buildroot-linux-gnu/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/alientek/example/32_rknn_retinaface_demo_video/rknn_retinaface_demo/src/image_utils.cc -o CMakeFiles/rknn_retinaface_demo.dir/src/image_utils.cc.s

# Object files for target rknn_retinaface_demo
rknn_retinaface_demo_OBJECTS = \
"CMakeFiles/rknn_retinaface_demo.dir/src/main.cc.o" \
"CMakeFiles/rknn_retinaface_demo.dir/src/retinaface.cc.o" \
"CMakeFiles/rknn_retinaface_demo.dir/src/file_utils.cc.o" \
"CMakeFiles/rknn_retinaface_demo.dir/src/image_drawing.cc.o" \
"CMakeFiles/rknn_retinaface_demo.dir/src/image_utils.cc.o"

# External object files for target rknn_retinaface_demo
rknn_retinaface_demo_EXTERNAL_OBJECTS =

rknn_retinaface_demo: CMakeFiles/rknn_retinaface_demo.dir/src/main.cc.o
rknn_retinaface_demo: CMakeFiles/rknn_retinaface_demo.dir/src/retinaface.cc.o
rknn_retinaface_demo: CMakeFiles/rknn_retinaface_demo.dir/src/file_utils.cc.o
rknn_retinaface_demo: CMakeFiles/rknn_retinaface_demo.dir/src/image_drawing.cc.o
rknn_retinaface_demo: CMakeFiles/rknn_retinaface_demo.dir/src/image_utils.cc.o
rknn_retinaface_demo: CMakeFiles/rknn_retinaface_demo.dir/build.make
rknn_retinaface_demo: ../../../runtime/RK356X/Linux/librknn_api/aarch64/librknnrt.so
rknn_retinaface_demo: ../../../3rdparty/rga/RK356X/lib/Linux/aarch64/librga.so
rknn_retinaface_demo: CMakeFiles/rknn_retinaface_demo.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/alientek/example/32_rknn_retinaface_demo_video/rknn_retinaface_demo/build/build_linux_aarch64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX executable rknn_retinaface_demo"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/rknn_retinaface_demo.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/rknn_retinaface_demo.dir/build: rknn_retinaface_demo

.PHONY : CMakeFiles/rknn_retinaface_demo.dir/build

CMakeFiles/rknn_retinaface_demo.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/rknn_retinaface_demo.dir/cmake_clean.cmake
.PHONY : CMakeFiles/rknn_retinaface_demo.dir/clean

CMakeFiles/rknn_retinaface_demo.dir/depend:
	cd /home/alientek/example/32_rknn_retinaface_demo_video/rknn_retinaface_demo/build/build_linux_aarch64 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/alientek/example/32_rknn_retinaface_demo_video/rknn_retinaface_demo /home/alientek/example/32_rknn_retinaface_demo_video/rknn_retinaface_demo /home/alientek/example/32_rknn_retinaface_demo_video/rknn_retinaface_demo/build/build_linux_aarch64 /home/alientek/example/32_rknn_retinaface_demo_video/rknn_retinaface_demo/build/build_linux_aarch64 /home/alientek/example/32_rknn_retinaface_demo_video/rknn_retinaface_demo/build/build_linux_aarch64/CMakeFiles/rknn_retinaface_demo.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/rknn_retinaface_demo.dir/depend

