# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.13

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
CMAKE_COMMAND = /snap/clion/61/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /snap/clion/61/bin/cmake/linux/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/lee/CLionProjects/untitled

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/lee/CLionProjects/untitled/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/grid_gms.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/grid_gms.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/grid_gms.dir/flags.make

CMakeFiles/grid_gms.dir/grid-gms.cpp.o: CMakeFiles/grid_gms.dir/flags.make
CMakeFiles/grid_gms.dir/grid-gms.cpp.o: ../grid-gms.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lee/CLionProjects/untitled/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/grid_gms.dir/grid-gms.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/grid_gms.dir/grid-gms.cpp.o -c /home/lee/CLionProjects/untitled/grid-gms.cpp

CMakeFiles/grid_gms.dir/grid-gms.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/grid_gms.dir/grid-gms.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lee/CLionProjects/untitled/grid-gms.cpp > CMakeFiles/grid_gms.dir/grid-gms.cpp.i

CMakeFiles/grid_gms.dir/grid-gms.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/grid_gms.dir/grid-gms.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lee/CLionProjects/untitled/grid-gms.cpp -o CMakeFiles/grid_gms.dir/grid-gms.cpp.s

# Object files for target grid_gms
grid_gms_OBJECTS = \
"CMakeFiles/grid_gms.dir/grid-gms.cpp.o"

# External object files for target grid_gms
grid_gms_EXTERNAL_OBJECTS =

grid_gms: CMakeFiles/grid_gms.dir/grid-gms.cpp.o
grid_gms: CMakeFiles/grid_gms.dir/build.make
grid_gms: /usr/local/lib/libopencv_videostab.so.2.4.13
grid_gms: /usr/local/lib/libopencv_ts.a
grid_gms: /usr/local/lib/libopencv_superres.so.2.4.13
grid_gms: /usr/local/lib/libopencv_stitching.so.2.4.13
grid_gms: /usr/local/lib/libopencv_contrib.so.2.4.13
grid_gms: /usr/local/lib/libopencv_nonfree.so.2.4.13
grid_gms: /usr/local/lib/libopencv_ocl.so.2.4.13
grid_gms: /usr/local/lib/libopencv_gpu.so.2.4.13
grid_gms: /usr/local/lib/libopencv_photo.so.2.4.13
grid_gms: /usr/local/lib/libopencv_objdetect.so.2.4.13
grid_gms: /usr/local/lib/libopencv_legacy.so.2.4.13
grid_gms: /usr/local/lib/libopencv_video.so.2.4.13
grid_gms: /usr/local/lib/libopencv_ml.so.2.4.13
grid_gms: /usr/local/lib/libopencv_calib3d.so.2.4.13
grid_gms: /usr/local/lib/libopencv_features2d.so.2.4.13
grid_gms: /usr/local/lib/libopencv_highgui.so.2.4.13
grid_gms: /usr/local/lib/libopencv_imgproc.so.2.4.13
grid_gms: /usr/local/lib/libopencv_flann.so.2.4.13
grid_gms: /usr/local/lib/libopencv_core.so.2.4.13
grid_gms: CMakeFiles/grid_gms.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/lee/CLionProjects/untitled/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable grid_gms"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/grid_gms.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/grid_gms.dir/build: grid_gms

.PHONY : CMakeFiles/grid_gms.dir/build

CMakeFiles/grid_gms.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/grid_gms.dir/cmake_clean.cmake
.PHONY : CMakeFiles/grid_gms.dir/clean

CMakeFiles/grid_gms.dir/depend:
	cd /home/lee/CLionProjects/untitled/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lee/CLionProjects/untitled /home/lee/CLionProjects/untitled /home/lee/CLionProjects/untitled/cmake-build-debug /home/lee/CLionProjects/untitled/cmake-build-debug /home/lee/CLionProjects/untitled/cmake-build-debug/CMakeFiles/grid_gms.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/grid_gms.dir/depend

