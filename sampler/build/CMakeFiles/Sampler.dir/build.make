# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.24

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/yorek/sample/sampler

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/yorek/sample/sampler/build

# Include any dependencies generated for this target.
include CMakeFiles/Sampler.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/Sampler.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/Sampler.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Sampler.dir/flags.make

CMakeFiles/Sampler.dir/src/main.cpp.o: CMakeFiles/Sampler.dir/flags.make
CMakeFiles/Sampler.dir/src/main.cpp.o: /home/yorek/sample/sampler/src/main.cpp
CMakeFiles/Sampler.dir/src/main.cpp.o: CMakeFiles/Sampler.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yorek/sample/sampler/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Sampler.dir/src/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/Sampler.dir/src/main.cpp.o -MF CMakeFiles/Sampler.dir/src/main.cpp.o.d -o CMakeFiles/Sampler.dir/src/main.cpp.o -c /home/yorek/sample/sampler/src/main.cpp

CMakeFiles/Sampler.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Sampler.dir/src/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yorek/sample/sampler/src/main.cpp > CMakeFiles/Sampler.dir/src/main.cpp.i

CMakeFiles/Sampler.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Sampler.dir/src/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yorek/sample/sampler/src/main.cpp -o CMakeFiles/Sampler.dir/src/main.cpp.s

CMakeFiles/Sampler.dir/src/sampler.cpp.o: CMakeFiles/Sampler.dir/flags.make
CMakeFiles/Sampler.dir/src/sampler.cpp.o: /home/yorek/sample/sampler/src/sampler.cpp
CMakeFiles/Sampler.dir/src/sampler.cpp.o: CMakeFiles/Sampler.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yorek/sample/sampler/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/Sampler.dir/src/sampler.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/Sampler.dir/src/sampler.cpp.o -MF CMakeFiles/Sampler.dir/src/sampler.cpp.o.d -o CMakeFiles/Sampler.dir/src/sampler.cpp.o -c /home/yorek/sample/sampler/src/sampler.cpp

CMakeFiles/Sampler.dir/src/sampler.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Sampler.dir/src/sampler.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yorek/sample/sampler/src/sampler.cpp > CMakeFiles/Sampler.dir/src/sampler.cpp.i

CMakeFiles/Sampler.dir/src/sampler.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Sampler.dir/src/sampler.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yorek/sample/sampler/src/sampler.cpp -o CMakeFiles/Sampler.dir/src/sampler.cpp.s

# Object files for target Sampler
Sampler_OBJECTS = \
"CMakeFiles/Sampler.dir/src/main.cpp.o" \
"CMakeFiles/Sampler.dir/src/sampler.cpp.o"

# External object files for target Sampler
Sampler_EXTERNAL_OBJECTS =

Sampler: CMakeFiles/Sampler.dir/src/main.cpp.o
Sampler: CMakeFiles/Sampler.dir/src/sampler.cpp.o
Sampler: CMakeFiles/Sampler.dir/build.make
Sampler: CMakeFiles/Sampler.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/yorek/sample/sampler/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable Sampler"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Sampler.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Sampler.dir/build: Sampler
.PHONY : CMakeFiles/Sampler.dir/build

CMakeFiles/Sampler.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Sampler.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Sampler.dir/clean

CMakeFiles/Sampler.dir/depend:
	cd /home/yorek/sample/sampler/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yorek/sample/sampler /home/yorek/sample/sampler /home/yorek/sample/sampler/build /home/yorek/sample/sampler/build /home/yorek/sample/sampler/build/CMakeFiles/Sampler.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Sampler.dir/depend
