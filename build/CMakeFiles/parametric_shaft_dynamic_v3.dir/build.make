# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.26

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
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/zaxosoik/dealii_oikonomou

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/zaxosoik/dealii_oikonomou/build

# Include any dependencies generated for this target.
include CMakeFiles/parametric_shaft_dynamic_v3.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/parametric_shaft_dynamic_v3.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/parametric_shaft_dynamic_v3.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/parametric_shaft_dynamic_v3.dir/flags.make

CMakeFiles/parametric_shaft_dynamic_v3.dir/src/parametric_shaft_dynamic_v3.cpp.o: CMakeFiles/parametric_shaft_dynamic_v3.dir/flags.make
CMakeFiles/parametric_shaft_dynamic_v3.dir/src/parametric_shaft_dynamic_v3.cpp.o: /home/zaxosoik/dealii_oikonomou/src/parametric_shaft_dynamic_v3.cpp
CMakeFiles/parametric_shaft_dynamic_v3.dir/src/parametric_shaft_dynamic_v3.cpp.o: CMakeFiles/parametric_shaft_dynamic_v3.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zaxosoik/dealii_oikonomou/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/parametric_shaft_dynamic_v3.dir/src/parametric_shaft_dynamic_v3.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/parametric_shaft_dynamic_v3.dir/src/parametric_shaft_dynamic_v3.cpp.o -MF CMakeFiles/parametric_shaft_dynamic_v3.dir/src/parametric_shaft_dynamic_v3.cpp.o.d -o CMakeFiles/parametric_shaft_dynamic_v3.dir/src/parametric_shaft_dynamic_v3.cpp.o -c /home/zaxosoik/dealii_oikonomou/src/parametric_shaft_dynamic_v3.cpp

CMakeFiles/parametric_shaft_dynamic_v3.dir/src/parametric_shaft_dynamic_v3.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/parametric_shaft_dynamic_v3.dir/src/parametric_shaft_dynamic_v3.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zaxosoik/dealii_oikonomou/src/parametric_shaft_dynamic_v3.cpp > CMakeFiles/parametric_shaft_dynamic_v3.dir/src/parametric_shaft_dynamic_v3.cpp.i

CMakeFiles/parametric_shaft_dynamic_v3.dir/src/parametric_shaft_dynamic_v3.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/parametric_shaft_dynamic_v3.dir/src/parametric_shaft_dynamic_v3.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zaxosoik/dealii_oikonomou/src/parametric_shaft_dynamic_v3.cpp -o CMakeFiles/parametric_shaft_dynamic_v3.dir/src/parametric_shaft_dynamic_v3.cpp.s

# Object files for target parametric_shaft_dynamic_v3
parametric_shaft_dynamic_v3_OBJECTS = \
"CMakeFiles/parametric_shaft_dynamic_v3.dir/src/parametric_shaft_dynamic_v3.cpp.o"

# External object files for target parametric_shaft_dynamic_v3
parametric_shaft_dynamic_v3_EXTERNAL_OBJECTS =

run/parametric_shaft_dynamic_v3/parametric_shaft_dynamic_v3: CMakeFiles/parametric_shaft_dynamic_v3.dir/src/parametric_shaft_dynamic_v3.cpp.o
run/parametric_shaft_dynamic_v3/parametric_shaft_dynamic_v3: CMakeFiles/parametric_shaft_dynamic_v3.dir/build.make
run/parametric_shaft_dynamic_v3/parametric_shaft_dynamic_v3: /usr/local/lib/libdeal_II.so.9.5.0-pre
run/parametric_shaft_dynamic_v3/parametric_shaft_dynamic_v3: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_cxx.so
run/parametric_shaft_dynamic_v3/parametric_shaft_dynamic_v3: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
run/parametric_shaft_dynamic_v3/parametric_shaft_dynamic_v3: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
run/parametric_shaft_dynamic_v3/parametric_shaft_dynamic_v3: /usr/lib/x86_64-linux-gnu/libboost_system.so
run/parametric_shaft_dynamic_v3/parametric_shaft_dynamic_v3: /usr/lib/x86_64-linux-gnu/libboost_thread.so
run/parametric_shaft_dynamic_v3/parametric_shaft_dynamic_v3: /usr/lib/x86_64-linux-gnu/libboost_regex.so
run/parametric_shaft_dynamic_v3/parametric_shaft_dynamic_v3: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
run/parametric_shaft_dynamic_v3/parametric_shaft_dynamic_v3: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
run/parametric_shaft_dynamic_v3/parametric_shaft_dynamic_v3: /usr/local/lib/libkokkoscore.so
run/parametric_shaft_dynamic_v3/parametric_shaft_dynamic_v3: /usr/local/lib/libkokkoscontainers.so
run/parametric_shaft_dynamic_v3/parametric_shaft_dynamic_v3: /usr/lib/x86_64-linux-gnu/libarpack.so
run/parametric_shaft_dynamic_v3/parametric_shaft_dynamic_v3: /usr/lib/x86_64-linux-gnu/libmpfr.so
run/parametric_shaft_dynamic_v3/parametric_shaft_dynamic_v3: /usr/lib/libhdf5.so
run/parametric_shaft_dynamic_v3/parametric_shaft_dynamic_v3: /usr/lib/libp4est.so
run/parametric_shaft_dynamic_v3/parametric_shaft_dynamic_v3: /usr/lib/libsc.so
run/parametric_shaft_dynamic_v3/parametric_shaft_dynamic_v3: /usr/lib/x86_64-linux-gnu/libscalapack-openmpi.so
run/parametric_shaft_dynamic_v3/parametric_shaft_dynamic_v3: /usr/lib/x86_64-linux-gnu/libopenblas.so
run/parametric_shaft_dynamic_v3/parametric_shaft_dynamic_v3: /usr/lib/x86_64-linux-gnu/libhwloc.so
run/parametric_shaft_dynamic_v3/parametric_shaft_dynamic_v3: /usr/lib/x86_64-linux-gnu/libevent_core.so
run/parametric_shaft_dynamic_v3/parametric_shaft_dynamic_v3: /usr/lib/x86_64-linux-gnu/libevent_pthreads.so
run/parametric_shaft_dynamic_v3/parametric_shaft_dynamic_v3: /usr/lib/x86_64-linux-gnu/libm.so
run/parametric_shaft_dynamic_v3/parametric_shaft_dynamic_v3: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so
run/parametric_shaft_dynamic_v3/parametric_shaft_dynamic_v3: /home/zaxosoik/petsc/x86_64/lib/libslepc.so
run/parametric_shaft_dynamic_v3/parametric_shaft_dynamic_v3: /home/zaxosoik/petsc/x86_64/lib/libpetsc.so
run/parametric_shaft_dynamic_v3/parametric_shaft_dynamic_v3: /home/zaxosoik/petsc/x86_64/lib/libHYPRE.so
run/parametric_shaft_dynamic_v3/parametric_shaft_dynamic_v3: /home/zaxosoik/petsc/x86_64/lib/libspqr.so
run/parametric_shaft_dynamic_v3/parametric_shaft_dynamic_v3: /home/zaxosoik/petsc/x86_64/lib/libumfpack.so
run/parametric_shaft_dynamic_v3/parametric_shaft_dynamic_v3: /home/zaxosoik/petsc/x86_64/lib/libklu.so
run/parametric_shaft_dynamic_v3/parametric_shaft_dynamic_v3: /home/zaxosoik/petsc/x86_64/lib/libcholmod.so
run/parametric_shaft_dynamic_v3/parametric_shaft_dynamic_v3: /home/zaxosoik/petsc/x86_64/lib/libbtf.so
run/parametric_shaft_dynamic_v3/parametric_shaft_dynamic_v3: /home/zaxosoik/petsc/x86_64/lib/libccolamd.so
run/parametric_shaft_dynamic_v3/parametric_shaft_dynamic_v3: /home/zaxosoik/petsc/x86_64/lib/libcolamd.so
run/parametric_shaft_dynamic_v3/parametric_shaft_dynamic_v3: /home/zaxosoik/petsc/x86_64/lib/libcamd.so
run/parametric_shaft_dynamic_v3/parametric_shaft_dynamic_v3: /home/zaxosoik/petsc/x86_64/lib/libamd.so
run/parametric_shaft_dynamic_v3/parametric_shaft_dynamic_v3: /home/zaxosoik/petsc/x86_64/lib/libsuitesparseconfig.so
run/parametric_shaft_dynamic_v3/parametric_shaft_dynamic_v3: /home/zaxosoik/petsc/x86_64/lib/libdmumps.a
run/parametric_shaft_dynamic_v3/parametric_shaft_dynamic_v3: /home/zaxosoik/petsc/x86_64/lib/libmumps_common.a
run/parametric_shaft_dynamic_v3/parametric_shaft_dynamic_v3: /home/zaxosoik/petsc/x86_64/lib/libpord.a
run/parametric_shaft_dynamic_v3/parametric_shaft_dynamic_v3: /home/zaxosoik/petsc/x86_64/lib/libscalapack.so
run/parametric_shaft_dynamic_v3/parametric_shaft_dynamic_v3: /home/zaxosoik/petsc/x86_64/lib/libparmetis.so
run/parametric_shaft_dynamic_v3/parametric_shaft_dynamic_v3: /usr/lib/x86_64-linux-gnu/libmpi_usempif08.so
run/parametric_shaft_dynamic_v3/parametric_shaft_dynamic_v3: /usr/lib/x86_64-linux-gnu/libmpi_usempi_ignore_tkr.so
run/parametric_shaft_dynamic_v3/parametric_shaft_dynamic_v3: /usr/lib/x86_64-linux-gnu/libmpi_mpifh.so
run/parametric_shaft_dynamic_v3/parametric_shaft_dynamic_v3: /usr/lib/x86_64-linux-gnu/libopen-rte.so
run/parametric_shaft_dynamic_v3/parametric_shaft_dynamic_v3: /usr/lib/x86_64-linux-gnu/libopen-pal.so
run/parametric_shaft_dynamic_v3/parametric_shaft_dynamic_v3: /usr/lib/x86_64-linux-gnu/libz.so
run/parametric_shaft_dynamic_v3/parametric_shaft_dynamic_v3: /home/zaxosoik/petsc/x86_64/lib/libmetis.so
run/parametric_shaft_dynamic_v3/parametric_shaft_dynamic_v3: /usr/local/lib/libsymengine.a
run/parametric_shaft_dynamic_v3/parametric_shaft_dynamic_v3: /usr/lib/x86_64-linux-gnu/libgmp.so
run/parametric_shaft_dynamic_v3/parametric_shaft_dynamic_v3: CMakeFiles/parametric_shaft_dynamic_v3.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zaxosoik/dealii_oikonomou/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable run/parametric_shaft_dynamic_v3/parametric_shaft_dynamic_v3"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/parametric_shaft_dynamic_v3.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/parametric_shaft_dynamic_v3.dir/build: run/parametric_shaft_dynamic_v3/parametric_shaft_dynamic_v3
.PHONY : CMakeFiles/parametric_shaft_dynamic_v3.dir/build

CMakeFiles/parametric_shaft_dynamic_v3.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/parametric_shaft_dynamic_v3.dir/cmake_clean.cmake
.PHONY : CMakeFiles/parametric_shaft_dynamic_v3.dir/clean

CMakeFiles/parametric_shaft_dynamic_v3.dir/depend:
	cd /home/zaxosoik/dealii_oikonomou/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zaxosoik/dealii_oikonomou /home/zaxosoik/dealii_oikonomou /home/zaxosoik/dealii_oikonomou/build /home/zaxosoik/dealii_oikonomou/build /home/zaxosoik/dealii_oikonomou/build/CMakeFiles/parametric_shaft_dynamic_v3.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/parametric_shaft_dynamic_v3.dir/depend

