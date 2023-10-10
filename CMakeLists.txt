cmake_minimum_required(VERSION 3.12)

if(${CMAKE_VERSION} VERSION_LESS 3.12)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()
FIND_PACKAGE(deal.II 9.4.0 REQUIRED)
FIND_PACKAGE(HDF5 REQUIRED)
FIND_PACKAGE(MPI REQUIRED)
#find_package(SLEPC)
find_package(PkgConfig)
if (PKG_CONFIG_FOUND)
    pkg_check_modules(PETSC PETSc)
    #pkg_check_modules(METIS METIS)
    #pkg_check_modules(SLEPC SLEPC)
endif()

#ADD_LIBRARY(METIS)
#FIND_PACKAGE(METIS REQUIRED)

set(scope "parametric_shaft_dynamic_v8")
#set(scope "step_18_cylv5")
#set(scope "step_18_cylv5")
#set(scope "step_36_parallel_3")
#set(scope step_3)
project(Zaxos_project VERSION 0.1
                DESCRIPTION "o xontros kanei fea"
                LANGUAGES CXX)
INCLUDE_DIRECTORIES(src build)        
ADD_EXECUTABLE(${scope} src/${scope}.cpp)
DEAL_II_SETUP_TARGET(${scope})

#FILE(MAKE_DIRECTORY run/${scope})
#FILE(COPY ${CMAKE_SOURCE_DIR}/run DESTINATION ${CMAKE_BINARY_DIR})

SET_PROPERTY(TARGET ${scope}
  PROPERTY RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/run/${scope}
  )
  


ADD_CUSTOM_TARGET(run
  COMMAND ${scope}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/run/${scope}
  )
# set (CMAKE_CXX_FLAGS_RELEASE "-O3 -cpp -ffree-line-length-512  \


