cmake_minimum_required(VERSION 3.16)
project(neutron_yield LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Geant4 REQUIRED COMPONENTS ui_all vis_all analysis)
include(${Geant4_USE_FILE})

set(SRC_FILES
    src/main.cc
    src/ActionInitialization.cc
    src/DetectorConstruction.cc
    src/PhysicsList.cc
    src/PrimaryGeneratorAction.cc
    src/RunAction.cc
    src/SteppingAction.cc
)

add_executable(neutron_yield ${SRC_FILES})
target_include_directories(neutron_yield PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(neutron_yield PRIVATE ${Geant4_LIBRARIES})

install(TARGETS neutron_yield DESTINATION bin)
