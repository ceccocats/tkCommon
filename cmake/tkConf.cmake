include_directories(
    ${CMAKE_INSTALL_PREFIX}/include
)
link_directories(${CMAKE_INSTALL_PREFIX}/lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTKPROJ_PATH='\"${CMAKE_CURRENT_SOURCE_DIR}/data/\"'")