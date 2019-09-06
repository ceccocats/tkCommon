include_directories(
    ${CMAKE_INSTALL_PREFIX}/include
)
link_directories(${CMAKE_INSTALL_PREFIX}/lib)

# suppress preprocessor warning 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wp,-w")

