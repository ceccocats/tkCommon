function(export_config prj_name)
    include(CMakePackageConfigHelpers)

    # generate the config file that is includes the exports
    configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
        "${CMAKE_CURRENT_BINARY_DIR}/${prj_name}Config.cmake"
        INSTALL_DESTINATION "share/${prj_name}/cmake/"
        NO_SET_AND_CHECK_MACRO
        NO_CHECK_REQUIRED_COMPONENTS_MACRO
    )

    # generate the version file for the config file
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/${prj_name}ConfigVersion.cmake"
        VERSION "1.0"
        COMPATIBILITY AnyNewerVersion
    )

    # export targets for build tree
    export(EXPORT ${prj_name}Targets NAMESPACE ${prj_name}::
        FILE "${CMAKE_CURRENT_BINARY_DIR}/${prj_name}Targets.cmake"
    )
endfunction(export_config)
