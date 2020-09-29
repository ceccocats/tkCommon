from _codeGen import genData

className = "CanData_gen"
DEPS = [ "#include <linux/can.h>", "#include <linux/can/raw.h>" ]
VARS = [ {"name":"frame", "type":"struct can_frame"},]
genData(className, VARS, DEPS)

className = "VehicleData_gen"
VARS = [ {"name":"speed",   "type":"double"},
         {"name":"yawRate", "type":"double"}  ]
genData(className, VARS)

className = "ImuData_gen"
DEPS = [ "#include \"tkCommon/math/Vec.h\"" ]
VARS = [ {"name":"acc",       "type":"tk::math::Vec3<double>"},
         {"name":"angleRate", "type":"tk::math::Vec3<double>"}, 
       ]
genData(className, VARS, DEPS)

className = "GpsData_gen"
VARS = [ {"name":"lat", "type":"double", "default":"0"}, 
         {"name":"lon", "type":"double", "default":"0"}, 
         {"name":"heigth", "type":"double", "default":"0"},
         {"name":"quality", "type":"int", "default":"0"} ]
genData(className, VARS)

className = "GpsImuData_gen"
DEPS = ["#include \"tkCommon/data/gen/ImuData_gen.h\"\n", 
        "#include \"tkCommon/data/gen/GpsData_gen.h\"\n"]
VARS = [ {"name":"gps", "type":"tk::data::GpsData_gen"}, 
         {"name":"imu", "type":"tk::data::ImuData_gen"} ]
genData(className, VARS, DEPS)

className = "CloudData_gen"
DEPS = ["#include \"tkCommon/math/Mat.h\"\n"]
VARS = [ {"name": "featureType_t", "type": "typedef std::string"},
         {"name": "FEATURES_NONE", "type": "const featureType_t", "default": "\"f_none\""},
         {"name": "FEATURES_I", "type": "const featureType_t", "default": "\"f_intensity\""},
         {"name":"points",   "type":"tk::math::Mat<float>"},
         {"name":"ranges",   "type":"tk::math::Mat<float>"},
         {"name":"features", "type":"tk::math::Mat<float>"},
         {"name":"features_map", "type":"std::map<featureType_t, int>"} ]
genData(className, VARS, DEPS)

className = "ImageData_gen"
DEPS = ["#include \"tkCommon/math/Mat.h\"\n"]
VARS = [ {"name":"data", "type":"tk::math::Mat<uint8_t>"},
         {"name":"width", "type":"uint32_t", "default": "0"},
         {"name":"height", "type":"uint32_t", "default": "0"},
         {"name":"channels", "type":"uint32_t", "default": "0"}]
genData(className, VARS, DEPS)

className = "RadarData_gen"
DEPS = ["#include \"tkCommon/data/gen/CloudData_gen.h\"\n"]
VARS = [ {"name":"near", "type":"tk::data::CloudData_gen"},
         {"name":"far", "type":"tk::data::CloudData_gen"} ]
genData(className, VARS, DEPS)

className = "CalibData_gen"
DEPS = ["#include \"tkCommon/math/Mat.h\"\n"]
VARS = [ {"name":"K",   "type":"tk::math::Mat<float>", "init":"K.resize(3,3)"},
         {"name":"D",   "type":"tk::math::Mat<float>", "init":"D.resize(1,5)"},
         {"name":"R",   "type":"tk::math::Mat<float>", "init":"R.resize(3,3)"} ]
genData(className, VARS, DEPS)

className = "ActuationData_gen"
DEPS = ["#include \"tkCommon/math/Mat.h\"\n"]
VARS = [ {"name":"steerAngle", "type":"double", "init":"0"},
         {"name":"accel",      "type":"double", "init":"0"},
         {"name":"speed",      "type":"double", "init":"0"} ]
genData(className, VARS, DEPS)
