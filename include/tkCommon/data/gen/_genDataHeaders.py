import sys
sys.path.insert(1, '../../../../scripts/gen/')
from codeGen import genData

className = "CanData_gen"
DEPS = [ "#include <linux/can.h>", "#include <linux/can/raw.h>" ]
VARS = [ {"name":"type",        "type":"static const DataType", "default": "DataType::CAN"},       
         {"name":"frame",       "type":"struct can_frame"} ]
genData(className, VARS, DEPS)

className = "OdomData_gen"
DEPS = [ "#include \"tkCommon/math/Vec.h\"" ]
VARS = [ {"name":"type",                "type":"static const DataType", "default": "DataType::ODOM"},  
         {"name":"pose",                "type":"tk::math::Vec3<double>"},
         {"name":"angle",               "type":"tk::math::Vec4<double>"},
         {"name":"linear_velocity",     "type":"tk::math::Vec3<double>"},
         {"name":"angular_velocity",    "type":"tk::math::Vec3<double>"}
        ]
genData(className, VARS, DEPS)

className = "VehicleData_gen"
DEPS = [ "#include \"tkCommon/math/Vec.h\"",
         "#include \"tkCommon/data/gen/OdomData_gen.h\"" ]
VARS = [ {"name":"type",                "type":"static const DataType", "default": "DataType::VEHICLE"},
         {"name":"CAR_WHEELBASE",       "type":"double"                , "default": "0"},
         {"name":"CAR_DIM",             "type":"tk::math::Vec3<double>"},
         {"name":"CAR_BACK2AXLE",       "type":"double"                , "default": "0"},
         {"name":"CAR_MASS",            "type":"double"                , "default": "0"}, 
         {"name":"CAR_FRONTAXLE_W",     "type":"double"                , "default": "0"},
         {"name":"CAR_BACKAXLE_W",      "type":"double"                , "default": "0"},
         {"name":"CAR_WHEEL_R",         "type":"double"                , "default": "0"},
         {"name":"speed",               "type":"double"                , "default": "0"},
         {"name":"speedKMH",            "type":"double"                , "default": "0"},
         {"name":"yawRate",             "type":"double"                , "default": "0"},
         {"name":"accX",                "type":"double"                , "default": "0"},
         {"name":"accY",                "type":"double"                , "default": "0"},
         {"name":"steerAngle",          "type":"double"                , "default": "0"},
         {"name":"steerAngleRate",      "type":"double"                , "default": "0"},
         {"name":"wheelAngle",          "type":"double"                , "default": "0"},
         {"name":"brakePedalStatus",    "type":"int"                   , "default": "0"},
         {"name":"brakeMasterPressure", "type":"double"                , "default": "0"},
         {"name":"gasPedal",            "type":"double"                , "default": "0"},
         {"name":"engineTorque",        "type":"double"                , "default": "0"},
         {"name":"engineFrictionTorque","type":"double"                , "default": "0"},
         {"name":"actualGear",          "type":"int"                   , "default": "0"},
         {"name":"RPM",                 "type":"int"                   , "default": "0"},
         {"name":"wheelFLspeed",        "type":"double"                , "default": "0"},
         {"name":"wheelFRspeed",        "type":"double"                , "default": "0"},
         {"name":"wheelRLspeed",        "type":"double"                , "default": "0"},
         {"name":"wheelRRspeed",        "type":"double"                , "default": "0"},
         {"name":"wheelFLdir",          "type":"int"                   , "default": "0"},
         {"name":"wheelFRdir",          "type":"int"                   , "default": "0"},
         {"name":"wheelRLdir",          "type":"int"                   , "default": "0"},
         {"name":"wheelRRdir",          "type":"int"                   , "default": "0"},
         {"name":"sideSlip",            "type":"double"                , "default": "0"},
         {"name":"tractionGrip",        "type":"int"                   , "default": "0"},
         {"name":"odom",                "type":"tk::data::OdomData_gen", "init": "odom.init();"}
        ]
genData(className, VARS, DEPS)

className = "ImuData_gen"
DEPS = [ "#include \"tkCommon/math/Vec.h\"",
         "#include \"tkCommon/math/Mat.h\"" ]
VARS = [ {"name":"type",        "type":"static const DataType", "default": "DataType::IMU"},
         {"name":"acc",         "type":"tk::math::Vec3<double>"},
         {"name":"angleVel",    "type":"tk::math::Vec3<double>"}, 
         {"name":"angle",       "type":"tk::math::Vec3<double>"}, 
         {"name":"mag",         "type":"tk::math::Vec3<double>"},
         {"name":"covAcc",      "type":"tk::math::Mat3d" },
         {"name":"covAngleVel", "type":"tk::math::Mat3d" },
         {"name":"covAngle",    "type":"tk::math::Mat3d" },
         {"name":"covMag",      "type":"tk::math::Mat3d" },
         {"name":"sideSlip",    "type":"double",  "default":"0"}, 
        ]
genData(className, VARS, DEPS)

className = "GpsData_gen"
DEPS = ["#include \"tkCommon/math/Mat.h\"" ]
VARS = [ {"name":"type",        "type":"static const DataType", "default": "DataType::GPS"},
         {"name":"utcStamp",    "type":"timeStamp_t","default":"0"},
         {"name":"lat",         "type":"double",  "default":"0"}, 
         {"name":"lon",         "type":"double",  "default":"0"}, 
         {"name":"heigth",      "type":"double",  "default":"0"},
         {"name":"speed",       "type":"tk::math::Vec3<double>"}, 
         {"name":"angle",       "type":"tk::math::Vec3<double>"}, 
         {"name":"quality",     "type":"int",     "default":"0"},
         {"name":"sats",        "type":"int",     "default":"0"},
         {"name":"age",         "type":"int",     "default":"0"},
         {"name":"cov",         "type":"tk::math::Mat3d" },
         {"name":"covSpeed",    "type":"tk::math::Mat3d" },
         {"name":"covAngle",    "type":"tk::math::Mat3d" },
         {"name":"pressure",    "type":"double",  "default":"0"}, 
         {"name":"temp",        "type":"double",  "default":"0"},
        ]
genData(className, VARS, DEPS)

className = "CloudData_gen"
DEPS = ["#include \"tkCommon/math/Mat.h\"\n"]
DEPS = ["#include \"tkCommon/math/Vec.h\"\n"]
VARS = [ {"name":"featureType_t",               "type": "typedef std::string"},
         {"name":"type",                        "type":"static const DataType", "default": "DataType::CLOUD"},
         {"name":"FEATURES_NONE",               "type":"static const CloudData_gen::featureType_t", "default": "\"f_none\""},
         {"name":"FEATURES_I",                  "type":"static const CloudData_gen::featureType_t", "default": "\"f_intensity\""},
         {"name":"FEATURES_NOISE",              "type":"static const CloudData_gen::featureType_t", "default": "\"f_noise\""},
         {"name":"FEATURES_SIGNAL_NOISE",       "type":"static const CloudData_gen::featureType_t", "default": "\"f_signal_noise\""},
         {"name":"FEATURES_CHANNEL",            "type":"static const CloudData_gen::featureType_t", "default": "\"f_channel\""},
         {"name":"FEATURES_CHANNEL_H",          "type":"static const CloudData_gen::featureType_t", "default": "\"f_channel_h\""},
         {"name":"FEATURES_R",                  "type":"static const CloudData_gen::featureType_t", "default": "\"f_r\""},
         {"name":"FEATURES_G",                  "type":"static const CloudData_gen::featureType_t", "default": "\"f_g\""},
         {"name":"FEATURES_B",                  "type":"static const CloudData_gen::featureType_t", "default": "\"f_b\""},
         {"name":"FEATURES_NX",                 "type":"static const CloudData_gen::featureType_t", "default": "\"f_nx\""},
         {"name":"FEATURES_NY",                 "type":"static const CloudData_gen::featureType_t", "default": "\"f_ny\""},
         {"name":"FEATURES_NZ",                 "type":"static const CloudData_gen::featureType_t", "default": "\"f_nz\""},
         {"name":"FEATURES_CLASS",              "type":"static const CloudData_gen::featureType_t", "default": "\"f_class\""},
         {"name":"FEATURES_DIST",               "type":"static const CloudData_gen::featureType_t", "default": "\"f_dist\""},
         {"name":"FEATURES_VELOCITY",           "type":"static const CloudData_gen::featureType_t", "default": "\"f_velocity\""},
         {"name":"FEATURES_FALSE_DET",          "type":"static const CloudData_gen::featureType_t", "default": "\"f_false_det\""},
         {"name":"FEATURES_RANGE_VAR",          "type":"static const CloudData_gen::featureType_t", "default": "\"f_range_var\""},
         {"name":"FEATURES_VELOCITY_VAR",       "type":"static const CloudData_gen::featureType_t", "default": "\"f_velocity_var\""},
         {"name":"FEATURES_ANGLE_VAR",          "type":"static const CloudData_gen::featureType_t", "default": "\"f_angle_var\""},
         {"name":"FEATURES_RCS",                "type":"static const CloudData_gen::featureType_t", "default": "\"f_rcs\""},
         {"name":"FEATURES_PROBABILITY",        "type":"static const CloudData_gen::featureType_t", "default": "\"f_probability\""},
         {"name":"FEATURES_NEAR_SCAN",          "type":"static const CloudData_gen::featureType_t", "default": "\"f_near_scan\""},
         {"name":"FEATURES_GROUND",             "type":"static const CloudData_gen::featureType_t", "default": "\"f_ground\""},
         {"name":"FEATURES_TIME",               "type":"static const CloudData_gen::featureType_t", "default": "\"f_time\""},
         {"name":"FEATURES_REFLECTOR",          "type":"static const CloudData_gen::featureType_t", "default": "\"f_reflector\""},
         {"name":"points",                      "type":"tk::math::Mat<float>"},
         {"name":"ranges",                      "type":"tk::math::Mat<float>"},
         {"name":"features",                    "type":"tk::common::Map<tk::math::Vec<float>>"}]
genData(className, VARS, DEPS)

className = "ImageData_gen<T>"
DEPS = ["#include \"tkCommon/math/Vec.h\"\n"]
VARS = [ {"name":"type",        "type":"static const DataType", "default": "DataType::IMAGE"},
         {"name":"T_type",      "type":"T_to_class_type<T>" },
         {"name":"data",        "type":"tk::math::Vec<T>"},
         {"name":"width",       "type":"uint32_t", "default": "0"},
         {"name":"height",      "type":"uint32_t", "default": "0"},
         {"name":"channels",    "type":"uint32_t", "default": "0"}]
genData(className, VARS, DEPS)

className = "RadarData_gen"
DEPS = ["#include \"tkCommon/data/CloudData.h\"\n"]
VARS = [ {"name":"type",        "type":"static const DataType", "default": "DataType::RADAR"},
         {"name":"near",        "type":"tk::data::CloudData", "init": "near.init()"},
         {"name":"far",         "type":"tk::data::CloudData", "init": "far.init()"} ]
genData(className, VARS, DEPS)

className = "CalibData_gen"
DEPS = ["#include \"tkCommon/math/Mat.h\"\n"]
VARS = [ {"name":"type",        "type":"static const DataType", "default": "DataType::NOT_SPEC"},
         {"name":"w",           "type":"int", "default":"0"},
         {"name":"h",           "type":"int", "default":"0"},
         {"name":"k",           "type":"tk::math::Mat<float>", "init":"k.resize(3,3)"},
         {"name":"d",           "type":"tk::math::Mat<float>", "init":"d.resize(1,5)"},
         {"name":"r",           "type":"tk::math::Mat<float>", "init":"r.resize(3,3)"} ]
genData(className, VARS, DEPS)

className = "ActuationData_gen"
DEPS = ["#include \"tkCommon/math/Mat.h\"\n"]
VARS = [ {"name":"type",        "type":"static const DataType", "default": "DataType::ACTUATION"},
         {"name":"steerAngle",  "type":"double", "default":"0"},
         {"name":"accel",       "type":"double", "default":"0"},
         {"name":"speed",       "type":"double", "default":"0"} ]
genData(className, VARS, DEPS)

className = "DepthData_gen"
DEPS = ["#include \"tkCommon/math/Vec.h\"\n"]
VARS = [ {"name":"type",        "type":"static const DataType", "default": "DataType::DEPTH"},
         {"name":"data",        "type":"tk::math::Vec<uint16_t>"},
         {"name":"width",       "type":"uint32_t", "default": "0"},
         {"name":"height",      "type":"uint32_t", "default": "0"} ]
genData(className, VARS, DEPS)

className = "StereoData_gen"
DEPS = ["#include \"tkCommon/data/ImageData.h\"\n"]
VARS = [ {"name":"type",        "type":"static const DataType", "default": "DataType::STEREO"},
         {"name":"data",        "type":"tk::data::ImageDataU8", "init": "data.init()"}, 
         {"name":"width",       "type":"int", "default":"0"},
         {"name":"height",      "type":"int", "default":"0"},
         {"name":"channels",    "type":"int", "default":"0"},
         {"name":"c_width",     "type":"int", "default":"0"},
         {"name":"c_height",    "type":"int", "default":"0"},
         {"name":"c_channels",  "type":"int", "default":"0"},
         {"name":"d_width",     "type":"int", "default":"0"},
         {"name":"d_height",    "type":"int", "default":"0"},
         {"name":"left",        "type":"tk::data::ImageData", "init": "left.init()"},
         {"name":"right",       "type":"tk::data::ImageData", "init": "right.init()"},
         {"name":"color",       "type":"tk::data::ImageData", "init": "color.init()"},
         {"name":"depth",       "type":"tk::data::ImageDataX<uint16_t>", "init": "depth.init()"}
        ]
genData(className, VARS, DEPS)