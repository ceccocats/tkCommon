from _codeGen import genData

className = "CanData_gen"
DEPS = [ "#include <linux/can.h>", "#include <linux/can/raw.h>" ]
VARS = [ {"name":"frame", "type":"struct can_frame"} ]
genData(className, VARS, DEPS)

className = "OdomData_gen"
DEPS = [ "#include \"tkCommon/math/Mat.h\"" ]
VARS = [ {"name":"x",      "type":"float"},
         {"name":"y",      "type":"float"},
         {"name":"yaw",    "type":"float"},
         {"name":"speed",  "type":"float"} ]
genData(className, VARS, DEPS)

className = "VehicleData_gen"
DEPS = [ "#include \"tkCommon/math/Vec.h\"",
         "#include \"tkCommon/data/gen/OdomData_gen.h\"" ]
VARS = [ {"name":"CAR_WHEELBASE",       "type":"double"                , "default": "0"},
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
VARS = [ {"name":"acc",         "type":"tk::math::Vec3<double>"},
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
VARS = [ {"name":"utcStamp","type":"timeStamp_t","default":"0"},
         {"name":"lat",     "type":"double",  "default":"0"}, 
         {"name":"lon",     "type":"double",  "default":"0"}, 
         {"name":"heigth",  "type":"double",  "default":"0"},
         {"name":"quality", "type":"int",     "default":"0"},
         {"name":"sats",    "type":"int",     "default":"0"},
         {"name":"age",     "type":"int",     "default":"0"},
         {"name":"cov",     "type":"tk::math::Mat3d" },
         {"name":"pressure","type":"double",  "default":"0"}, 
         {"name":"temp",    "type":"double",  "default":"0"},
        ]
genData(className, VARS, DEPS)

className = "GpsImuData_gen"
DEPS = ["#include \"tkCommon/data/ImuData.h\"\n", 
        "#include \"tkCommon/data/GpsData.h\"\n"]
VARS = [ {"name":"gps", "type":"tk::data::GpsData", "init": "gps.init();"}, 
         {"name":"imu", "type":"tk::data::ImuData", "init": "imu.init();"},
         {"name":"vel", "type":"tk::math::Vec3<double>"},
        ]
genData(className, VARS, DEPS)

className = "CloudData_gen"
DEPS = ["#include \"tkCommon/math/Mat.h\"\n"]
DEPS = ["#include \"tkCommon/math/Vec.h\"\n"]
VARS = [ {"name": "featureType_t", "type": "typedef std::string"},
         {"name": "FEATURES_NONE"        , "type": "static const featureType_t", "default": "\"f_none\""},
         {"name": "FEATURES_I"           , "type": "static const featureType_t", "default": "\"f_intensity\""},
         {"name": "FEATURES_NOISE"       , "type": "static const featureType_t", "default": "\"f_noise\""},
         {"name": "FEATURES_CHANNEL"     , "type": "static const featureType_t", "default": "\"f_channel\""},
         {"name": "FEATURES_R"           , "type": "static const featureType_t", "default": "\"f_r\""},
         {"name": "FEATURES_G"           , "type": "static const featureType_t", "default": "\"f_g\""},
         {"name": "FEATURES_B"           , "type": "static const featureType_t", "default": "\"f_b\""},
         {"name": "FEATURES_NX"          , "type": "static const featureType_t", "default": "\"f_nx\""},
         {"name": "FEATURES_NY"          , "type": "static const featureType_t", "default": "\"f_ny\""},
         {"name": "FEATURES_NZ"          , "type": "static const featureType_t", "default": "\"f_nz\""},
         {"name": "FEATURES_CLASS"       , "type": "static const featureType_t", "default": "\"f_class\""},
         {"name": "FEATURES_DIST"        , "type": "static const featureType_t", "default": "\"f_dist\""},
         {"name": "FEATURES_VELOCITY"    , "type": "static const featureType_t", "default": "\"f_velocity\""},
         {"name": "FEATURES_FALSE_DET"   , "type": "static const featureType_t", "default": "\"f_false_det\""},
         {"name": "FEATURES_RANGE_VAR"   , "type": "static const featureType_t", "default": "\"f_range_var\""},
         {"name": "FEATURES_VELOCITY_VAR", "type": "static const featureType_t", "default": "\"f_velocity_var\""},
         {"name": "FEATURES_ANGLE_VAR"   , "type": "static const featureType_t", "default": "\"f_angle_var\""},
         {"name": "FEATURES_RCS"         , "type": "static const featureType_t", "default": "\"f_rcs\""},
         {"name": "FEATURES_PROBABILITY" , "type": "static const featureType_t", "default": "\"f_probability\""},
         {"name": "FEATURES_NEAR_SCAN"   , "type": "static const featureType_t", "default": "\"f_near_scan\""},
         {"name": "FEATURES_GROUND"      , "type": "static const featureType_t", "default": "\"f_ground\""},
         {"name": "TIMESTAMP"            , "type": "static const featureType_t", "default": "\"f_timestamp\""},
         {"name":"points",   "type":"tk::math::Mat<float>"},
         {"name":"ranges",   "type":"tk::math::Mat<float>"},
         {"name":"features", "type":"tk::common::Map<tk::math::Vec<float>>"}]
genData(className, VARS, DEPS)

className = "ImageData_gen"
DEPS = ["#include \"tkCommon/math/Vec.h\"\n"]
VARS = [ {"name":"data", "type":"tk::math::Vec<uint8_t>"},
         {"name":"width", "type":"uint32_t", "default": "0"},
         {"name":"height", "type":"uint32_t", "default": "0"},
         {"name":"channels", "type":"uint32_t", "default": "0"}]
genData(className, VARS, DEPS)

className = "RadarData_gen"
DEPS = ["#include \"tkCommon/data/CloudData.h\"\n"]
VARS = [ {"name":"near", "type":"tk::data::CloudData", "init": "near.init()"},
         {"name":"far",  "type":"tk::data::CloudData", "init": "far.init()"} ]
genData(className, VARS, DEPS)

className = "CalibData_gen"
DEPS = ["#include \"tkCommon/math/Mat.h\"\n"]
VARS = [ {"name":"w",   "type":"int", "default":"0"},
         {"name":"h",   "type":"int", "default":"0"},
         {"name":"k",   "type":"tk::math::Mat<float>", "init":"k.resize(3,3)"},
         {"name":"d",   "type":"tk::math::Mat<float>", "init":"d.resize(1,5)"},
         {"name":"r",   "type":"tk::math::Mat<float>", "init":"r.resize(3,3)"} ]
genData(className, VARS, DEPS)

className = "ActuationData_gen"
DEPS = ["#include \"tkCommon/math/Mat.h\"\n"]
VARS = [ {"name":"steerAngle", "type":"double", "default":"0"},
         {"name":"accel",      "type":"double", "default":"0"},
         {"name":"speed",      "type":"double", "default":"0"} ]
genData(className, VARS, DEPS)

className = "DepthData_gen"
DEPS = ["#include \"tkCommon/math/Vec.h\"\n"]
VARS = [ {"name":"data", "type":"tk::math::Vec<uint16_t>"},
         {"name":"width", "type":"uint32_t", "default": "0"},
         {"name":"height", "type":"uint32_t", "default": "0"} ]
genData(className, VARS, DEPS)