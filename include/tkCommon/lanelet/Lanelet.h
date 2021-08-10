#pragma once

#include "tkCommon/common.h"

#ifdef LANELET_ENABLED
#include <lanelet2_core/LaneletMap.h>
#include <lanelet2_core/geometry/Area.h>
#include <lanelet2_core/geometry/Lanelet.h>
#include <lanelet2_core/primitives/Area.h>
#include <lanelet2_core/primitives/Lanelet.h>
#include <lanelet2_core/primitives/LineString.h>
#include <lanelet2_core/primitives/Point.h>
#include <lanelet2_core/primitives/Polygon.h>
#include <lanelet2_core/utility/Units.h>
#include <lanelet2_io/Io.h>
#include <lanelet2_io/io_handlers/Factory.h>
#include <lanelet2_io/io_handlers/Writer.h>
#include <lanelet2_projection/UTM.h>
#include <lanelet2_routing/Route.h>
#include <lanelet2_routing/RoutingCost.h>
#include <lanelet2_routing/RoutingGraph.h>
#include <lanelet2_routing/RoutingGraphContainer.h>
#include <lanelet2_traffic_rules/TrafficRulesFactory.h>
#endif

namespace tk { namespace common {
    class Lanelet {
    public:
         Lanelet() = default;
        ~Lanelet() = default;

    #ifdef LANELET_ENABLED

        bool init(const std::string &aConfPath) {
            YAML::Node conf     = YAML::LoadFile(aConfPath);
            
            std::string local_p = tk::common::YAMLgetConf<std::string>(conf, "file", "lanelet.osm");
            std::string map_p   = aConfPath.substr(0, aConfPath.find_last_of('/')) + "/" + local_p;
            mOriginLat          = tk::common::YAMLgetConf<double>(conf, "lat", 0.0f);
            mOriginLon          = tk::common::YAMLgetConf<double>(conf, "lon", 0.0f);
            
            mProjector          = new lanelet::projection::UtmProjector(lanelet::Origin({mOriginLat, mOriginLon}));
            mMap                = lanelet::load(map_p, *mProjector);
            return true;
        }

        lanelet::LaneletMapUPtr mMap;
        double mOriginLat, mOriginLon;
        lanelet::projection::UtmProjector *mProjector;
    #else
        bool init(const std::string &aConfPath) {
            tkERR("You need to compile with lanelet2");
            return false;    
        }
    #endif
    };
}}