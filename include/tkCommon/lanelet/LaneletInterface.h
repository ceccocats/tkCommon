#pragma once

#include "tkCommon/common.h"

#ifdef LANELET_ENABLED
#include <lanelet2_core/LaneletMap.h>
#include <lanelet2_core/geometry/Area.h>
#include <lanelet2_core/geometry/Lanelet.h>
#include <lanelet2_core/geometry/LineString.h>
#include <lanelet2_core/primitives/Area.h>
#include <lanelet2_core/primitives/Lanelet.h>
#include <lanelet2_core/primitives/LineString.h>
#include <lanelet2_core/primitives/Point.h>
#include <lanelet2_core/primitives/Polygon.h>
#include <lanelet2_core/primitives/BasicRegulatoryElements.h>
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
#include <lanelet2_core/primitives/BasicRegulatoryElements.h>
#endif

namespace tk { namespace common {
    class LaneletInterface {
    public:
         LaneletInterface() = default;
        ~LaneletInterface() = default;

        bool init(const std::string &aConfPath);
        void buildRoutingGraph();

#ifdef LANELET_ENABLED
        lanelet::routing::RoutingGraphPtr mRoutingGraph;
        lanelet::LaneletMapUPtr mMap;
        double mOriginLat, mOriginLon;
        lanelet::projection::UtmProjector *mProjector;
#endif
    };
}}