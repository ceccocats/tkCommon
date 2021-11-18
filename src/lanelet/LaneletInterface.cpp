#include "tkCommon/lanelet/LaneletInterface.h"

using namespace tk::common;
#ifdef LANELET_ENABLED
bool 
LaneletInterface::init(const std::string &aConfPath) 
{
    YAML::Node conf = YAML::LoadFile(aConfPath);
    std::string map_p;
    if (aConfPath.find_last_of('/') != std::string::npos) 
        map_p = aConfPath.substr(0, aConfPath.find_last_of('/')) + "/";
    map_p       += tk::common::YAMLgetConf<std::string>(conf, "file", "lanelet.osm");
    mOriginLat  = tk::common::YAMLgetConf<double>(conf, "lat", 0.0f);
    mOriginLon  = tk::common::YAMLgetConf<double>(conf, "lon", 0.0f);
    
    mProjector  = new lanelet::projection::UtmProjector(lanelet::Origin({mOriginLat, mOriginLon}));
    mMap        = lanelet::load(map_p, *mProjector);

    if (!mMap->empty())
        return true;
    else
        return false;
}

void 
LaneletInterface::buildRoutingGraph() 
{
    auto trafficRules = lanelet::traffic_rules::TrafficRulesFactory::create(lanelet::Locations::Germany, lanelet::Participants::Vehicle);
    tkDBG("Building routing graph");
    mRoutingGraph = lanelet::routing::RoutingGraph::build(*mMap, *trafficRules);
}
#else
bool 
LaneletInterface::init(const std::string &aConfPath) 
{
    tkERR("You need to compile with lanelet2");
    return false;    
}

void 
LaneletInterface::buildRoutingGraph() 
{
    tkERR("You need to compile with lanelet2");
}
#endif