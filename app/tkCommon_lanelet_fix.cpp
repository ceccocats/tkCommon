#include <tkCommon/common.h>

#include <tkCommon/lanelet/Lanelet.h>

YAML::Node  conf;
std::string laneletConfPath, laneletMapPath, laneletMapFixedPath;
float       lat, lon;
bool        debug;
int         merged;

#ifdef LANELET_ENABLED
void
createLanelet(lanelet::LineString3d ls, lanelet::LaneletMap *map, float width = 1.0f, float merge_distance = 0.4f)
{
    lanelet::LineString3d left, right;
            
    //calculate left
    auto lsHybrid   = lanelet::utils::toHybrid(lanelet::utils::to2D(ls));
    auto basicLeft  = lanelet::geometry::offsetNoThrow(lsHybrid, width);
    for (int i = 0; i < basicLeft.size(); ++i) {
        if (map->pointLayer.size() == 0) {
            left.push_back(lanelet::Point3d(lanelet::utils::getId(), {basicLeft.at(i).x(), basicLeft.at(i).y(), 0.0}));
        } else {
            auto points = lanelet::geometry::findNearest(map->pointLayer, basicLeft.at(i), 1);
            assert(!points.empty());

            if (points[0].first <= merge_distance) {
                left.push_back(points[0].second); 
                ++merged;
            } else {
                left.push_back(lanelet::Point3d(lanelet::utils::getId(), {basicLeft.at(i).x(), basicLeft.at(i).y(), 0.0}));
            }
        }
    }
    

    // calculate right
    auto basicRight = lanelet::geometry::offset(lsHybrid, -width);
    for (int i = 0; i < basicRight.size(); ++i) {
        if (map->pointLayer.size() == 0) {
            right.push_back(lanelet::Point3d(lanelet::utils::getId(), {basicRight.at(i).x(), basicRight.at(i).y(), 0.0}));
        } else {
            auto points = lanelet::geometry::findNearest(map->pointLayer, basicRight.at(i), 1);
            assert(!points.empty());

            if (points[0].first <= merge_distance) {
                right.push_back(points[0].second); 
                ++merged;
            } else { 
                right.push_back(lanelet::Point3d(lanelet::utils::getId(), {basicRight.at(i).x(), basicRight.at(i).y(), 0.0}));
            }
        }
    }

    // set line attribute
    if (ls.attribute(lanelet::AttributeName::Subtype) == "-") {
        left.setAttribute(lanelet::AttributeName::Type, "line_thin");
        if (ls.hasAttribute("dashed") && ls.attribute("dashed") == "l")
            left.setAttribute(lanelet::AttributeName::Subtype, "dashed");
        else 
            left.setAttribute(lanelet::AttributeName::Subtype, "solid");

        right.setAttribute(lanelet::AttributeName::Type, "line_thin");
        right.setAttribute(lanelet::AttributeName::Subtype, "solid");
    } else if (ls.attribute(lanelet::AttributeName::Subtype) == "crossing") {
        left.setAttribute(lanelet::AttributeName::Type, "zebra_marking");
        left.setAttribute(lanelet::AttributeName::Subtype, "-");
        right.setAttribute(lanelet::AttributeName::Type, "zebra_marking");
        right.setAttribute(lanelet::AttributeName::Subtype, "-");
    }
    

    // create lanelet
    lanelet::Lanelet lanelet1(lanelet::utils::getId(), left, right);
    lanelet1.setAttribute(lanelet::AttributeName::Type,     "lanelet");
    if (ls.attribute(lanelet::AttributeName::Subtype) == "-") {
        lanelet1.setAttribute(lanelet::AttributeName::Subtype,  "road");
        lanelet1.setAttribute(lanelet::AttributeName::Location, "urban");
        if (ls.hasAttribute("speed"))
            lanelet1.setAttribute(lanelet::AttributeName::SpeedLimit, ls.attributes()["speed"]);
    } else if (ls.attribute(lanelet::AttributeName::Subtype) == "crossing") {
        lanelet1.setAttribute(lanelet::AttributeName::Subtype,  "crosswalk");
        lanelet1.setAttribute(lanelet::AttributeName::Location, "-");
    }

    map->add(lanelet1);
}
#endif

int
main (int argc, char *argv[])
{
    tk::common::CmdParser cmd(argv, "tkCommon_osm_to_lanelet2");
    laneletConfPath     = cmd.addArg("lanelet_path", "lanelet.osm.yaml", "lanelet2 input file");
    laneletMapFixedPath = cmd.addOpt("-out", "lanelet_fixed.osm", "lanelet2 output file");
    debug               = cmd.addBoolOpt("-d", "enable debug file generation");
    cmd.parse();

    // get conf from YAML
    tk::common::Lanelet lanelet;
    lanelet.init(laneletConfPath);

    #ifdef LANELET_ENABLED

    // load lanelet map
    // http://www.dirsig.org/docs/new/coordinates.html
    lanelet::LaneletMap tmpLane;
    lanelet::LaneletMap tmpJunction;

    if (lanelet.mMap->empty()) {
        tkERR("Empty map.\n");
        return -1;
    }

    // param
    float   lane_merge_distance = 0.4f;
    float   junction_merge_distance = 1.5f;
    float   road_width = 3.2f/2.0f;
    float   crossing_width = 1.0f;

    for(auto& line : lanelet.mMap->lineStringLayer) {
        if (!line.hasAttribute(lanelet::AttributeName::Type) || !line.hasAttribute(lanelet::AttributeName::Type))
            continue;
        
        // crossing
        if (line.attribute(lanelet::AttributeName::Type) == "virtual" && line.attribute(lanelet::AttributeName::Subtype) == "crossing") {
            createLanelet(line, &tmpLane, crossing_width, 0.0f);
        }

        // normal lane
        if (line.attribute(lanelet::AttributeName::Type) == "virtual" && line.attribute(lanelet::AttributeName::Subtype) == "-") {
            createLanelet(line, &tmpLane, road_width, lane_merge_distance);
        }
    }

    tkDBG("Meged "<<merged<<" lane points\n");
    merged = 0;

    // junction
    for(auto& line : lanelet.mMap->lineStringLayer) {
        if (!line.hasAttribute(lanelet::AttributeName::Type) || !line.hasAttribute(lanelet::AttributeName::Type))
            continue;
        
        if (line.attribute(lanelet::AttributeName::Type) == "virtual" && line.attribute(lanelet::AttributeName::Subtype) == "internal") {
            float l = lanelet::geometry::length(line);
            if (l < lane_merge_distance) {
                tkDBG("Skipping junction with lenght "<<l<<"\n")
                continue;
            }
            lanelet::LineString3d left, right;       

            //calculate left
            auto lsHybrid   = lanelet::utils::toHybrid(lanelet::utils::to2D(line));
            auto basicLeft  = lanelet::geometry::offsetNoThrow(lsHybrid, road_width);
            for (int i = 0; i < basicLeft.size(); ++i) {
                auto lane_points = lanelet::geometry::findNearest(tmpLane.pointLayer, basicLeft.at(i), 1);
                assert(!lane_points.empty());
                
                if (lane_points[0].first <= junction_merge_distance) {
                    left.push_back(lane_points[0].second); 
                    ++merged;
                } else {
                    if (tmpJunction.pointLayer.size() == 0) {
                        left.push_back(lanelet::Point3d(lanelet::utils::getId(), {basicLeft.at(i).x(), basicLeft.at(i).y(), 0.0}));
                    } else {
                        auto junction_points = lanelet::geometry::findNearest(tmpJunction.pointLayer, basicLeft.at(i), 1);
                        assert(!junction_points.empty());

                        if (junction_points[0].first <= 0.2) {
                            left.push_back(junction_points[0].second); 
                            ++merged;
                        } else {
                            left.push_back(lanelet::Point3d(lanelet::utils::getId(), {basicLeft.at(i).x(), basicLeft.at(i).y(), 0.0}));
                        }
                    }
                }
            }   
            left.setAttribute(lanelet::AttributeName::Type, "virtual");
            left.setAttribute(lanelet::AttributeName::Subtype, "-");

            // calculate right
            auto basicRight = lanelet::geometry::offset(lsHybrid, -road_width);
            for (int i = 0; i < basicRight.size(); ++i) {
                auto lane_points = lanelet::geometry::findNearest(tmpLane.pointLayer, basicRight.at(i), 1);
                assert(!lane_points.empty());
                
                if (lane_points[0].first <= junction_merge_distance) {
                    right.push_back(lane_points[0].second); 
                    ++merged;
                } else {
                    if (tmpJunction.pointLayer.size() == 0) {
                        right.push_back(lanelet::Point3d(lanelet::utils::getId(), {basicRight.at(i).x(), basicRight.at(i).y(), 0.0}));
                    } else {
                        auto junction_points = lanelet::geometry::findNearest(tmpJunction.pointLayer, basicRight.at(i), 1);
                        assert(!junction_points.empty());

                        if (junction_points[0].first <= 0.2) {
                            right.push_back(junction_points[0].second); 
                            ++merged;
                        } else {
                            right.push_back(lanelet::Point3d(lanelet::utils::getId(), {basicRight.at(i).x(), basicRight.at(i).y(), 0.0}));
                        }
                    }
                }
            }
            right.setAttribute(lanelet::AttributeName::Type, "virtual");
            right.setAttribute(lanelet::AttributeName::Subtype, "-");

            // create lanelet
            lanelet::Lanelet lanelet1(lanelet::utils::getId(), left, right);
            lanelet1.setAttribute(lanelet::AttributeName::Type, "lanelet");
            lanelet1.setAttribute(lanelet::AttributeName::Subtype, "road");
            lanelet1.setAttribute(lanelet::AttributeName::Location, "urban");
            if (line.hasAttribute("speed"))
                lanelet1.setAttribute(lanelet::AttributeName::SpeedLimit, line.attributes()["speed"]);

            tmpJunction.add(lanelet1);
        }
    }
    tkDBG("Meged "<<merged<<" junction points\n");

    // fix junction lane marking
    /*
    for(auto& line : tmpJunction.lineStringLayer) {

        //tmpJunction.lineStringLayer.nearest()
        auto searchBox      = lanelet::BoundingBox2d(lanelet::utils::to2D(lanelet::utils::toBasicPoint(line[0])), lanelet::utils::to2D(lanelet::utils::toBasicPoint(line[line.size() -1])));
        auto searchResult   = tmpJunction.lineStringLayer.search(searchBox);
        bool intersect      = false;
        for (auto &result : searchResult) {
            if (line.id() == result.id())
                continue;
            
            intersect = lanelet::geometry::intersects(lanelet::utils::to2D(lanelet::utils::toHybrid(line)), lanelet::utils::to2D(lanelet::utils::toHybrid(result)));
            if (intersect) 
                break;
        }

        if (!intersect) {
            line.setAttribute(lanelet::AttributeName::Type, "line_thin");
            line.setAttribute(lanelet::AttributeName::Subtype, "solid");
        }
    }
    */

    for (auto& point : lanelet.mMap->pointLayer) {
        if (point.hasAttribute(lanelet::AttributeName::Type) && point.attribute(lanelet::AttributeName::Type) == "traffic_light") {
            std::cout<<point.x()<<"\t"<<point.y()<<"\n";
        }
    }

    // merge
    lanelet::LaneletMap outMap;
    for(auto& lanelet : tmpLane.laneletLayer)
        outMap.add(lanelet);
    for(auto& lanelet : tmpJunction.laneletLayer)
        outMap.add(lanelet);
    for(auto& area : lanelet.mMap->areaLayer)
        outMap.add(area);

    // save
    lanelet::write(laneletMapFixedPath, outMap, *lanelet.mProjector);
    
    // debug
    if (debug) {
        lanelet::write("lanelet_fixed_lane.osm", tmpLane, *lanelet.mProjector);
        lanelet::write("lanelet_fixed_junction.osm", tmpJunction, *lanelet.mProjector);
    }

    #endif
    return 0;
}