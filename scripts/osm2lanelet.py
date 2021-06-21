#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import print_function

import os, sys
import stat
import traceback
import webbrowser
import datetime
from argparse import ArgumentParser
import json
import threading
import subprocess
import tempfile
import shutil
from zipfile import ZipFile
import base64

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:   
    sys.exit("please declare environment variable 'SUMO_HOME'")
import osmGet
import osmBuild
import randomTrips
import ptlines2flows
import tileGet
import sumolib  # noqa
from webWizard.SimpleWebSocketServer import SimpleWebSocketServer, WebSocket
import yaml
import lanelet2
from lanelet2.core import Lanelet, AttributeMap, LineString3d, Point2d, Point3d, getId, LaneletMap, BoundingBox2d, BasicPoint2d, Area, BasicPoint3d, TrafficLight
from lanelet2.projection import UtmProjector

SUMO_HOME = os.environ.get("SUMO_HOME", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))

typemapdir = os.path.join(SUMO_HOME, "data", "typemap")
typemaps = {
    "net": os.path.join(typemapdir, "osmNetconvert.typ.xml"),
    "poly": os.path.join(typemapdir, "osmPolyconvert.typ.xml"),
    "urban": os.path.join(typemapdir, "osmNetconvertUrbanDe.typ.xml"),
    "pedestrians": os.path.join(typemapdir, "osmNetconvertPedestrians.typ.xml"),
    "ships": os.path.join(typemapdir, "osmNetconvertShips.typ.xml"),
    "bicycles": os.path.join(typemapdir, "osmNetconvertBicycle.typ.xml"),
}

vehicleParameters = {
    "passenger":  ["--vehicle-class", "passenger",  "--vclass", "passenger",  "--prefix", "veh",
                   "--min-distance", "300",  "--trip-attributes", 'departLane="best"',
                   "--fringe-start-attributes", 'departSpeed="max"',
                   "--allow-fringe.min-length", "1000",
                   "--lanes", "--validate"],
    "truck":      ["--vehicle-class", "truck", "--vclass", "truck", "--prefix", "truck", "--min-distance", "600",
                   "--fringe-start-attributes", 'departSpeed="max"',
                   "--trip-attributes", 'departLane="best"', "--validate"],
    "bus":        ["--vehicle-class", "bus",   "--vclass", "bus",   "--prefix", "bus",   "--min-distance", "600",
                   "--fringe-start-attributes", 'departSpeed="max"',
                   "--trip-attributes", 'departLane="best"', "--validate"],
    "motorcycle": ["--vehicle-class", "motorcycle", "--vclass", "motorcycle", "--prefix", "moto",
                   "--fringe-start-attributes", 'departSpeed="max"',
                   "--max-distance", "1200", "--trip-attributes", 'departLane="best"', "--validate"],
    "bicycle":    ["--vehicle-class", "bicycle",    "--vclass", "bicycle",    "--prefix", "bike",
                   "--fringe-start-attributes", 'departSpeed="max"',
                   "--max-distance", "8000", "--trip-attributes", 'departLane="best"', "--validate"],
    "tram":       ["--vehicle-class", "tram",       "--vclass", "tram",       "--prefix", "tram",
                   "--fringe-start-attributes", 'departSpeed="max"',
                   "--min-distance", "1200", "--trip-attributes",                'departLane="best"', "--validate"],
    "rail_urban": ["--vehicle-class", "rail_urban", "--vclass", "rail_urban", "--prefix", "urban",
                   "--fringe-start-attributes", 'departSpeed="max"',
                   "--min-distance", "1800", "--trip-attributes",                'departLane="best"', "--validate"],
    "rail":       ["--vehicle-class", "rail",       "--vclass", "rail",       "--prefix", "rail",
                   "--fringe-start-attributes", 'departSpeed="max"',
                   "--min-distance", "2400", "--trip-attributes",                'departLane="best"', "--validate"],
    "ship":       ["--vehicle-class", "ship",       "--vclass", "ship",       "--prefix", "ship", "--validate",
                   "--fringe-start-attributes", 'departSpeed="max"'],
    "pedestrian": ["--vehicle-class", "pedestrian", "--pedestrians", "--prefix", "ped",
                   "--max-distance", "2000", ],
    "persontrips": ["--vehicle-class", "pedestrian", "--persontrips", "--prefix", "ped",
                    "--trip-attributes", 'modes="public"', ],
}

vehicleNames = {
    "passenger": "Cars",
    "truck": "Trucks",
    "bus": "Bus",
    "motorcycle": "Motorcycles",
    "bicycle": "Bicycles",
    "pedestrian": "Pedestrians",
    "tram": "Trams",
    "rail_urban": "Urban Trains",
    "rail": "Trains",
    "ship": "Ships"
}        

class Builder(object):
    prefix = "osm"

    def __init__(self, data, local):
        self.files = {}
        self.files_relative = {}
        self.data = data
        self.originLat = 0
        self.originLon = 0

        self.tmp = None
        if local:
            now = data.get("outputDir",
                           datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
            for base in ['', os.path.expanduser('~/Sumo')]:
                try:
                    self.tmp = os.path.abspath(os.path.join(base, now))
                    os.makedirs(self.tmp)
                    break
                except Exception:
                    print("Cannot create directory '%s'" % self.tmp)
                    self.tmp = None
        if self.tmp is None:
            self.tmp = tempfile.mkdtemp()

        self.origDir = os.getcwd()
        print("Building scenario in '%s'" % self.tmp)

    def report(self, message):
        pass

    def filename(self, use, name, usePrefix=True):
        prefix = self.prefix if usePrefix else ''
        self.files_relative[use] = prefix + name
        self.files[use] = os.path.join(self.tmp, prefix + name)

    def getRelative(self, options):
        result = []
        dirname = self.tmp
        ld = len(dirname)
        for o in options:
            if isinstance(o, basestring) and o[:ld] == dirname:
                remove = o[:ld+1]
                result.append(o.replace(remove, ''))
            else:
                result.append(o)
        return result

    def export(self):
        self.report("Converting to lanelet2")

        net = sumolib.net.readNet(os.path.join(self.tmp, self.files["net"]), withInternal=True, withPedestrianConnections=True)

        print(net.getBoundary())
        print(net.getBBoxXY())
        print(net.getLocationOffset())

        #offsetX = net.getBoundary()[2] / 2
        #offsetY = net.getBoundary()[3] / 2
        offsetX = 0
        offsetY = 0

        lanelet_map = LaneletMap()
        for edge in net.getEdges():
            if edge.getFunction() == "connector" or edge.getFunction() == "walkingarea":
                continue
            '''
            if edge.getFunction() == "crossing":
                ls = LineString3d(getId(), [])
                ls.attributes["type"] = "virtual"
                ls.attributes["subtype"] = "crossing"
                
                for lane in edge.getLanes():
                    # add point to linestring
                    for point in lane.getShape3D(includeJunctions=False):
                        #print(point)
                        p = Point3d(getId(), point[0] - offsetX, point[1] - offsetY, point[2])
                    
                        # search nearest point
                        nearest_point = lanelet_map.pointLayer.nearest(BasicPoint2d(p.x, p.y), 1)
                        if len(nearest_point) > 0 and lanelet2.geometry.distance(p, nearest_point[0]) <= 0.05:
                            ls.append(nearest_point[0])
                        else:
                            ls.append(p)
                        
                    # add line to lanelet_map
                    lanelet_map.add(ls)
                continue
            '''
            dashed = "n"
            for ed in net.getEdges():
                if edge.getFromNode() == ed.getToNode() and edge.getToNode() == ed.getFromNode():
                    dashed = "l"
                    break

            for lane in edge.getLanes():

                if not lane.allows("passenger"):
                    continue

                ls = LineString3d(getId(), [])
                
                ls.attributes["type"] = "virtual"
                if edge.getFunction() == "internal":
                    ls.attributes["subtype"] = "internal"
                else:
                    ls.attributes["subtype"] = "-"
                
                ls.attributes["osm_type"] = edge.getType()
                ls.attributes["dashed"] = dashed

                # add point to linestring
                for point in lane.getShape3D(includeJunctions=False):
                    #print(point)
                    p = Point3d(getId(), point[0] - offsetX, point[1] - offsetY, point[2])
                  
                    # search nearest point
                    nearest_point = lanelet_map.pointLayer.nearest(BasicPoint2d(p.x, p.y), 1)
                    if len(nearest_point) > 0 and lanelet2.geometry.distance(p, nearest_point[0]) <= 0.05:
                        ls.append(nearest_point[0])
                    else:
                        ls.append(p)
                    
                # add line to lanelet_map
                lanelet_map.add(ls)

        # add building
        for poly in sumolib.xml.parse(os.path.join(self.tmp, self.files["poly"]), "poly"):        
            
            if "building" in poly.type:
                area_type = "building"
            elif "leisure.park" in poly.type:
                area_type = "vegetation"
            elif "leisure.garden" in poly.type:
                area_type = "vegetation"
            elif "amenity.parking" in poly.type:
                area_type = "parking"
            else:
                continue
            

            outer = LineString3d(getId(), [])
            for idx,s_point in enumerate(poly.shape.split(" ")):
                # dont add last element
                if idx == (len(poly.shape.split(" ")) - 1):
                    break
                
                point = s_point.split(",")
                
                # search if point alredy exist
                searchBox = BoundingBox2d(BasicPoint2d(float(point[0])-0.05, float(point[1])-0.05), BasicPoint2d(float(point[0])+0.05, float(point[1])+0.05))
                pt = lanelet_map.pointLayer.search(searchBox)
                if len(pt)>=1:
                    p = pt[0]
                else:
                    p = Point2d(getId(), float(point[0]) - offsetX, float(point[1]) - offsetY)

                # save first element for later
                if idx == 0:
                    first_p = p

                outer.append(lanelet2.geometry.to3D(p))

            outer.append(lanelet2.geometry.to3D(first_p))
            area = Area(getId(), [outer])
            area.attributes["subtype"] = area_type

            # add area to map
            lanelet_map.add(area)
        
        # save lanelet_map
        print("Writing lanelet.osm")
        projector = UtmProjector(lanelet2.io.Origin(self.originLat, self.originLon))
        lanelet2.io.write("lanelet.osm", lanelet_map, projector)

        print("Writing lanelet.osm.yaml")
        yml_dict = {}
        yml_dict['file'] = os.path.join(self.tmp, "lanelet.osm")
        yml_dict['lat'] = self.originLat
        yml_dict['lon'] = self.originLon
        with open("lanelet.osm.yaml", 'w') as file:
            file.write(yaml.dump(yml_dict))
    
        print("Success.")

    def build(self):
        # output name for the osm file, will be used by osmBuild, can be
        # deleted after the process
        self.filename("osm", "_bbox.osm.xml")
        # output name for the net file, will be used by osmBuild, randomTrips and
        # sumo-gui
        self.filename("net", ".net.xml")

        # download map data
        self.report("Downloading map data")
        #print(map(str, self.data["coords"]))
        osmGet.get(
            ["-b", ",".join(map(str, self.data["coords"])), "-p", self.prefix, "-d", self.tmp])

        options = ["-f", self.files["osm"], "-p", self.prefix, "-d", self.tmp]
        self.additionalFiles = []
        self.routenames = []

        if self.data["poly"]:
            # output name for the poly file, will be used by osmBuild and
            # sumo-gui
            self.filename("poly", ".poly.xml")
            options += ["-m", typemaps["poly"]]
            self.additionalFiles.append(self.files["poly"])

        typefiles = [typemaps["net"]]
        # leading space ensures that arguments starting with -- are not
        # misinterpreted as options
        #netconvertOptions = " " + osmBuild.DEFAULT_NETCONVERT_OPTS
        netconvertOptions = " --geometry.remove,--roundabouts.guess,--ramps.guess,-v,--junctions.join,--output.original-names,--junctions.corner-detail,5,--output.street-names"
        #netconvertOptions += ",--tls.guess-signals" #,--tls.discard-simple,--tls.join,--tls.default-type,actuated"
        # remove turnaround
        netconvertOptions += ",--no-turnarounds"
        # add elevation
        netconvertOptions += ",--osm.elevation,--osm.layer-elevation,5"
        if "pedestrian" in self.data["vehicles"]:
            # sidewalks are already included via typefile
            netconvertOptions += ",--crossings.guess"
            typefiles.append(typemaps["urban"])
            typefiles.append(typemaps["pedestrians"])
        if "ship" in self.data["vehicles"]:
            typefiles.append(typemaps["ships"])
        if "bicycle" in self.data["vehicles"]:
            typefiles.append(typemaps["bicycles"])
        # special treatment for public transport
        if self.data["publicTransport"]:
            self.filename("stops", "_stops.add.xml")
            netconvertOptions += ",--ptstop-output,%s" % self.files["stops"]
            self.filename("ptlines", "_ptlines.xml")
            self.filename("ptroutes", "_pt.rou.xml")
            netconvertOptions += ",--ptline-output,%s" % self.files["ptlines"]
            self.additionalFiles.append(self.files["stops"])
            self.routenames.append(self.files["ptroutes"])
            netconvertOptions += ",--railway.topology.repair"
        if self.data["leftHand"]:
            netconvertOptions += ",--lefthand"
        if self.data["carOnlyNetwork"]:
            if self.data["publicTransport"]:
                options += ["--vehicle-classes", "publicTransport"]
            else:
                options += ["--vehicle-classes", "passenger"]

        # get origin
        self.originLat = (self.data["coords"][1] + self.data["coords"][3])/2
        self.originLon = (self.data["coords"][0] + self.data["coords"][2])/2

        # apply utm projection
        netconvertOptions +=",--proj,+proj=utm +zone=32 +ellps=WGS84 +datum=WGS84 +units=m +no_defs +lat_0="+str(self.originLat)+" +lon_0="+str(self.originLon)
        #netconvertOptions +=",--edges.join,--default.lanenumber,2,--default.lanewidth,3.5"


        options += ["--netconvert-typemap", ','.join(typefiles)]
        options += ["--netconvert-options", netconvertOptions]

        self.report("Converting map data")
        osmBuild.build(options)
    
    def finalize(self):
        try:
            shutil.rmtree(self.tmp)
        except Exception:
            pass

class OSMImporterWebSocket(WebSocket):

    local = False
    outputDir = None

    def report(self, message):
        print(message)
        self.sendMessage(u"report " + message)
        # number of remaining steps
        self.steps -= 1

    def handleMessage(self):
        data = json.loads(self.data)

        thread = threading.Thread(target=self.build, args=(data,))
        thread.start()

    def build(self, data):
        print("build")

        #if self.outputDir is not None:
        #    data['outputDir'] = self.outputDir
        builder = Builder(data, self.local)
        builder.report = self.report

        self.steps = 3
        self.sendMessage(u"steps %s" % self.steps)

        try:
            builder.build()
            builder.export()
            #builder.finalize()

        except Exception:
            print(traceback.format_exc())
            # reset 'Generate Scenario' button
            while self.steps > 0:
                self.report("Recovering")
        os.chdir(builder.origDir)

if __name__ == "__main__":
    OSMImporterWebSocket.local = True

    # open sumo web gui to select area
    webbrowser.open("file://" + os.path.join(SUMO_HOME, "tools", "webWizard", "index.html"))

    server = SimpleWebSocketServer("", 8010, OSMImporterWebSocket)
    server.serveforever()