from _codeGen import genData
 
className = "GPSData"
VARS = [ {"name":"lat", "type":"double", "default":"0"}, 
         {"name":"lon", "type":"double", "default":"0"}, 
         {"name":"heigth", "type":"double", "default":"0"},
         {"name":"quality", "type":"int", "default":"0"} ]
genData(className, VARS)

className = "ImageData"
VARS = [ {"name":"data", "type":"uint8_t", "default":"0"} ]
genData(className, VARS)