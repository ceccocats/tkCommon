#include "tkCommon/projection/Projector.h"

using namespace tk::projection;

Projector::Projector()
{
    mType           = ProjectionType::NONE;
    mHasReference   = false;
}


UtmProjector::UtmProjector() 
{
    mType                   = ProjectionType::UTM;
    mIsInNorthernHemisphere = true;
}

void 
UtmProjector::init(const tk::math::Vec3d aGps)
{
    tkDBG("init proj: "<<aGps.x()<<" "<<aGps.y()<<" "<<aGps.z());
    init(aGps.x(), aGps.y(), aGps.z());
}

void 
UtmProjector::init(const double aOriginLat, const double aOriginLon, const double aOriginEle)
{   
    tkDBG("init proj: "<<aOriginLat<<" "<<aOriginLon<<" "<<aOriginEle);
    GeographicLib::UTMUPS::Forward(aOriginLat, aOriginLon, mZone, mIsInNorthernHemisphere, mOffset.x(), mOffset.y()); 
    mOffset.z() = 0.0;

    mUseOffset      = true;
    mHasReference   = true;
}

tk::math::Vec3d 
UtmProjector::forward(const double aLat, const double aLon, const double aEle)
{    
    int     zone;
    bool    northp;
    tk::math::Vec3d point;
    try {
        GeographicLib::UTMUPS::Forward(aLat, aLon, zone, northp, point.x(), point.y());
    } catch (GeographicLib::GeographicErr& e) {
        tkERR(e.what());
    }

    if (zone != mZone || northp != mIsInNorthernHemisphere) {
        tkWRN("You have left the UTM zone or changed the hemisphere!");
        
        // try to transfer to the desired zone
        double xAfterTransfer = 0;
        double yAfterTransfer = 0;
        int zoneAfterTransfer = 0;
        try {
            GeographicLib::UTMUPS::Transfer(zone, northp, point.x(), point.y(), mZone, mIsInNorthernHemisphere, xAfterTransfer,
                                        yAfterTransfer, zoneAfterTransfer);
        } catch (GeographicLib::GeographicErr& e) {
            tkERR(e.what());
        }

        if (zoneAfterTransfer != mZone) {
            tkWRN("You have left the padding area of the UTM zone!");
        }

        point.x() = xAfterTransfer;
        point.y() = yAfterTransfer;
    }

    // apply reference
    if (mUseOffset)
        point.writableMatrix() -= mOffset.matrix();
    
    return point;
}

tk::math::Vec3d 
UtmProjector::forward(const tk::math::Vec3d aGps)
{
    return forward(aGps.x(), aGps.y(), aGps.z());
}

tk::math::Vec3d  
UtmProjector::reverse(const double aX, const double aY, const double aZ)
{
    tk::math::Vec3d gps;

    try {
        GeographicLib::UTMUPS::Reverse(mZone, mIsInNorthernHemisphere, mUseOffset ? aX + mOffset.x() : aX,
                                       mUseOffset ? aY + mOffset.y() : aY, gps.x(), gps.y());
        gps.z() = 0.0;
    } catch (GeographicLib::GeographicErr& e) {
        tkERR(e.what());
    }

    return gps;
}

tk::math::Vec3d  
UtmProjector::reverse(const tk::math::Vec3d aPoint)
{    
    return reverse(aPoint.x(), aPoint.y(), aPoint.z());
}