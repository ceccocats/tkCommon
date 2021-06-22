#pragma once

#include "tkCommon/common.h"

#include <GeographicLib/UTMUPS.hpp>

namespace tk { namespace projection {
    enum class ProjectionType {
        NONE,
        UTM
    };

    class Projector {
    public:
         Projector();
        ~Projector() = default;

        virtual void init(const tk::math::Vec3d aGps) = 0;
        virtual void init(const double aOriginLat, const double aOriginLon, const double aOriginEle) = 0;
        
        /**
         * @brief Project a point from lat/lon/ele coordinates to a local coordinate system.
         * 
         * @param aGps 
         * @return tk::math::Vec3d 
         */
        virtual tk::math::Vec3d forward(const tk::math::Vec3d aGps) = 0;

        /**
         * @brief Project a point from lat/lon/ele coordinates to a local coordinate system.
         * 
         * @param aLat 
         * @param aLon 
         * @param aEle 
         * @return tk::math::Vec3d 
         */
        virtual tk::math::Vec3d forward(const double aLat, const double aLon, const double aEle) = 0;
        
        /**
         * @brief Project a point from local coordinates to global lat/lon/ele coordinates.
         * 
         * @param aPoint 
         * @return tk::math::Vec3d 
         */
        virtual tk::math::Vec3d reverse(const tk::math::Vec3d aPoint) = 0;

        /**
         * @brief Project a point from local coordinates to global lat/lon/ele coordinates.
         * 
         * @param aX 
         * @param aY 
         * @param aZ 
         * @return tk::math::Vec3d 
         */
        virtual tk::math::Vec3d reverse(const double aX, const double aY, const double aZ) = 0;
    
        bool hasReference() const { return mHasReference; }
        tk::math::Vec3d getReference() const { return mReference; }
        tk::math::Vec3d getOffset() const { return mOffset; }
        ProjectionType getType() const { return mType; }
    protected:
        bool                mHasReference, mUseOffset;
        tk::math::Vec3d     mReference;
        tk::math::Vec3d     mOffset;
        ProjectionType      mType;
    };

    class UtmProjector : public Projector {
    public:
         UtmProjector();
        ~UtmProjector() = default;

        void init(const tk::math::Vec3d aGps) final;
        void init(const double aOriginLat, const double aOriginLon, const double aOriginEle) final;
        tk::math::Vec3d forward(tk::math::Vec3d aGps) final;
        tk::math::Vec3d forward(const double aLat, const double aLon, const double aEle) final;
        tk::math::Vec3d reverse(const tk::math::Vec3d aPoint) final;
        tk::math::Vec3d reverse(const double aX, const double aY, const double aZ) final;
    
        int getZone() const { return mZone; }
    private:
        int     mZone;
        bool    mIsInNorthernHemisphere;
    };
}}
