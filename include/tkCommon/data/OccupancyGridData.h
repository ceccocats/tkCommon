#pragma once

#include "tkCommon/data/CloudData.h"
#include "tkCommon/data/ImageData.h"

#ifdef TKROS_ENABLED
#if TKROS_VERSION == 1
#include <nav_msgs/OccupancyGrid.h>
#endif
#if TKROS_VERSION == 2
#include <nav_msgs/msg/OccupancyGrid.hpp>
#endif
#endif

namespace tk {
namespace data {
class OccupancyGridData : public ImageDataF
{
  private:
    bool initted = false;
    
  public:
    float resolution;
    tk::math::Vec3<double> origin;

    void init(size_t aWidth,
              size_t aHeight,
              float aResolution = 0.1f,
              tk::math::Vec3<double> aOrigin = { 0.0, 0.0, 0.0 })
    {
        ImageDataF::init(aWidth, aHeight, 1);
        ImageDataF::clear();
        this->origin = aOrigin;
        this->resolution = aResolution;
        //this->data.writableMatrix().setZero();
        this->initted = true;
    }

    void fromCloud(tk::data::CloudData *aCloud,
                   float aResolution = 0.1f,
                   uint8_t aCellThreshold = 2,
                   size_t aMinWidth = 0,
                   size_t aMinHeight = 0,
                   float aMinObstacleHeight = 0.0f)
    {
        // find bounds
        float min_x, max_x, min_y, max_y, min_z, max_z;
        aCloud->bounds(min_x, max_x, min_y, max_y, min_z, max_z);

        // init
        float size_x = std::max(std::fabs(min_x), std::fabs(max_x));
        float size_y = std::max(std::fabs(min_y), std::fabs(max_y));
        size_t cloud_h = size_t((size_y * 2) / aResolution);
        size_t cloud_w = size_t((size_x * 2) / aResolution);
        if (cloud_w < aMinWidth && cloud_h < aMinHeight)
            init(aMinWidth,
                 aMinHeight,
                 aResolution,
                 { -(double(aMinWidth) / 2.0) * aResolution,
                   -(double(aMinHeight) / 2.0) * aResolution,
                   0.0 });
        else
            init(cloud_w, cloud_h, aResolution, { -size_x, -size_y, 0.0 });

        size_t c, r;
        if (aCellThreshold > 1) {
            tk::math::Mat<uint8_t> occGrid;
            occGrid.resize(this->height, this->width);
            occGrid.writableMatrix().setZero();
            for (size_t i = 0; i < aCloud->points.cols(); ++i) {
                if (point2grid(aCloud->points(0, i), aCloud->points(1, i), r, c))
                    if (aCloud->points(i, 2) > aMinObstacleHeight && occGrid(r, c) < 255)
                        occGrid(r, c) += 1;
            }
            
            for (size_t r = 0; r < this->height; ++r) {
                for (size_t c = 0; c < this->width; ++c) {
                    if (occGrid(r, c) > aCellThreshold)
                        this->at(r, c)[0] = 1.0f;
                    else
                        this->at(r, c)[0] = 0.0f;
                }
            }
        } else {
            for (size_t i = 0; i < aCloud->points.cols(); ++i)
                if (point2grid(aCloud->points(0, i), aCloud->points(1, i), r, c) &&
                    aCloud->points(i, 2) > aMinObstacleHeight)
                    this->at(r, c)[0] = 1.0f;
        }
    }

    bool point2grid(const float aX, const float aY, size_t &aR, size_t &aC) const
    {
        if (!this->initted)
            return false;

        aR = size_t((-aX) / this->resolution + this->height/2);
        aC = size_t((-aY) / this->resolution + this->width/2);

        if (aC >= 0 && aR < this->height && aR >= 0 && aC < this->width)
            return true;
        else
            return false;
    }

    bool grid2point(const size_t aR, const size_t aC, float &aX, float &aY) const
    {
        if (!this->initted)
            return false;

        aX = float(this->height/2 - aR) * resolution;
        aY = float(this->width/2  - aC) * resolution;

        return true;
    }

    OccupancyGridData& operator=(const OccupancyGridData& s)
    {
        ImageDataF::operator=(s);
        this->resolution = s.resolution;
        this->origin = s.origin;
        this->initted = s.initted;
        return *this;
    }
    friend std::ostream& operator<<(std::ostream& os, OccupancyGridData& s)
    {
        os<<"OccupancyGridData"<<std::endl;
        os<<"	type:  "; s.T_type.print(os); os<<std::endl;
        os<<"	header.name:  "<<s.header.name<<std::endl;
        os<<"	header.stamp: "<<s.header.stamp<<std::endl;
        os<<"	header.fps:   "<<s.header.fps<<std::endl;
        os<<"	data: "<<s.data<<std::endl;
        os<<"	width: "<<s.width<<std::endl;
        os<<"	height: "<<s.height<<std::endl;
        os<<"	channels: "<<s.channels<<std::endl;
        os<<"   resolution: "<<s.resolution<<std::endl;
        os<<"   origin: "<<s.origin<<std::endl;
        return os;
    }

#ifdef TKROS_ENABLED
#if TKROS_VERSION == 1
    void toRos(nav_msgs::OccupancyGrid &msg)
    {
#endif
#if TKROS_VERSION == 2
        void toRos(nav_msgs::msg::OccupancyGrid & msg)
        {
#endif
            if (!this->initted)
                return;
            this->header.toRos(msg.header);
            msg.info.width = this->height;
            msg.info.height = this->width;
            msg.info.origin.position.x = this->origin.y();
            msg.info.origin.position.y = this->origin.x();
            msg.info.resolution = this->resolution;
            msg.data.resize(this->width * this->height);
            for (int i = 0; i < this->height; ++i) {
                for (int j = 0; j < this->width; ++j) {

                    msg.data[j * this->height + i] = int8_t(this->at(this->height - i, this->width - j)[0] * 100.0f);
                }
            }
        }

#if TKROS_VERSION == 1
        void fromRos(nav_msgs::OccupancyGrid & msg)
        {
#endif
#if TKROS_VERSION == 2
            void fromRos(nav_msgs::msg::OccupancyGrid & msg)
            {
#endif
                this->header.fromRos(msg.header);
                init(msg.info.width, msg.info.height, msg.info.resolution, {msg.info.origin.position.x, msg.info.origin.position.y, 0.0});
                for (int i = 0; i < this->height; ++i) {
                    for (int j = 0; j < this->width; ++j) {
                        this->at(i, j)[0] = msg.data[i * this->height + j] / 100.0f;
                    }
                }
            }
#endif
        };
    }
}