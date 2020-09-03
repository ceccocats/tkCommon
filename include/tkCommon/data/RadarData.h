#pragma once

#include "tkCommon/data/SensorData.h"
#include "tkCommon/data/CloudData.h"
#include <vector>

namespace tk { namespace data {
    static const int RADAR_MAX          = 6;
    static const int RADAR_MAX_PONINTS  = 5000;
    static const int RADAR_MAX_FEATURES = 8;

    /**
     *
     */
    enum RadarFeatureType {
        VELOCITY       = 0,
        FALSE_DET      = 1,
        RANGE_VAR      = 2,
        VELOCITY_VAR   = 3,
        ANGLE_VAR      = 4,
        RCS            = 5,
        ID             = 6,
        PROB           = 7
    };

    /**
     * @brief Radar data class.
     * Contains all the information generated from a generic Radar sensor.
     */
    class RadarData : public SensorData {
    public:
        int nRadar;     /**< Number of radar  */
        std::vector<tk::data::CloudData>  near_data;    /**< Vector of @see tk::data::CloudData */
        std::vector<tk::data::CloudData>  far_data;     /**< Vector of @see tk::data::CloudData */
        std::vector<Eigen::MatrixXf>      near_features;
        std::vector<Eigen::MatrixXf>      far_features;


        /**
         * @brief Initialization method.
         * Handles the allocation of dynamic vector of @see tk::data::CloudData at maximum size.
         */
        void init() {
            SensorData::init();

            this->near_data.resize(RADAR_MAX);
            this->far_data.resize(RADAR_MAX);
            this->near_features.resize(RADAR_MAX);
            this->far_features.resize(RADAR_MAX);

            this->nRadar = RADAR_MAX;
            for (int i = 0; i < RADAR_MAX; i++) {
                this->near_data[i].init(RADAR_MAX_PONINTS);
                this->far_data[i].init(RADAR_MAX_PONINTS);

                this->near_features[i].resize(RADAR_MAX_FEATURES, RADAR_MAX_PONINTS);
                this->far_features[i].resize(RADAR_MAX_FEATURES, RADAR_MAX_PONINTS);
            }
            header.sensor = tk::data::sensorName::RADAR;
        }

        /**
         * @brief Release method.
         * Handles the deallocation of of dynamic vector of @see tk::data::CloudData.
         */
        void release() {
            for (int i = 0; i < RADAR_MAX; i++) {
                this->near_data[i].release();
                this->far_data[i].release();

                this->near_features[i].resize(0, 0);
                this->far_features[i].resize(0, 0);
            }

            this->near_data.clear();
            this->far_data.clear();
            this->near_features.clear();
            this->far_features.clear();
            this->nRadar = 0;
        }



        bool checkDimension(SensorData *s) {
            auto *source = dynamic_cast<RadarData*>(s);
            if (this->near_data.size() < source->near_data.size() ||
                this->far_data.size() < source->far_data.size() ||
                this->near_features.size() < source->near_features.size() ||
                this->far_features.size() < source->far_features.size()
                ) {
                return false;
            } else {
                return true;
            }
        }

        /**
         * @brief Overloading of operator = for class copy.
         * @param s
         * @return
         */
        RadarData& operator=(const RadarData& s) {
            SensorData::operator=(s);

            this->nRadar    = s.nRadar;
            for (int i = 0; i < nRadar; i++) {
                this->near_data[i]    = s.near_data[i];
                this->far_data[i]     = s.far_data[i];

                std::memcpy(this->near_features[i].data(), s.near_features[i].data(), RADAR_MAX_PONINTS * RADAR_MAX_FEATURES * sizeof(float));
                std::memcpy(this->far_features[i].data(), s.far_features[i].data(), RADAR_MAX_PONINTS * RADAR_MAX_FEATURES * sizeof(float));
            }

            return *this;
        }

        void draw(tk::gui::Viewer *viewer){
            glPushMatrix();

			// TODO: move to tk::gui::Viewer
            tk::gui::Viewer::tkApplyTf(header.tf);

			tk::common::Vector3<float> pose;
			tk::common::Tfpose  correction = tk::common::odom2tf(0, 0, 0, +M_PI/2);
			for(int i = 0; i < nRadar; i++) {
				glPushMatrix();
				tk::gui::Viewer::tkDrawTf(near_data[i].header.name, (near_data[i].header.tf * correction));
				tk::gui::Viewer::tkApplyTf(near_data[i].header.tf);
				// draw near
				for (int j = 0; j < near_data[i].nPoints; j++) {
					float rcs = near_features[i](tk::data::RadarFeatureType::RCS, j);

					//NewValue = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
					float hue = (((rcs + 40) * (1 - 0)) / (20 + 40)) + 0;
					tk::gui::Viewer::tkSetRainbowColor(hue);

					pose.x = near_data[i].points(0, j);
					pose.y = near_data[i].points(1, j);
					pose.z = near_data[i].points(2, j);

					tk::gui::Viewer::tkDrawCircle(pose, 0.05);
				}
				//// draw far
				for (int j = 0; j < far_data[i].nPoints; j++) {
				    float rcs = far_features[i](tk::data::RadarFeatureType::RCS, j);

			        //NewValue = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
			        float hue = (((rcs + 40) * (1 - 0)) / (20 + 40)) + 0;
			        tk::gui::Viewer::tkSetRainbowColor(hue);

				    pose.x = far_data[i].points(0, j);
				    pose.y = far_data[i].points(1, j);
				    pose.z = far_data[i].points(2, j);

				    tk::gui::Viewer::tkDrawCircle(pose, 0.05);
				}
				glPopMatrix();
			}
            glPopMatrix();
        }

    };
}}