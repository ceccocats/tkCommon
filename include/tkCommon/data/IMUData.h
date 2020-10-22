#pragma once

#include "tkCommon/data/SensorData.h"

namespace tk { namespace data {

    /**
     *  IMU message frame definition
     */
    class IMUData : public SensorData {
    public:
        uint64_t    system_timestamp;

        uint64_t    accelerometer_timestamp;
        float       acceleration_x;
        float       acceleration_y;
        float       acceleration_z;

        uint64_t    gyroscope_timestamp;
        float       anglualrVelocity_x;
        float       anglualrVelocity_y;
        float       anglualrVelocity_z;

        void init() override {
            SensorData::init();

            system_timestamp        = 0;
            accelerometer_timestamp = 0;
            gyroscope_timestamp     = 0;

            acceleration_x = 0;
            acceleration_y = 0;
            acceleration_z = 0;

            anglualrVelocity_x = 0;
            anglualrVelocity_y = 0;
            anglualrVelocity_z = 0;

            header.sensor = sensorName::IMU;
        }

        void release() override {}

        /**
         *
         * @param s
         * @return
         */
        bool checkDimension(SensorData *s) override {
            auto *source = dynamic_cast<IMUData *>(s);
            return true;
        }

        /**
         *
         * @param s
         * @return
         */
        IMUData& operator=(const IMUData& s) {
            SensorData::operator=(s);

            this->system_timestamp          = s.system_timestamp;
            this->accelerometer_timestamp   = s.accelerometer_timestamp;
            this->gyroscope_timestamp       = s.gyroscope_timestamp;

            this->acceleration_x = s.acceleration_x;
            this->acceleration_y = s.acceleration_y;
            this->acceleration_z = s.acceleration_z;

            this->anglualrVelocity_x = s.anglualrVelocity_x;
            this->anglualrVelocity_y = s.anglualrVelocity_y;
            this->anglualrVelocity_z = s.anglualrVelocity_z;

            return *this;
        }

        void draw(){
        	int n = 4;

        	glPushMatrix();
        	{
				tk::gui::Viewer::tkDrawTf(header.name, header.tf);
				tk::gui::Viewer::tkApplyTf(header.tf);

				float angle = atan2(acceleration_y, acceleration_x) * 180.0f / M_PI;
				float length = sqrt(acceleration_y*acceleration_y + acceleration_x*acceleration_x);
				glRotatef(angle, 0, 0, 1);
				glScalef(length, length, length);

				glBegin(GL_TRIANGLES);
				glColor4f(0,0,1,0.8);
				for(int i = 0; i < n; i++){
					glVertex3f(i + 0,-0.5, 0);
					glVertex3f(i + 0, 0.5, 0);
					glVertex3f(i + 0.7, 0, 0);
				}
				glEnd();
        	}
        	glPopMatrix();

        }
    };
    
}}