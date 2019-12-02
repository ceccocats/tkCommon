#include "tkCommon/data/HeaderData.h"

namespace tk { namespace data {
    HeaderData::HeaderData() {
        this->stamp         = 0;
        this->tf            = tk::common::Tfpose::Identity();
        this->sensorID      = 0;
        this->messageID     = 0;
        this->name          = "sensor";
    }

    HeaderData&
    HeaderData::operator=(const HeaderData &s) {
        this->stamp         = s.stamp;
        this->tf            = s.tf;
        this->sensorID      = s.sensorID;
        this->messageID     = s.messageID;
        this->name          = s.name;
    }
}}