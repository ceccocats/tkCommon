#pragma once
#include "tkCommon/common.h"
#include "tkCommon/data/SensorData.h"
#include "tkCommon/data/GPSData.h"
#include "tkCommon/data/VehicleData.h"
#include "tkCommon/data/ImageData.h"
#include <vector>

///////////////////////////////////////////////in fondo la vecchia

namespace tk{ namespace perception{

/**
 * @brief enum perception type used for polymorphism
 */
enum class type{
    NOT_SET  = 0,
    LANE     = 1,
    BOX2D    = 2,
    RT2DBOX  = 3,
    BOX3D    = 4,
    RT3DBOX  = 5,
    SIGN     = 6,
	BOUNDARY = 7
};

/**
 * @brief lane type
 */
class lane_type{

    public:
        enum Value : uint8_t{
        NOT_CLS = 0,
        DASHED  = 1,
        OTHER   = 2,
        SOLID   = 3};

        /**
         * @brief   method for convert id to lane string name
         */
        std::string toString(){
            if(value == lane_type::NOT_CLS) return std::string{"not classified"};
            if(value == lane_type::DASHED)  return std::string{"skip"};
            if(value == lane_type::OTHER)   return std::string{"other"};
            if(value == lane_type::SOLID)   return std::string{"solid"};
            return std::string{"type error"};
        }

        lane_type& operator=(const Value & s){
			this->value = s;
			return *this;
		}

        bool operator!=(lane_type::Value v) noexcept {
            return v != value;
        }

        bool operator==(lane_type::Value v) noexcept {
            return v == value;
        }

        lane_type::Value value;
};

/**
 * @brief detected object type
 */
class obj_class{

    public:
        //Setted like masa-protocol
        enum Value : uint8_t{
        NOT_CLS     = 0,
        PEDESTRIAN  = 14,
        CAR         = 6,
        MOTOBIKE    = 13,
        CYCLE       = 1,
        // Added later
		ROADSIGN	= 5,
		LIGHT		= 7};

        /**
         * @brief   method for convert id to box detection string name
         */
        std::string toString(){
            if(value == obj_class::NOT_CLS)      return std::string{"not classified"};
            if(value == obj_class::PEDESTRIAN)   return std::string{"pedestrian"};
            if(value == obj_class::CAR)          return std::string{"car"};
            if(value == obj_class::MOTOBIKE)     return std::string{"motobike"};
            if(value == obj_class::CYCLE)        return std::string{"cycle"};
			if(value == obj_class::ROADSIGN)     return std::string{"road sign"};
			if(value == obj_class::LIGHT)     	 return std::string{"traffic light"};
            return std::string{"type error"};
        }

		obj_class& operator=(const Value & s){
			this->value = s;
			return *this;
		}

		bool operator!=(obj_class::Value v) noexcept {
			return v != value;
		}

		bool operator==(obj_class::Value v) noexcept {
			return v == value;
		}

		obj_class::Value value;
};

/**
 * @brief sign types
 */
class sign_type{

    public:
        enum Value : uint8_t{
        NOT_CLS             = 0,
        STOP                = 1,
        RIGHT_CURVE         = 2,
        LEFT_CURVE          = 3,
        RIGHT_REVERSE_BEND  = 4,
        LEFT_REVERSE_BEND   = 5,
        STEEP_ASCENT        = 6,
        STEEP_DESCENDT      = 7,
        NARROW_ROAD         = 8,
        SLIPPERY_ROAD       = 9,
        MAX_VELOCITY        = 10,
        MIN_VELOCITY        = 11,
        PESESTRIAN_CROSSING = 12,
        AHEAD_ONLY          = 13,
        TURN_LEFT           = 14,
        TURN_RIGHT          = 15,
        DEAD_STREET         = 16,
        PRIORITY_IN_VEH     = 17,
        PARKING             = 18,
        SCHOOL_CROSSING     = 19,
        NO_WAITING          = 20,
        NO_STOPPING         = 21,
        NO_ENTRY            = 22};

        /**
         * @brief   method for convert id to road sign detection string name
         */
        std::string toString(){
            if(value == sign_type::NOT_CLS)             return std::string{"not classified"};
            if(value == sign_type::STOP)                return std::string{"stop"};
            if(value == sign_type::RIGHT_CURVE)         return std::string{"right curve"};
            if(value == sign_type::LEFT_CURVE)          return std::string{"left curve"};
            if(value == sign_type::RIGHT_REVERSE_BEND)  return std::string{"right reverse bend"};
            if(value == sign_type::LEFT_REVERSE_BEND)   return std::string{"left reverse bend"};
            if(value == sign_type::STEEP_ASCENT)        return std::string{"steep ascent"};
            if(value == sign_type::STEEP_DESCENDT)      return std::string{"steep descendt"};
            if(value == sign_type::NARROW_ROAD)         return std::string{"narrow road"};
            if(value == sign_type::SLIPPERY_ROAD)       return std::string{"slippery road"};
            if(value == sign_type::MAX_VELOCITY)        return std::string{"max velocity"};
            if(value == sign_type::MIN_VELOCITY)        return std::string{"min velocity"};
            if(value == sign_type::PESESTRIAN_CROSSING) return std::string{"pedestrian crossing"};
            if(value == sign_type::AHEAD_ONLY)          return std::string{"ahead only"};
            if(value == sign_type::TURN_LEFT)           return std::string{"turn left"};
            if(value == sign_type::TURN_RIGHT)          return std::string{"turn right"};
            if(value == sign_type::DEAD_STREET)         return std::string{"dead street"};
            if(value == sign_type::PRIORITY_IN_VEH)     return std::string{"priority incoming vehicle"};
            if(value == sign_type::PARKING)             return std::string{"parking"};
            if(value == sign_type::SCHOOL_CROSSING)     return std::string{"school crossing"};
            if(value == sign_type::NO_WAITING)          return std::string{"no waiting"};
            if(value == sign_type::NO_STOPPING)         return std::string{"no stopping"};
            if(value == sign_type::NO_ENTRY)            return std::string{"no entry"};
            return std::string{"type error"};
        }
		sign_type& operator=(const Value & s){
			this->value = s;
			return *this;
		}


        bool operator!=(sign_type::Value v) noexcept {
            return v != value;
        }

        bool operator==(sign_type::Value v) noexcept {
            return v == value;
        }

        void operator=(sign_type::Value v) noexcept {
            value = v;
        }

    public:
        sign_type::Value value;
};

/**
 * @brief sem status
 */
class semaphore_status{
    public:
        //Setted like masa-protocol
        enum Value : uint8_t{
        NOT_CLS     = 0,
        RED         = 3,
        YELLOW      = 2,
        GREEN       = 1,
        BLINK       = 4};

        /**
         * @brief   method for convert id to semaphore status string name
         */
        std::string toString(){
            if(value == semaphore_status::NOT_CLS)  return std::string{"not classified"};
            if(value == semaphore_status::RED)      return std::string{"red"};
            if(value == semaphore_status::YELLOW)   return std::string{"yellow"};
            if(value == semaphore_status::GREEN)    return std::string{"green"};
            if(value == semaphore_status::BLINK)    return std::string{"blink"};
            return std::string{"type error"};
        }

		semaphore_status& operator=(const Value & s){
			this->value = s;
			return *this;
		}
    
    public:
        semaphore_status::Value value;
};

/**
 * @brief boundary point type
 */
class boundary_type {
public:
	enum Value : uint8_t{
		OTHER     	= 0,
		CURB      	= 1,
		VEHICLE		= 2,
		PERSON		= 3,
		UNDEFINED	= 4};

	/**
	 * @brief   method for convert id to semaphore status string name
	 */
	std::string toString(){
		if(value == boundary_type::OTHER)     	return std::string{"other"};
		if(value == boundary_type::CURB)   		return std::string{"curb"};
		if(value == boundary_type::VEHICLE)    	return std::string{"vehicle"};
		if(value == boundary_type::PERSON)    	return std::string{"person"};
		if(value == boundary_type::UNDEFINED)   return std::string{"undefined"};
		return std::string{"type error"};
	}

	boundary_type& operator=(const Value & s){
		this->value = s;
		return *this;
	}

private:
	boundary_type::Value value;
};

/**
 * @brief generic perception class used for polymorphism
 */
class generic : public tk::gui::Drawable{
    protected:
        type classtype;

    public:
		int sensorID = 0;
		double confidence;

        bool init(){
            classtype = type::NOT_SET;
            return true;
        }
        type getType(){
            return classtype;
        }
};

/**
 * @brief single lane data
 */
class lane2D : public generic{
    public:
        std::vector<tk::common::Vector2<float>> points;
        lane_type                          laneType;
    
    public:
        bool init(){
            classtype = type::LANE;
            return true;
        }
        lane2D& operator=(const lane2D& s){

            this->points    = s.points;
            this->laneType  = s.laneType;
            return *this;
        }
		void draw2D(int width, int height, float xLim, float yLim){
			float w,h;

			glPushMatrix(); {
				tk::gui::Viewer::tkViewportImage(width, height, xLim, yLim, sensorID, w, h);

				glTranslatef(-0.5, 0.5, 0);
				glScalef(1.0f/tk::gui::Viewer::image_width, -1.0f/tk::gui::Viewer::image_height, 1);

				glColor4f(0,0,1,1);
				for(int i = 0; i < points.size()-1; i++){
					tk:gui::Viewer::tkDrawLine(
						tk::common::Vector3<float>(points[i].x, points[i].y, 0),
						tk::common::Vector3<float>(points[i+1].x, points[i+1].y, 0)
				);
				}

			} glPopMatrix();
		}
};

/**
 * @brief single 2D box data
 */
class box2D {
	public:
		int x=0, y=0, w=0, h=0;
		box2D& operator=(const box2D& s){

			this->x   = s.x;
			this->y   = s.y;
			this->w   = s.w;
			this->h   = s.h;
			return *this;
		}
};

/**
 * @brief detected 2D object
 */
class object2D : public generic{
    public:
        box2D 		box;
        obj_class   objType;
        float       orientation;

        bool init(){
            classtype = type::BOX2D;
            return true;
        }
		object2D& operator=(const object2D& s){

            this->box 		= s.box;
            this->objType   = s.objType;
            return *this;
        }

        void draw2D(int width, int height, float xLim, float yLim){
			float w,h;

			glPushMatrix(); {
				tk::gui::Viewer::tkViewportImage(width, height, xLim, yLim, sensorID, w, h);

				glTranslatef(-0.5, 0.5, 0);
				glScalef(1.0f/tk::gui::Viewer::image_width, -1.0f/tk::gui::Viewer::image_height, 1);

				glColor4f(0,1,0,1);
				tk::gui::Viewer::tkDrawRectangle(
						tk::common::Vector3<float>( (float)box.x + (float)box.w/2, (float)box.y + (float)box.h/2, 0),
						tk::common::Vector3<float>( (float)box.w, (float)box.h, 0),
						false
						);

			} glPopMatrix();
        }
};

/**
 * @brief boundary data
 */
class boundary : public generic{
	public:
		std::vector<tk::common::Vector3<float>> points;
		std::vector<boundary_type> types;

		bool init(){
			classtype = type::BOUNDARY;
		}

		boundary& operator=(const boundary& s){

			this->points  = s.points;
			this->types   = s.types;
			return *this;
		}

		void draw2D(int width, int height, float xLim, float yLim){
			float w,h;

			glPushMatrix(); {
				tk::gui::Viewer::tkViewportImage(width, height, xLim, yLim, sensorID, w, h);

				glTranslatef(-0.5, 0.5, 0);
				glScalef(1.0f/tk::gui::Viewer::image_width, -1.0f/tk::gui::Viewer::image_height, 1);

				glColor4f(1,0,0,1);
				for(int i = 0; i < points.size()-1; i++){
					tk:gui::Viewer::tkDrawLine(
							tk::common::Vector3<float>(points[i].x, points[i].y, points[i].z),
							tk::common::Vector3<float>(points[i+1].x, points[i+1].y, points[i+1].z)
							);
				}

			} glPopMatrix();
		}
};

/**
 * @brief single 2D rotated box data
 */
class rotatedBox2D : public generic{
    public:
        tk::common::Vector2<float>  pos;
        tk::common::Vector2<float>  dim;
        tk::common::Vector2<float>  rot;
        obj_class                   objType;
    
    public:
        bool init(){
            classtype = type::RT2DBOX;
            return true;
        }
        rotatedBox2D& operator=(const rotatedBox2D& s){

            this->pos       = s.pos;
            this->dim       = s.dim;
            this->rot       = s.rot;
            this->objType   = s.objType;
            return *this;
        }
};

/**
 * @brief single 3D box data
 */
class box3D : public generic{
    public:
        tk::common::Vector3<float>  pos;
        tk::common::Vector2<float>  dim;
		obj_class                    objType;
    
    public:
        bool init(){
            classtype = type::BOX3D;
            return true;
        }
        box3D& operator=(const box3D& s){

            this->pos       = s.pos;
            this->dim       = s.dim;
            this->objType   = s.objType;
            return *this;
        }
};

/**
 * @brief single 3D rotated box data
 */
class rotatedBox3D : public generic{
    public:
        tk::common::Vector3<float>  pos;
        tk::common::Vector3<float>  dim;
        tk::common::Vector3<float>  rot;
		obj_class                    objType;
    
    public:
        bool init(){
            classtype = type::RT3DBOX;
            return true;
        }
        rotatedBox3D& operator=(const rotatedBox3D& s){

            this->pos       = s.pos;
            this->dim       = s.dim;
            this->rot       = s.rot;
            this->objType   = s.objType;
            return *this;
        }
};

/**
 * @brief single road sign data
 */
class roadSing : public generic{
    public:
        tk::common::Vector3<float>  pos;
        sign_type                   sign;
        semaphore_status            semStatus;
        int                         value;
    
    public:
        bool init(){
            classtype = type::SIGN;
            return true;
        }
        roadSing& operator=(const roadSing& s){

            this->pos       = s.pos;
            this->sign      = s.sign;
            this->semStatus = s.semStatus;
            this->value     = s.value;
            return *this;
        }
};

/**
 * @brief generic perception pointer vector
 */
typedef std::vector<generic*> genericsdata;

/**
 * @brief lane vector
 */
typedef std::vector<lane2D> lanesData;

/**
 * @brief 2D box vector
 */
typedef std::vector<box2D> box2DsData;

/**
 * @brief 2D rotated box vector
 */
typedef std::vector<rotatedBox2D> rotatedBox2DsData;

/**
 * @brief 3D box vector
 */
typedef std::vector<box3D> box3DsData;

/**
 * @brief 3D rotated box vector
 */
typedef std::vector<rotatedBox3D> rotatedBox3DsData;

/**
 * @brief road sign vector
 */
typedef std::vector<roadSing> roadSingsData;


class perceptionData : public tk::data::SensorData{
    public:

        std::vector<rotatedBox3D>   boxes;
        std::vector<object2D>       camera_objects;
        std::vector<roadSing>       signs;
        std::vector<lane2D>         camera_lanes;

        void init() override {
            tk::data::SensorData::init();
            header.sensorID = tk::data::sensorName::PERCEPTION;
        }

        void release() override {}

        bool checkDimension(SensorData *s) override {
            return true;//TODO
        }

        perceptionData& operator=(const perceptionData &s) {
            SensorData::operator=(s);

            this->boxes  = s.boxes;
            this->signs = s.signs;
            this->camera_lanes = s.camera_lanes;
            this->camera_objects = s.camera_objects;

            return *this;
        }

		void draw2D(int width, int height, float xLim, float yLim){
        	for(int i = 0; i < camera_objects.size(); i++){
        		camera_objects[i].draw2D(width, height, xLim, yLim);
        	}
			for(int i = 0; i < camera_lanes.size(); i++){
				camera_lanes[i].draw2D(width, height, xLim, yLim);
			}
        }

		void draw(){
			for(int i = 0; i < boxes.size(); i++){
				glPushMatrix();
				{
					glColor4f(0,1,0,0.5);
					glTranslatef(boxes[i].pos.x, boxes[i].pos.y, boxes[i].pos.z);
					glRotatef(boxes[i].rot.z, 0,0,1);
					glRotatef(boxes[i].rot.y, 0,1,0);
					glRotatef(boxes[i].rot.x, 1,0,0);
					tk::gui::Viewer::tkDrawCube(tk::common::Vector3<float>(0,0,0), boxes[i].dim);
				}
				glPopMatrix();
			}
		}
};


}
}


//////////////////////////////////////////////////older

namespace tk{namespace data{

struct ObjectData2D_t{
    tk::common::Rect<float>    box;
    tk::common::Vector4<float> color;
    std::string label;

    ObjectData2D_t &operator=(const ObjectData2D_t& s){
        box = s.box;
        color = s.color;
        label = s.label;
    }
};

struct ObjectData3D_t{
    tk::common::Vector3<float> pose;
    tk::common::Vector3<float> size;
    tk::common::Vector4<float> color;
    std::string label;

    ObjectData3D_t &operator=(const ObjectData3D_t& s){
        pose = s.pose;
        size = s.size;
        color = s.color;
        label = s.label;
    }
};

template <class T>
struct LineData_t{
    std::vector<T> points;
    tk::common::Vector4<float> color;

    LineData_t &operator=(const LineData_t& s){
        points = s.points;
        color = s.color;
    }
    void push(T point){
        points.push_back(point);
    }
    void pop(){
        points.pop_back();
    }
};

template <class T>
struct BoundaryData_t{
    std::vector<T> points;
    std::vector<tk::common::Vector4<float>> color;

    BoundaryData_t &operator=(const BoundaryData_t& s){
        points = s.points;
        color = s.color;
    }
    void push(T point, tk::common::Vector4<float> col){
        points.push_back(point);
        color.push_back(col);
    }
    void pop(){
        points.pop_back();
        color.pop_back();
    }
};

typedef LineData_t<tk::common::Vector2<float>> LineData2D_t;
typedef LineData_t<tk::common::Vector3<float>> LineData3D_t;

typedef BoundaryData_t<tk::common::Vector2<float>> BoundaryData2D_t;
typedef BoundaryData_t<tk::common::Vector3<float>> BoundaryData3D_t;

class LinesData_t : public tk::data::SensorData {
public:
    std::vector<LineData3D_t> data;

        void init() {
            SensorData::init();
            header.sensor = tk::data::sensorName::LINES;
        }

        void release(){
            return;
        }

        bool checkDimension(SensorData *s){
            return true;
        }

    matvar_t *toMatVar(std::string name = "lines") {
        size_t dims[] = {data.size(), 1};
        matvar_t *array = Mat_VarCreate(name.c_str(),MAT_C_CELL,MAT_T_CELL,2,dims,NULL,0);

        for(int i=0; i<data.size(); i++) {
            Eigen::MatrixXf p(3, data[i].points.size());
            for(int j=0; j<p.cols(); j++) {
                p(0, j) = data[i].points[j].x;
                p(1, j) = data[i].points[j].y;
                p(2, j) = data[i].points[j].z;
            }
            matvar_t *var = tk::common::eigenXf2matvar(p, "line");
            Mat_VarSetCell(array, i,var);
        }
        return array;
    }

    bool fromMatVar(matvar_t *var) {

        int n = var->dims[0];
        for(int i=0; i<n; i++) {
            matvar_t *pvar = Mat_VarGetCell(var, i);
            Eigen::MatrixXf mat = tk::common::matvar2eigenXf(pvar);

            LineData3D_t line;
            for(int j=0; j<mat.cols(); j++) {
                tk::common::Vector3<float> p;
                p.x = mat(0, j);
                p.y = mat(1, j);
                p.z = mat(2, j);
                line.points.push_back(p);
            }
            data.push_back(line);
        }
        return true;
    }
};


}}
