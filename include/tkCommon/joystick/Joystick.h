#include "gamepad/gamepad.h"
#include <iostream>

class Joystick{

    public:

        bool init();

        void update();
        
        void getStickLeft(float& x, float& y);

        void getStickRight(float& x, float& y);

        bool getButtonPressed(GAMEPAD_BUTTON button);

        void getTriggerLeft(float& x);

        void getTriggerRight(float& x);

};