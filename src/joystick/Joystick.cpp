#include "tkCommon/joystick/Joystick.h"
#include <iostream>


bool 
Joystick::init(){
    GamepadInit();
    if( !GamepadIsConnected((GAMEPAD_DEVICE)0) ){
        std::cout<<"Joystick not connected\n";
        return false;
    }
    GamepadSetRumble((GAMEPAD_DEVICE)0, 1.0, 1.0);
    return true;
}

void
Joystick::update(){
    GamepadUpdate();
}

void 
Joystick::getStickLeft(float& x, float& y){
    int tx,ty;
    GamepadStickXY((GAMEPAD_DEVICE)0, STICK_LEFT, &tx, &ty);
    x = (float)tx/32767.0;
    y = (float)ty/32767.0;
}

void 
Joystick::getStickRight(float& x, float& y){
    int tx,ty;
    GamepadStickXY((GAMEPAD_DEVICE)0, STICK_RIGHT, &tx, &ty);
    x = (float)tx/32767.0;
    y = (float)ty/32767.0;
}

bool 
Joystick::getButtonPressed(GAMEPAD_BUTTON button){
    return GamepadButtonDown((GAMEPAD_DEVICE)0,button);
}

void 
Joystick::getTriggerLeft(float& x){
    x = GamepadTriggerLength((GAMEPAD_DEVICE)0, TRIGGER_LEFT);
}

void
Joystick::getTriggerRight(float& x){
    x = GamepadTriggerLength((GAMEPAD_DEVICE)0, TRIGGER_RIGHT);
}