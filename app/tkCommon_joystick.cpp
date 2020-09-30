#include "tkCommon/joystick/Joystick.h"
#include <iomanip>

int main(){

	Joystick a;
	bool run = a.init();

	while (run)
	{
		a.update();

		float x,y,x1,y1,xLeft,xRight;
		bool pressedA;

		a.getStickLeft(x,y);
		a.getStickRight(x1,y1);
		a.getTriggerLeft(xLeft);
		a.getTriggerRight(xRight);
		pressedA = a.getButtonPressed(BUTTON_A);
		std::cout<<std::setprecision(3)<<std::fixed<<"left("<<x<<";"<<y<<") right:("<<x1<<";"<<y1<<") buttonA:"<<pressedA<<" left:"<<xLeft<<" right:"<<xRight<<std::endl;
	}

	return 0;
}