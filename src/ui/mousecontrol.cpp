#include "mousecontrol.h"

namespace Ui
{


MouseControl::
        MouseControl()
            :
            down( false ),
            hold( 0 )
{}


float MouseControl::
        deltaX( float x )
{
    if( down )
        return x - lastx;

    return 0;
}


float MouseControl::
        deltaY( float y )
{
    if( down )
        return y - lasty;

    return 0;
}


void MouseControl::
        press( float x, float y )
{
    update( x, y );
    down = true;
}


void MouseControl::
        update( float x, float y )
{
    touch();
    lastx = x;
    lasty = y;
}


void MouseControl::
        release()
{
    //touch();
    down = false;
}


bool MouseControl::
        isTouched()
{
    if(hold == 0)
        return true;
    else
        return false;
}

} // namespace Ui
