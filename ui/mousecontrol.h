#ifndef UI_MOUSECONTROL_H
#define UI_MOUSECONTROL_H

namespace Ui
{

class MouseControl // TODO move to Tools::Support
{
private:
    float lastx;
    float lasty;
    bool down;
    unsigned int hold;

public:
    MouseControl();

    float deltaX( float x );
    float deltaY( float y );

    bool worldPos(double &ox, double &oy, float scale);
    static bool worldPos(double x, double y, double &ox, double &oy, float scale);
    static bool planePos(double x, double y, float &ox, float &oy, float scale);

    /**
      worldPos projects screen coordinates onto the xz-plane in world space.
      spacePos simply returns the space pos of screen coordinates.
      */
    bool spacePos(double &out_x, double &out_y);
    static bool spacePos(double in_x, double in_y, double &out_x, double &out_y);

    bool isDown(){return down;}
    bool isTouched();
    int getHold(){return hold;}

    void press( float x, float y );
    void update( float x, float y );
    void release();
    void touch(){hold = 0;}
    void untouch(){hold++;}
};
} // namespace Ui

#endif // UI_MOUSECONTROL_H
