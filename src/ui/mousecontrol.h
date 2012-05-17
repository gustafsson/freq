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

    float getLastx() { return lastx; }
    float getLasty() { return lasty; }
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
