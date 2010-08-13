#include "sawe-basictool.h"

namespace Sawe {

class NavigationTool: public BasicTool
{
public:
    NavigationTool(DisplayWidget *dw);
};

class MotionTool: public BasicTool
{
public:
    MotionTool(DisplayWidget *dw):BasicTool(dw){}

protected:
    MouseControl moveButton;

    void mousePressEvent(QMouseEvent * e);
    void mouseMoveEvent(QMouseEvent * e);
    void mouseReleaseEvent(QMouseEvent * e);
    
};

class RotationTool: public BasicTool
{
public:
    RotationTool(DisplayWidget *dw):BasicTool(dw){}

protected:
    MouseControl rotateButton;

    void mousePressEvent(QMouseEvent * e);
    void mouseMoveEvent(QMouseEvent * e);
    void mouseReleaseEvent(QMouseEvent * e);
    
};

class ZoomTool: public BasicTool
{
public:
    ZoomTool(DisplayWidget *dw):BasicTool(dw){}

protected:
    void wheelEvent(QWheelEvent *e);
    
};


};