#ifndef _INFOTOOL_H_
#define _INFOTOOL_H_

#include "sawe-basictool.h"
#include "sawe-movetool.h"

namespace Sawe {

class InfoTool: public BasicTool
{
public:
    InfoTool(DisplayWidget *dw):BasicTool(dw){usingInfo = false;}

protected:
    bool usingInfo;
    
    void mousePressEvent(QMouseEvent * e);
    void mouseMoveEvent(QMouseEvent * e);
    void mouseReleaseEvent(QMouseEvent * e);
    
};


class InfoToolWithNavigation: public BasicTool
{
public:
    InfoToolWithNavigation(DisplayWidget *dw):BasicTool(dw)
    {
        MotionTool *motionTool = new MotionTool(dw);
        RotationTool *rotationTool = new RotationTool(dw);
        ZoomTool *zoomTool = new ZoomTool(dw);
        InfoTool *infoTool = new InfoTool(dw);
        
        zoomTool->push(infoTool);
        rotationTool->push(zoomTool);
        motionTool->push(rotationTool);
        push(motionTool);
    }
};

};

#endif