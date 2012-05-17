#ifndef _BRUSHTOOL_H_
#define _BRUSHTOOL_H_

#include "sawe-basictool.h"
#include <vector.h>

namespace Sawe {

struct BrushPoint
{
    BrushPoint(double x, double y, double size){this->x = x; this->y = y; this->size = size;}
    double x, y, size;
};

class BrushTool: public BasicTool
{
public:
    BrushTool(DisplayWidget *dw);
    
    virtual void render();
    virtual QWidget *getSettingsWidget();

protected:
    vector<BrushPoint> stroke;
    bool isPainting;
    
    void tabletEvent(QTabletEvent *event);
    void mousePressEvent(QMouseEvent * e);
    void mouseMoveEvent(QMouseEvent * e);
    void mouseReleaseEvent(QMouseEvent * e);
    
};


};

#endif