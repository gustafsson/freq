#ifndef _SELECTIONTOOL_H_
#define _SELECTIONTOOL_H_

#include "sawe-basictool.h"

namespace Sawe {

class SelectionTool: public BasicTool
{
public:
    SelectionTool(DisplayWidget *dw):BasicTool(dw){selection = false; making_selection = false;}

protected:
    MyVector start, end;
    bool selection, making_selection;

    void render();
    void mousePressEvent(QMouseEvent * e);
    void mouseMoveEvent(QMouseEvent * e);
    void mouseReleaseEvent(QMouseEvent * e);
    
};


};
#endif