#include "sawe-basictool.h"

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


};