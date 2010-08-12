#include "sawe-basictool.h"

namespace Sawe {

class NavigationTool: public BasicTool
{
public:
    NavigationTool(DisplayWidget *dw);
    
    virtual void render();
    virtual QWidget *getSettingsWidget();

protected:
    void mousePressEvent(QMouseEvent * e);
    void mouseMoveEvent(QMouseEvent * e);
    void mouseReleaseEvent(QMouseEvent * e);
    
};


};