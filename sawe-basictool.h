#include <QWidget>
#include <QMouseEvent>
#include <QVBoxLayout>
#include "displaywidget.h"


#ifndef SAWE_BASICTOOL
#define SAWE_BASICTOOL

namespace Sawe {

class ToolInterface
{
public:
    ToolInterface(DisplayWidget *dw)
    {
        displayWidget = dw;
    }
    
    virtual void render() = 0;
    virtual QWidget *getSettingsWidget() = 0;
    
    virtual bool mouseMoveEvent(QMouseEvent * e){return false;}
    virtual bool wheelEvent(QWheelEvent *e){return false;}
    virtual bool mouseReleaseEvent(QMouseEvent * e){return false;}
    virtual bool mousePressEvent(QMouseEvent * e){return false;}
    virtual bool tabletEvent(QTabletEvent *e){return false;}
    
    virtual int getPriority(){return 0;};

protected:
    DisplayWidget *displayWidget;

};

class BasicTool: public QWidget
{
public:
    BasicTool(DisplayWidget *dw)
    {
        setSizePolicy(QSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding));
        setAutoFillBackground(false);
        setMask(QRegion(0, 0, 1, 1, QRegion::Rectangle));
    
        setAttribute(Qt::WA_MouseNoMask, true);
        
        displayWidget = dw;
    }
    
    virtual void render() = 0;
    virtual QWidget *getSettingsWidget() = 0;
    
    void push(BasicTool *tool)
    {
	    QVBoxLayout *verticalLayout = new QVBoxLayout();
	    verticalLayout->addWidget(tool);
	    verticalLayout->setContentsMargins(0, 0, 0, 0);
	    setLayout(verticalLayout);
    }

protected:
    DisplayWidget *displayWidget;
    
};


};

#endif