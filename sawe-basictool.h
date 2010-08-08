#include <QWidget>

namespace Sawe {
class BasicTool: public QWidget
{
public:
    BasicTool()
    {
        setSizePolicy(QSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding));
        setAutoFillBackground(false);
        setMask(QRegion(0, 0, 1, 1, QRegion::Rectangle));
    
        setAttribute(Qt::WA_MouseNoMask, true);
    }


    virtual void mousePressEvent(QMouseEvent *event)
    {
        printf("lala instinct!\n");
    }

    virtual void tabletEvent ( QTabletEvent * event )
    {
        printf("I'm a tablet!\n");
    }
    
    virtual void resizeEvent ( QResizeEvent * event )
    {
        printf("REsizing!\n");
    }
};
};