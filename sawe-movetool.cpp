#include "sawe-movetool.h"

namespace Sawe {

NavigationTool::NavigationTool(DisplayWidget *dw): BasicTool(dw)
{
    
}
    
void NavigationTool::render()
{
}
QWidget *NavigationTool::getSettingsWidget()
{
}

void NavigationTool::mousePressEvent(QMouseEvent * e)
{
    if( (e->button() & Qt::LeftButton) == Qt::LeftButton)
        moveButton.press( e->x(), this->height() - e->y() );

    if( (e->button() & Qt::RightButton) == Qt::RightButton)
        rotateButton.press( e->x(), this->height() - e->y() );
        
    displayWidget->update();
}
void NavigationTool::mouseMoveEvent(QMouseEvent * e)
{
    float rs = 0.2;
    int x = e->x(), y = this->height() - e->y();

    float rx, ry, rz;
    if( rotateButton.isDown() ){
        displayWidget->getrxyz(rx, ry, rz);
        //Controlling the rotation with the left button.
        ry += (1-displayWidget->orthoview) * rs * rotateButton.deltaX( x );
        rx -= rs * rotateButton.deltaY( y );
        if (rx<10) rx=10;
        if (rx>90) { rx=90; displayWidget->orthoview=1; }
        if (0<displayWidget->orthoview && rx<90) { rx=90; displayWidget->orthoview=0; }
        
        displayWidget->setrxyz(rx, ry, rz);
        
    }
    double qx, qy, qz;
    if( moveButton.isDown() )
    {
        displayWidget->getqxyz(qx, qy, qz);
        //Controlling the position with the right button.
        GLvector last, current;
        if( moveButton.worldPos(last[0], last[1], displayWidget->xscale, displayWidget) &&
            displayWidget->worldPos(x, y, current[0], current[1], displayWidget->xscale) )
        {
            float l = displayWidget->worker()->source()->length();
            
            qx -= current[0] - last[0];
            qz -= current[1] - last[1];
            
            if (qx<0) qx=0;
            if (qz<0) qz=0;
            if (qz>1) qz=1;
            if (qx>l) qx=l;
        }
        
        displayWidget->setqxyz(qx, qy, qz);
    }
    rotateButton.update(x, y);
    moveButton.update(x, y);
    displayWidget->worker()->requested_fps(30);
    displayWidget->update();
}
void NavigationTool::mouseReleaseEvent(QMouseEvent * e)
{
    switch ( e->button() )
    {
        case Qt::LeftButton:
            moveButton.release();
            break;

        case Qt::RightButton:
            rotateButton.release();
            break;
            
        default:
            break;
    }
    displayWidget->update();
}

};