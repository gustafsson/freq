#include "sawe-movetool.h"
#include <QMouseEvent>

namespace Sawe {

////////////////////////////////////////////////////////////////////////////////
// NavigationTool
////////////////////////////////////////////////////////////////////////////////
NavigationTool::NavigationTool(DisplayWidget *dw): BasicTool(dw)
{
    MotionTool *motionTool = new MotionTool(dw);
    RotationTool *rotationTool = new RotationTool(dw);
    ZoomTool *zoomTool = new ZoomTool(dw);
    rotationTool->push(zoomTool);
    motionTool->push(rotationTool);
    push(motionTool);
}

////////////////////////////////////////////////////////////////////////////////
// MotionTool
////////////////////////////////////////////////////////////////////////////////
void MotionTool::mousePressEvent(QMouseEvent * e)
{
    printf("MotionTool\n");
    if( (e->button() & Qt::LeftButton) == Qt::LeftButton)
        moveButton.press( e->x(), this->height() - e->y() );
    else
        e->ignore();
}

void MotionTool::mouseMoveEvent(QMouseEvent * e)
{
    int x = e->x(), y = this->height() - e->y();

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
        moveButton.update(x, y);
    
        displayWidget->worker()->requested_fps(30);
        displayWidget->update();
    }
    else
    {
        e->ignore();
    }
}

void MotionTool::mouseReleaseEvent(QMouseEvent * e)
{
    if(e->button() == Qt::LeftButton)
    {
        moveButton.release();
    }
    else
    {
        e->ignore();
    }
}


////////////////////////////////////////////////////////////////////////////////
// RotationTool
////////////////////////////////////////////////////////////////////////////////
void RotationTool::mousePressEvent(QMouseEvent * e)
{
    printf("RotationTool\n");
    if( (e->button() & Qt::RightButton) == Qt::RightButton)
        rotateButton.press( e->x(), this->height() - e->y() );
    else
        e->ignore();
}

void RotationTool::mouseMoveEvent(QMouseEvent * e)
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
        rotateButton.update(x, y);
        
        displayWidget->worker()->requested_fps(30);
        displayWidget->update();
    }
    else
    {
        e->ignore();
    }
}

void RotationTool::mouseReleaseEvent(QMouseEvent * e)
{
    if(e->button() == Qt::RightButton)
    {
        rotateButton.release();
    }
    else
    {
        e->ignore();
    }
}


////////////////////////////////////////////////////////////////////////////////
// ZoomTool
////////////////////////////////////////////////////////////////////////////////
void ZoomTool::wheelEvent(QWheelEvent *e)
{
    printf("ZoomTool\n");
    float ps = 0.0005;
    float rs = 0.08;
    
    if( e->orientation() == Qt::Horizontal )
    {
        float rx, ry, rz;
        displayWidget->getrxyz(rx, ry, rz);
        
        if(e->modifiers().testFlag(Qt::ShiftModifier))
            displayWidget->xscale *= (1-ps * e->delta());
        else
            ry -= rs * e->delta();
            
        displayWidget->setrxyz(rx, ry, rz);
        displayWidget->update();
    }
    else
    {
        float px, py, pz;
        displayWidget->getpxyz(px, py, pz);
        
		if(e->modifiers().testFlag(Qt::ShiftModifier))
            displayWidget->xscale *= (1-ps * e->delta());
        else
	        pz *= (1+ps * e->delta());

        if (pz<-40) pz = -40;
        if (pz>-.4) pz = -.4;
        
        displayWidget->setpxyz(px, py, pz);
        displayWidget->update();
    }
    
    update();
}

};