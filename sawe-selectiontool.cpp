#include "sawe-selectiontool.h"

void drawBlock(float z1, float x1, float z2, float x2)
{
    glBegin(GL_QUADS);
    glVertex3f(x1, 1.0, z1);
    glVertex3f(x1, 1.0, z2);
    glVertex3f(x1, 0.0, z2);
    glVertex3f(x1, 0.0, z1);
    
    glVertex3f(x1, 1.0, z2);
    glVertex3f(x2, 1.0, z2);
    glVertex3f(x2, 0.0, z2);
    glVertex3f(x1, 0.0, z2);
    
    glVertex3f(x2, 1.0, z2);
    glVertex3f(x2, 1.0, z1);
    glVertex3f(x2, 0.0, z1);
    glVertex3f(x2, 0.0, z2);
    
    glVertex3f(x2, 1.0, z1);
    glVertex3f(x1, 1.0, z1);
    glVertex3f(x1, 0.0, z1);
    glVertex3f(x2, 0.0, z1);
    glEnd();
    
    
    glLineWidth(2.0f);
    glBegin(GL_LINES);
    glVertex3f(x1, 1.0, z1);
    glVertex3f(x1, 1.0, z2);
    
    glVertex3f(x1, 1.0, z2);
    glVertex3f(x2, 1.0, z2);
    
    glVertex3f(x2, 1.0, z2);
    glVertex3f(x2, 1.0, z1);
    
    glVertex3f(x2, 1.0, z1);
    glVertex3f(x1, 1.0, z1);
    glEnd();
    glLineWidth(0.5f);
}

namespace Sawe{

void SelectionTool::render()
{
    if(selection)
        drawBlock(start.z, start.x, end.z, end.x);
}

void SelectionTool::mousePressEvent(QMouseEvent * e)
{
    printf("SelectionTool\n");
    int x = e->x(), y = this->height() - e->y();

    GLvector current;
    if( (e->button() & Qt::LeftButton) == Qt::LeftButton &&
        displayWidget->worldPos(x, y, current[0], current[1], displayWidget->xscale) )
    {
        making_selection = true;
        selection = true;
        start.x = current[0];
        start.z = current[1];
        end.x = current[0];
        end.z = current[1];
        
        displayWidget->update();
    }
    else
        e->ignore();
}

void SelectionTool::mouseMoveEvent(QMouseEvent * e)
{
    int x = e->x(), y = this->height() - e->y();

    GLvector current;
    if(making_selection && displayWidget->worldPos(x, y, current[0], current[1], displayWidget->xscale) )
    {
        end.x = current[0];
        end.z = current[1];
        
        displayWidget->setSelection(start, end, true);
        displayWidget->update();
    }
    else
        e->ignore();
}

void SelectionTool::mouseReleaseEvent(QMouseEvent * e)
{
    making_selection = false;
    if(start.x - end.x == 0 && start.z - end.z == 0)
    {
        selection = false;
    }
    else
    {
        displayWidget->setSelection(start, end, true);
        e->ignore();
    }
}

};