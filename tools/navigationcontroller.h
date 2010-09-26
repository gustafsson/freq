#ifndef NAVIGATIONCONTROLLER_H
#define NAVIGATIONCONTROLLER_H

#include "ui/mousecontrol.h"

#include <QWidget>

namespace Tools
{
    class RenderView;

    class NavigationController: public QWidget
    {
        Q_OBJECT
    public:
        NavigationController(RenderView* view);

    private slots:
        void receiveToggleNavigation(bool active);

    private:
        // Event handlers
        virtual void mousePressEvent ( QMouseEvent * e );
        virtual void mouseReleaseEvent ( QMouseEvent * e );
        virtual void wheelEvent ( QWheelEvent *event );
        virtual void mouseMoveEvent ( QMouseEvent * e );

        // View
        // View that is controlled, this controller doesn't have a model
        // and shares control of the renderview with rendercontoller
        RenderView* _view;

        // GUI
        void setupGui();

        // State
        Ui::MouseControl moveButton;
        Ui::MouseControl rotateButton;
        Ui::MouseControl scaleButton;
    };
} // namespace Tools

#endif // NAVIGATIONCONTROLLER_H
