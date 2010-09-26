#ifndef TOOLTIPCONTROLLER_H
#define TOOLTIPCONTROLLER_H

#include "ui/mousecontrol.h"

#include <QWidget>

namespace Tools
{
    class RenderView;
    class RenderModel;

    class TooltipController: public QWidget
    {
        Q_OBJECT
    public:
        TooltipController(RenderView *view, RenderModel *model);

    private slots:
        virtual void receiveToggleInfoTool(bool);

    private:
        // Event handlers
        virtual void mousePressEvent ( QMouseEvent * e );
        virtual void mouseReleaseEvent ( QMouseEvent * e );
        virtual void mouseMoveEvent ( QMouseEvent * e );

        // Model and View
        RenderView* _view;
        RenderModel* _model;

        // GUI
        void setupGui();

        // State
        Ui::MouseControl infoToolButton;
    };
}

#endif // TOOLTIPCONTROLLER_H
