#ifndef BRUSHCONTROLLER_H
#define BRUSHCONTROLLER_H

#include "brushview.h"
#include "brushmodel.h"

#include "ui/mousecontrol.h"
#include <QWidget>

namespace Tools {

    class RenderView;
    class BrushView;

    class BrushController: public QWidget
    {
        Q_OBJECT
    public:
        BrushController( BrushView* view, RenderView* render_view );
        ~BrushController();

    signals:
        void enabledChanged(bool active);

    private slots:
        // Action slots
        void receiveToggleBrush(bool active);

    private:
        // Event handlers
        virtual void mousePressEvent ( QMouseEvent * e );
        virtual void mouseReleaseEvent ( QMouseEvent * e );
        virtual void mouseMoveEvent ( QMouseEvent * e );
        virtual void changeEvent ( QEvent * event );

        // View
        BrushModel* model() { return view_->model; }
        BrushView* view_;
        RenderView* render_view_;
        Signal::Worker* worker_;

        // GUI
        void setupGui();

        // State
        bool drawing_;
        Ui::MouseControl draw_button_;
    };

} // namespace Tools

#endif // BRUSHCONTROLLER_H
