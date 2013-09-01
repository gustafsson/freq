#if 0
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
        void enabledChanged(bool v); // Will only be emitted when 'v==false'

    private slots:
        // Action slots
        void receiveToggleBrush(bool active);

    private:
        // Event handlers
        virtual void mousePressEvent ( QMouseEvent * e );
        virtual void mouseReleaseEvent ( QMouseEvent * e );
        virtual void mouseMoveEvent ( QMouseEvent * e );
        virtual void changeEvent ( QEvent * event );
        virtual void wheelEvent ( QWheelEvent *event );

        // View
        BrushModel* model() { return view_->model_; }
        BrushView* view_;
        RenderView* render_view_;

        // GUI
        void setupGui();
        Qt::MouseButton paint_button_;

        // State
        //bool drawing_;
        Signal::Intervals drawn_interval_;
        //Ui::MouseControl draw_button_;
    };

} // namespace Tools

#endif // BRUSHCONTROLLER_H
#endif
