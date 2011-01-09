#ifndef NAVIGATIONCONTROLLER_H
#define NAVIGATIONCONTROLLER_H

#include "ui/mousecontrol.h"
#include "ui/comboboxaction.h"

#include <QWidget>
#include <QPointer>

namespace Tools
{
    class RenderView;

    class NavigationController: public QWidget
    {
        Q_OBJECT
    public:
        NavigationController(RenderView* view);
        ~NavigationController();

    signals:
        void enabledChanged(bool active);

    private slots:
        void receiveToggleNavigation(bool active);
        void receiveToggleZoom(bool active);


    private:
        // Event handlers
        virtual void mousePressEvent ( QMouseEvent * e );
        virtual void mouseReleaseEvent ( QMouseEvent * e );
        virtual void wheelEvent ( QWheelEvent *event );
        virtual void mouseMoveEvent ( QMouseEvent * e );
        virtual void changeEvent(QEvent *);

        // View
        // View that is controlled, this controller doesn't have a model
        // and shares control of the renderview with rendercontoller
        RenderView* _view;

        // GUI
        void connectGui();

        // State
        bool zoom_only_;
        Ui::MouseControl moveButton;
        Ui::MouseControl rotateButton;
        Ui::MouseControl scaleButton;
        QScopedPointer<Ui::ComboBoxAction> one_action_at_a_time_;

        void zoom(int delta, bool xscale);
    };
} // namespace Tools

#endif // NAVIGATIONCONTROLLER_H
