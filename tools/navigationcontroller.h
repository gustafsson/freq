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

        void moveUp();
        void moveDown();
        void moveLeft();
        void moveRight();
        void scaleUp();
        void scaleDown();
        void scaleLeft();
        void scaleRight();


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

        enum ZoomMode {
            Zoom,
            ScaleX,
            ScaleZ
        };
        void zoom(int delta, ZoomMode mode);
        void moveCamera( float dt, float ds );
        void zoomCamera( float dt, float ds, float dz );
        void rotateCamera( float dt, float ds );
        void bindKeyToSlot( QWidget* owner, const char* keySequence, const QObject* receiver, const char* slot );

	};
} // namespace Tools

#endif // NAVIGATIONCONTROLLER_H
