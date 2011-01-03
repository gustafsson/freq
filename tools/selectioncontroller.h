#ifndef SELECTIONCONTROLLER_H
#define SELECTIONCONTROLLER_H

#include "selectionview.h"
#include "selectionmodel.h"

#include "ui/mousecontrol.h"
#include <QWidget>

namespace Signal { class Worker; }

namespace Tools
{
    class RenderView;
    class SelectionView;

    class SelectionController: public QWidget
    {
        Q_OBJECT
    public:
        SelectionController( SelectionView* view, RenderView* render_view);
        ~SelectionController();

    signals:
        void enabledChanged(bool active);

    private slots:
        // Action slots
        void receiveToggleSelection(bool active);
        void receiveAddSelection(bool);
        void receiveAddClearSelection(bool);

        void receiveCropSelection();
        void receiveMoveSelection(bool);
        void receiveMoveSelectionInTime(bool);

        // todo move somewhere else
        void receiveCurrentSelection(int, bool);
        void receiveFilterRemoval(int);

    private:
        // Event handlers
        virtual void mousePressEvent ( QMouseEvent * e );
        virtual void mouseReleaseEvent ( QMouseEvent * e );
        virtual void mouseMoveEvent ( QMouseEvent * e );
        virtual void changeEvent ( QEvent * event );

        // View
        SelectionModel* model() { return _view->model; }
        SelectionView* _view;
        RenderView* _render_view;
        Signal::Worker* _worker;

        // GUI
        void setupGui();

        // State
        bool selecting;
        Ui::MouseControl selectionButton;
        MyVector selectionStart;

        void setSelection(int i, bool enabled);
        void removeFilter(int i);
    };
}

#endif // SELECTIONCONTROLLER_H
