#ifndef SELECTIONCONTROLLER_H
#define SELECTIONCONTROLLER_H

#include "selectionmodel.h"

#include "ui/mousecontrol.h"
#include "ui/comboboxaction.h"
#include "tools/support/toolselector.h"

#include <QWidget>
#include <QPointer>

namespace Signal { class Worker; }

namespace Tools
{
    class RenderView;
    namespace Selections {
        class EllipseModel;
        class EllipseView;
        class EllipseController;

        class SquareModel;
        class SquareView;
        class SquareController;

        class PeakModel;
        class PeakView;
        class PeakController;
    }

    class SelectionController: public QWidget
    {
        Q_OBJECT
    public:
        SelectionController( SelectionModel* model, RenderView* render_view);
        ~SelectionController();


        RenderView* render_view() { return _render_view; }
        SelectionModel* model() { return _model; }

        void addComboBoxAction( QAction* action );
        void setCurrentTool( QWidget* tool, bool active );
        void setCurrentSelection( Signal::pOperation filter );

    signals:
        void enabledChanged(bool active);

    public slots:
        // Action slots
        void setThisAsCurrentTool( bool active );

    private slots:
        // Action slots
        void receiveAddSelection(bool);
        void receiveAddClearSelection(bool);

        void receiveCropSelection();
        //void receiveMoveSelection(bool);
        //void receiveMoveSelectionInTime(bool);

        // todo move somewhere else
        void receiveCurrentSelection(int, bool);
        void receiveFilterRemoval(int);

    private:
        // View
        SelectionModel* _model;
        RenderView* _render_view;
        Signal::Worker* _worker;

        // GUI
        void setupGui();
        void toolfactory();
        Ui::ComboBoxAction * selectionComboBox_;
        boost::scoped_ptr<Support::ToolSelector> tool_selector_;

        // State
        bool selecting;

        //void setSelection(int i, bool enabled);
        //void removeFilter(int i);

        // Tools
        QScopedPointer<Selections::EllipseModel> ellipse_model_;
        QScopedPointer<Selections::EllipseView> ellipse_view_;
        QPointer<Selections::EllipseController> ellipse_controller_;

        QScopedPointer<Selections::SquareModel> square_model_;
        QScopedPointer<Selections::SquareView> square_view_;
        QPointer<Selections::SquareController> square_controller_;

        QScopedPointer<Selections::PeakModel> peak_model_;
        QScopedPointer<Selections::PeakView> peak_view_;
        QPointer<Selections::PeakController> peak_controller_;
    };
}

#endif // SELECTIONCONTROLLER_H
