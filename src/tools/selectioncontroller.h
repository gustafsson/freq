#ifndef SELECTIONCONTROLLER_H
#define SELECTIONCONTROLLER_H

#include "selectionmodel.h"

#include "ui/mousecontrol.h"
#include "ui/comboboxaction.h"
#include "tools/support/toolselector.h"

#include <QWidget>
#include <QPointer>

#include <boost/scoped_ptr.hpp>

namespace Sawe { class ToolModel; }
namespace Signal { class Worker; }

namespace Tools
{
    class ToolModel;
    class RenderView;

    namespace Selections {
        class EllipseModel;
        class EllipseView;
        class EllipseController;

        class PeakModel;
        class PeakView;
        class PeakController;

        class SplineModel;
        class SplineView;
        class SplineController;

        class RectangleModel;
        class RectangleView;
        class RectangleController;
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
        void setCurrentSelectionCommand( Signal::pOperation selection );
        void setCurrentSelection( Signal::pOperation filter );

    signals:
        void enabledChanged(bool active);

    public slots:
        // Action slots
        void setThisAsCurrentTool( bool active );
        void onSelectionChanged();
        //void tryHeadAsSelection();

    private slots:
        // Action slots
        //void receiveAddSelection();
        void receiveAddClearSelection();

        void receiveCropSelection();
        //void receiveMoveSelection(bool);
        //void receiveMoveSelectionInTime(bool);

        // todo move somewhere else
        void receiveCurrentSelection(int, bool);
        void receiveFilterRemoval(int);

        void selectionComboBoxToggled();

        void renderModelChanged(Tools::ToolModel* model);

        void deselect();

    private:
        // Event handlers
        virtual void changeEvent ( QEvent * event );
        virtual void mousePressEvent ( QMouseEvent * e );

        // View
        SelectionModel* _model;
        RenderView* _render_view;
        //Signal::Worker* _worker;

        // GUI
        void setupGui();
        void toolfactory();
        ::Ui::ComboBoxAction * selectionComboBox_;
        boost::scoped_ptr<Support::ToolSelector> tool_selector_;
        QAction* deselect_action_;

        // State
        bool selecting;

        //void setSelection(int i, bool enabled);
        //void removeFilter(int i);

        // Tools
        QScopedPointer<Selections::EllipseModel> ellipse_model_;
        QScopedPointer<Selections::EllipseView> ellipse_view_;
        QPointer<Selections::EllipseController> ellipse_controller_;

        QScopedPointer<Selections::PeakModel> peak_model_;
        QScopedPointer<Selections::PeakView> peak_view_;
        QPointer<Selections::PeakController> peak_controller_;

        QScopedPointer<Selections::SplineModel> spline_model_;
        QScopedPointer<Selections::SplineView> spline_view_;
        QPointer<Selections::SplineController> spline_controller_;

        QScopedPointer<Selections::RectangleModel> rectangle_model_;
        QScopedPointer<Selections::RectangleView> rectangle_view_;
        QPointer<Selections::RectangleController> rectangle_controller_;
    };
}

#endif // SELECTIONCONTROLLER_H
