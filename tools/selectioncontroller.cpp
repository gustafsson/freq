#include "selectioncontroller.h"

// Sonic AWE
#include "renderview.h"
#include "sawe/project.h"
#include "ui_mainwindow.h"
#include "ui/mainwindow.h"
#include "signal/operation-basic.h"
#include "support/operation-composite.h"
#include "filters/ellipse.h"
#include "filters/rectangle.h"
#include "tools/selections/support/splinefilter.h"

#include "selections/ellipsecontroller.h"
#include "selections/ellipsemodel.h"
#include "selections/ellipseview.h"

#include "selections/peakcontroller.h"
#include "selections/peakmodel.h"
#include "selections/peakview.h"

#include "selections/splinecontroller.h"
#include "selections/splinemodel.h"
#include "selections/splineview.h"

#include "selections/squarecontroller.h"
#include "selections/squaremodel.h"
#include "selections/squareview.h"


namespace Tools
{
    SelectionController::
            SelectionController( SelectionModel* model, RenderView* render_view )
                :
                _model(model),
                _render_view(render_view),
                _worker(&render_view->model->project()->worker),
                selectionComboBox_(0),
                tool_selector_( new Support::ToolSelector(this))
    {
        setupGui();

        setAttribute(Qt::WA_DontShowOnScreen, true);
        setEnabled( false );
    }


    SelectionController::
            ~SelectionController()
    {
        TaskTimer(__FUNCTION__).suppressTiming();
    }


    void SelectionController::
            setupGui()
    {
        Ui::SaweMainWindow* main = _model->project->mainWindow();
        Ui::MainWindow* ui = main->getItems();

        connect(ui->actionActionAdd_selection, SIGNAL(triggered(bool)), SLOT(receiveAddSelection(bool)));
        connect(ui->actionActionRemove_selection, SIGNAL(triggered(bool)), SLOT(receiveAddClearSelection(bool)));
        connect(ui->actionCropSelection, SIGNAL(triggered()), SLOT(receiveCropSelection()));
        //connect(ui->actionMoveSelection, SIGNAL(triggered(bool)), SLOT(receiveMoveSelection(bool)));
        //connect(ui->actionMoveSelectionTime, SIGNAL(triggered(bool)), SLOT(receiveMoveSelectionInTime(bool)));

        ui->actionActionAdd_selection->setEnabled( false );
        ui->actionActionRemove_selection->setEnabled( false );
        ui->actionMoveSelection->setEnabled( false );
        ui->actionCropSelection->setEnabled( false );
        ui->actionMoveSelectionTime->setEnabled( false );

        QToolBar* toolBarTool = new QToolBar(main);
        toolBarTool->setObjectName(QString::fromUtf8("toolBarTool"));
        toolBarTool->setEnabled(true);
        toolBarTool->setContextMenuPolicy(Qt::NoContextMenu);
        toolBarTool->setToolButtonStyle(Qt::ToolButtonIconOnly);
        main->addToolBar(Qt::TopToolBarArea, toolBarTool);

        selectionComboBox_ = new Ui::ComboBoxAction();
        toolBarTool->addWidget( selectionComboBox_ );

        toolfactory();
    }


    void SelectionController::
            toolfactory()
    {
        setLayout(new QHBoxLayout());
        layout()->setMargin(0);

        ellipse_model_.reset( new Selections::EllipseModel(      render_view()->model->display_scale()));
        ellipse_view_.reset( new Selections::EllipseView(        ellipse_model_.data() ));
        ellipse_controller_ = new Selections::EllipseController( ellipse_view_.data(), this );

        peak_model_.reset( new Selections::PeakModel(       render_view()->model->display_scale()) );
        peak_view_.reset( new Selections::PeakView(         peak_model_.data(), &render_view()->model->project()->worker ));
        peak_controller_ = new Selections::PeakController(  peak_view_.data(), this );

        spline_model_.reset( new Selections::SplineModel(      render_view()->model->display_scale()));
        spline_view_.reset( new Selections::SplineView(        spline_model_.data(), &render_view()->model->project()->worker ));
        spline_controller_ = new Selections::SplineController( spline_view_.data(), this );

        square_model_.reset( new Selections::SquareModel(      render_view()->model->display_scale()));
        square_view_.reset( new Selections::SquareView(        square_model_.data(), &render_view()->model->project()->worker ));
        square_controller_ = new Selections::SquareController( square_view_.data(), this );
    }


    void SelectionController::
            setCurrentTool( QWidget* tool, bool active )
    {
        tool_selector_->setCurrentTool( tool, active );

        render_view()->toolSelector()->setCurrentTool(
                this, 0!=tool_selector_->currentTool() );
    }


    void SelectionController::
            setCurrentSelection( Signal::pOperation filter )
    {
        _model->current_filter_ = filter;
    }


    void SelectionController::
            addComboBoxAction( QAction* action )
    {
        selectionComboBox_->addActionItem( action );
    }


    void SelectionController::
            setThisAsCurrentTool( bool active )
    {
        _render_view->toolSelector()->setCurrentTool( this, active );
    }


/*    void SelectionController::
            changeEvent ( QEvent * event )
    {
        if (event->type() & QEvent::EnabledChange)
        {
            setThisAsCurrentTool( isEnabled() );
            emit enabledChanged(isEnabled());
        }
    }*/


    void SelectionController::
            receiveAddSelection(bool active)
    {
        Tfr::Filter* f = dynamic_cast<Tfr::Filter*>(
                _model->current_filter_.get());

        if (!f) // No selection, nothing to do
            return;

        receiveAddClearSelection(active);
        f->enabled(false);
    }


    void SelectionController::
            receiveAddClearSelection(bool /*active*/)
    {
        Filters::Ellipse* ellipse =
                dynamic_cast<Filters::Ellipse*>(
                        _model->current_filter_.get() );

        if (ellipse)
        { // If selection is an ellipse, remove tfr data inside the ellipse
            ellipse->_save_inside = false;
        }

        Filters::Rectangle* rectangle=
                dynamic_cast<Filters::Rectangle*>(
                        _model->current_filter_.get() );
        if (rectangle)
            rectangle->_save_inside = false;


        Selections::Support::SplineFilter* spline=
                dynamic_cast<Selections::Support::SplineFilter*>(
                        _model->current_filter_.get() );
        if (spline)
            spline->_save_inside = false;

        _worker->appendOperation( _model->current_filter_ );
        _model->all_filters.push_back( _model->current_filter_ );
    }


    void SelectionController::
            receiveCropSelection()
    {
        // affected_samples need a sample rate
        Signal::pOperation f = _model->current_filter_;
        f->source( _worker->source() );
        Signal::Interval i = f->affected_samples().coveredInterval();

        if (0<i.count())
            return;

        // Create OperationRemoveSection to remove that section from the stream
        Signal::pOperation remove(new Signal::OperationRemoveSection(
                Signal::pOperation(), i.first, i.count() ));
        _worker->appendOperation( remove );
    }


/*    void SelectionController::
            receiveMoveSelection(bool v)
    {
        MyVector* selection = _model->selection;
        MyVector* sourceSelection = _model->sourceSelection;

        if (true==v) { // Button pressed
            // Remember selection
            sourceSelection[0] = selection[0];
            sourceSelection[1] = selection[1];

        } else { // Button released
            Signal::pOperation filter(new Filters::Ellipse(sourceSelection[0].x, sourceSelection[0].z, sourceSelection[1].x, sourceSelection[1].z, true ));

            float FS = _worker->source()->sample_rate();
            int delta = (int)(FS * (selection[0].x - sourceSelection[0].x));

            Signal::pOperation moveSelection( new Support::OperationMoveSelection(
                    Signal::pOperation(),
                    filter,
                    delta,
                    sourceSelection[0].z - selection[0].z));

            _worker->appendOperation( moveSelection );
        }
    }


    void SelectionController::
            receiveMoveSelectionInTime(bool v)
    {
        MyVector* selection = _model->selection;
        MyVector* sourceSelection = _model->sourceSelection;

        if (true==v) { // Button pressed
            // Remember selection
            sourceSelection[0] = selection[0];
            sourceSelection[1] = selection[1];

        } else { // Button released

            // Create operation to move and merge selection,
            float FS = _worker->source()->sample_rate();
            float fL = fabsf(sourceSelection[0].x - sourceSelection[1].x);
            if (sourceSelection[0].x < 0)
                fL -= sourceSelection[0].x;
            if (sourceSelection[1].x < 0)
                fL -= sourceSelection[1].x;
            unsigned L = FS * fL;
            if (fL < 0 || 0==L)
                return;
            unsigned oldStart = FS * std::max( 0.f, sourceSelection[0].x );
            unsigned newStart = FS * std::max( 0.f, selection[0].x );
            Signal::pOperation moveSelection(new Support::OperationMove( Signal::pOperation(),
                                                        oldStart,
                                                        L,
                                                        newStart));

            _worker->appendOperation( moveSelection );
        }
    }*/


    void SelectionController::
            receiveCurrentSelection(int index, bool enabled)
    {
        throw std::logic_error("receiveCurrentSelection: Not implemented");
        //setSelection(index, enabled);
    }


    void SelectionController::
            receiveFilterRemoval(int index)
    {
        throw std::logic_error("receiveFilterRemoval: Not implemented");
        //removeFilter(index);
    }


/*    void SelectionController::
            setSelection(int index, bool enabled)
    {
        TaskTimer tt("Current selection: %d", index);

        Tfr::Filter* filter = dynamic_cast<Tfr::Filter*>( _model->all_filters[index].get() );

        Filters::Ellipse *e = dynamic_cast<Filters::Ellipse*>(filter);
        if (e)
        {
            selection[0].x = e->_t1;
            selection[0].z = e->_f1;
            selection[1].x = e->_t2;
            selection[1].z = e->_f2;

            Filters::Ellipse *e2 = new Filters::Ellipse(*e );
            e2->_save_inside = true;
            e2->enabled( true );

            Signal::pOperation selectionfilter( e2 );
            selectionfilter->source( Signal::pOperation() );

            _model->getPostSink()->filter( selectionfilter );
        }

        if(filter->enabled() != enabled)
        {
            filter->enabled( enabled );

            _worker->postSink()->invalidate_samples( filter->affected_samples() );
        }
    }


    void SelectionController::
            removeFilter(int index)
    {
        TaskTimer tt("Removing filter: %d", index);

        Signal::pOperation next = _model->all_filters[index];

        if (_worker->source() == next )
        {
            _worker->source( next->source() );
            return;
        }

        Signal::pOperation prev = _worker->source();
        while ( prev->source() != next )
        {
            prev = prev->source();
            BOOST_ASSERT( prev );
        }

        prev->source( next->source() );
    }
*/
} // namespace Tools
