#include "selectioncontroller.h"

#include <QMouseEvent>

// Sonic AWE
#include "renderview.h"
#include "sawe/project.h"
#include "ui_mainwindow.h"
#include "ui/mainwindow.h"
#include "support/operation-composite.h"
#include "support/toolbar.h"
#include "commands/changeselectioncommand.h"
#include "sawe/configuration.h"

#include "selections/ellipsecontroller.h"
#include "selections/ellipsemodel.h"
#include "selections/ellipseview.h"

#include "selections/peakcontroller.h"
#include "selections/peakmodel.h"
#include "selections/peakview.h"

#include "selections/splinecontroller.h"
#include "selections/splinemodel.h"
#include "selections/splineview.h"

#include "selections/rectanglecontroller.h"
#include "selections/rectanglemodel.h"
#include "selections/rectangleview.h"

#include "log.h"

namespace Tools
{
    SelectionController::
            SelectionController( SelectionModel* model, RenderView* render_view )
                :
                _model(model),
                _render_view(render_view),
                //_worker(&render_view->model->project()->worker),
                selectionComboBox_(0),
                tool_selector_( new Support::ToolSelector(render_view->model->project()->commandInvoker(), this)),
                deselect_action_(0),
                selecting(false)
    {
        setEnabled( false );

        setupGui();
    }


    SelectionController::
            ~SelectionController()
    {
        TaskInfo ti("%s", __FUNCTION__);

        if (!ellipse_controller_.isNull())
            delete ellipse_controller_;

//        if (!peak_controller_.isNull())
//            delete peak_controller_;

//        if (!spline_controller_.isNull())
//            delete spline_controller_;

        if (!rectangle_controller_.isNull())
            delete rectangle_controller_;
    }


    void SelectionController::
            setupGui()
    {
        Ui::SaweMainWindow* main = _model->project()->mainWindow();
        Ui::MainWindow* ui = main->getItems();

        //connect(ui->actionActionAdd_selection, SIGNAL(triggered()), SLOT(receiveAddSelection()));
        connect(ui->actionActionRemove_selection, SIGNAL(triggered()), SLOT(receiveAddClearSelection()));
        connect(ui->actionCropSelection, SIGNAL(triggered()), SLOT(receiveCropSelection()));
        //connect(ui->actionMoveSelection, SIGNAL(toggled(bool)), SLOT(receiveMoveSelection(bool)));
        //connect(ui->actionMoveSelectionTime, SIGNAL(toggled(bool)), SLOT(receiveMoveSelectionInTime(bool)));

        ui->actionActionAdd_selection->setEnabled( false );
        ui->actionActionRemove_selection->setEnabled( true );
        ui->actionCropSelection->setEnabled( true );
        ui->actionMoveSelection->setEnabled( false );
        ui->actionMoveSelectionTime->setEnabled( false );

        Support::ToolBar* toolBarTool = new Support::ToolBar(main);
        toolBarTool->setObjectName(QString::fromUtf8("toolBarSelectionController"));
        toolBarTool->setEnabled(true);
        toolBarTool->setContextMenuPolicy(Qt::NoContextMenu);
        toolBarTool->setToolButtonStyle(Qt::ToolButtonIconOnly);
        main->addToolBar(Qt::TopToolBarArea, toolBarTool);

        connect(ui->actionToggleSelectionToolBox, SIGNAL(toggled(bool)), toolBarTool, SLOT(setVisible(bool)));
        connect(toolBarTool, SIGNAL(visibleChanged(bool)), ui->actionToggleSelectionToolBox, SLOT(setChecked(bool)));

        selectionComboBox_ = new Ui::ComboBoxAction();
        toolBarTool->addWidget( selectionComboBox_ );
        toolBarTool->addAction( ui->actionActionRemove_selection );
        toolBarTool->addAction( ui->actionCropSelection );
        toolBarTool->setVisible( true );

        connect(_model, SIGNAL(selectionChanged()), SLOT(onSelectionChanged()));
        //connect(_model->project()->head.get(), SIGNAL(headChanged()), SLOT(tryHeadAsSelection()));
        connect(selectionComboBox_, SIGNAL(toggled(bool)), SLOT(selectionComboBoxToggled()));
        connect(this, SIGNAL(enabledChanged(bool)), selectionComboBox_, SLOT(setChecked(bool)));

        deselect_action_ = new QAction(this);
        connect(deselect_action_, SIGNAL(triggered()), SLOT(deselect()));
        deselect_action_->setShortcut(Qt::Key_Escape);
        addAction(deselect_action_);


        setCurrentSelection(Signal::OperationDesc::Ptr());

        toolfactory();

        _render_view->tool_selector->default_tool = this;

#ifdef TARGET_hast
        ui->actionActionRemove_selection->setVisible(false);
        ui->actionCropSelection->setVisible(false);
#endif

    }


    void SelectionController::
            toolfactory()
    {
        setLayout(new QHBoxLayout());
        layout()->setMargin(0);
        /*

#ifndef TARGET_hast
        ellipse_model_.reset( new Selections::EllipseModel(      render_view()->model));
        ellipse_view_.reset( new Selections::EllipseView(        ellipse_model_.data() ));
        ellipse_controller_ = new Selections::EllipseController( ellipse_view_.data(), this );
#endif
*/
        rectangle_model_.reset( new Selections::RectangleModel(      render_view()->model, render_view()->model->project() ));
        rectangle_view_.reset( new Selections::RectangleView(        rectangle_model_.data() ));
        rectangle_controller_ = new Selections::RectangleController( rectangle_view_.data(), this );
/*
#ifndef TARGET_hast
        spline_model_.reset( new Selections::SplineModel(      render_view()->model));
        spline_view_.reset( new Selections::SplineView(        spline_model_.data(), &render_view()->model->project()->worker ));
        spline_controller_ = new Selections::SplineController( spline_view_.data(), this );

        peak_model_.reset( new Selections::PeakModel(       render_view()->model) );
        peak_view_.reset( new Selections::PeakView(         peak_model_.data(), &render_view()->model->project()->worker ));
        peak_controller_ = new Selections::PeakController(  peak_view_.data(), this );
#endif
*/
        connect( render_view()->model, SIGNAL(modelChanged(Tools::ToolModel*)), SLOT(renderModelChanged(Tools::ToolModel*)) );
    }


    void SelectionController::
            renderModelChanged(Tools::ToolModel* m)
    {
        Tools::RenderModel* renderModel = dynamic_cast<Tools::RenderModel*>( m );
        bool ortho1D = 0==renderModel->_rx;
        bool frequencySelection = fmod( renderModel->effective_ry() + 90, 180.f ) == 0.f;
        bool timeSelection = fmod( renderModel->effective_ry(), 180.f ) == 0.f;
        Ui::MainWindow* ui = model()->project()->mainWindow()->getItems();

        ui->actionEllipseSelection->setEnabled( !ortho1D );
        ui->actionRectangleSelection->setEnabled( !ortho1D );
        ui->actionSplineSelection->setEnabled( !ortho1D );
        ui->actionPeakSelection->setEnabled( !ortho1D );
        ui->actionPolygonSelection->setEnabled( !ortho1D );
        ui->actionTimeSelection->setEnabled( timeSelection || !ortho1D );
        ui->actionFrequencySelection->setEnabled( frequencySelection || !ortho1D );
    }


    void SelectionController::
            deselect()
    {
        setCurrentSelection(Signal::OperationDesc::Ptr());
    }


    void SelectionController::
            setCurrentTool( QWidget* tool, bool active )
    {
//        render_view()->toolSelector()->setCurrentTool(
//                this, (active && tool) || (tool && tool==tool_selector_->currentTool()));

//        if (tool_selector_->currentTool() && tool_selector_->currentTool()!=tool)
//            tool_selector_->currentTool()->setVisible( false );
        tool_selector_->setCurrentTool( tool, active );
//        if (tool_selector_->currentTool())
//            tool_selector_->currentTool()->setVisible( true );
        setThisAsCurrentTool( 0!=tool_selector_->currentTool() );
    }


    void SelectionController::
            setCurrentSelectionCommand( Signal::OperationDesc::Ptr selection )
    {
        _model->set_current_selection( selection );

        bool enabled_actions = 0 != selection.get();

        Ui::SaweMainWindow* main = _model->project()->mainWindow();
        Ui::MainWindow* ui = main->getItems();

        //ui->actionActionAdd_selection->setEnabled( enabled_actions );
        ui->actionActionRemove_selection->setEnabled( enabled_actions );
        ui->actionCropSelection->setEnabled( enabled_actions );
    }


    void SelectionController::
            setCurrentSelection( Signal::OperationDesc::Ptr selection )
    {
        if (_model->current_selection() != selection)
        {
            Commands::pCommand p( new Commands::ChangeSelectionCommand(this, selection));
            this->model()->project()->commandInvoker()->invokeCommand( p );

            //model()->project()->mainWindow()->getItems()->actionPlaySelection->trigger();
        }
        else if (Signal::OperationDesc::Ptr() == selection)
        {
            this->model ()->set_current_selection(Signal::OperationDesc::Ptr());
        }
    }


    void SelectionController::
            onSelectionChanged()
    {
    }

#if 0
    void SelectionController::
            tryHeadAsSelection()
    {
        // SelectionController can't see the head.
        // The SelectionChangedCommand can take care of this instead.
        Signal::OperationDesc::Ptr t = _model->project()->head->head_source();
        if (dynamic_cast<Tools::Support::OperationOnSelection*>(t.get()))
            t = dynamic_cast<Tools::Support::OperationOnSelection*>(t.get())->selection();

        if (t && t == _model->current_selection())
        {
            // To reset the same selection again, first clear the current selection
            _model->set_current_selection( Signal::OperationDesc::Ptr() );
        }

        _model->try_set_current_selection( t );

        if (!_model->current_selection())
        {
            setThisAsCurrentTool( false );
        }
    }
#endif

    void SelectionController::
            addComboBoxAction( QAction* action )
    {
        //selectionComboBox_->parentWidget()->addAction(action);
        selectionComboBox_->addActionItem( action );
    }


    void SelectionController::
            setThisAsCurrentTool( bool active )
    {
        if (!active)
            setCurrentSelection( Signal::OperationDesc::Ptr() );

        if (active)
            _render_view->toolSelector()->setCurrentToolCommand( this );
        else
            _render_view->toolSelector()->setCurrentTool( this, active );
    }


    void SelectionController::
            changeEvent ( QEvent * event )
    {
        if (event->type() == QEvent::EnabledChange)
        {
            setThisAsCurrentTool( isEnabled() );
            emit enabledChanged(isEnabled());
        }
    }


    void SelectionController::
            mousePressEvent ( QMouseEvent * e )
    {
        if (Qt::RightButton == e->button())
        {
            this->setCurrentSelection(Signal::OperationDesc::Ptr());
            render_view()->userinput_update();
        }
    }

//    void SelectionController::
//            receiveAddSelection()
//    {
//        if (!_model->current_selection())
//            return;

//        receiveAddClearSelection();

//        _worker->source()->enabled(false);
//    }


    void SelectionController::
            receiveAddClearSelection()
    {
        if (!_model->current_selection())
            return;

        Signal::OperationDesc::Ptr o = _model->current_selection_copy( SelectionModel::SaveInside_FALSE );

        _model->set_current_selection( Signal::OperationDesc::Ptr() );
        _model->project()->appendOperation( o );
        _model->set_current_selection( o );
        _model->all_selections.push_back( o );

        Log("Clear selection\n%s") % read1(o)->toString().toStdString();
    }


    void SelectionController::
            receiveCropSelection()
    {
        if (!_model->current_selection())
            return;

        Signal::OperationDesc::Ptr o = _model->current_selection_copy( SelectionModel::SaveInside_TRUE );

        _model->set_current_selection( Signal::OperationDesc::Ptr() );
        _model->project()->appendOperation( o );
        _model->set_current_selection( o );
        _model->all_selections.push_back( o );

        Log("Crop selection\n%s") % read1(o)->toString().toStdString();
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
            Signal::OperationDesc::Ptr filter(new Filters::Ellipse(sourceSelection[0].x, sourceSelection[0].z, sourceSelection[1].x, sourceSelection[1].z, true ));

            float FS = _worker->source()->sample_rate();
            int delta = (int)(FS * (selection[0].x - sourceSelection[0].x));

            Signal::OperationDesc::Ptr moveSelection( new Support::OperationMoveSelection(
                    Signal::OperationDesc::Ptr(),
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
            Signal::OperationDesc::Ptr moveSelection(new Support::OperationMove( Signal::OperationDesc::Ptr(),
                                                        oldStart,
                                                        L,
                                                        newStart));

            _worker->appendOperation( moveSelection );
        }
    }*/


    void SelectionController::
            receiveCurrentSelection(int /*index*/, bool /*enabled*/)
    {
        throw std::logic_error("receiveCurrentSelection: Not implemented");
        //setSelection(index, enabled);
    }


    void SelectionController::
            receiveFilterRemoval(int /*index*/)
    {
        throw std::logic_error("receiveFilterRemoval: Not implemented");
        //removeFilter(index);
    }


    void SelectionController::
            selectionComboBoxToggled()
    {
        setThisAsCurrentTool( selectionComboBox_->isChecked() );
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

            Signal::OperationDesc::Ptr selectionfilter( e2 );
            selectionfilter->source( Signal::OperationDesc::Ptr() );

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

        Signal::OperationDesc::Ptr next = _model->all_filters[index];

        if (_worker->source() == next )
        {
            _worker->source( next->source() );
            return;
        }

        Signal::OperationDesc::Ptr prev = _worker->source();
        while ( prev->source() != next )
        {
            prev = prev->source();
            EXCEPTION_ASSERT( prev );
        }

        prev->source( next->source() );
    }
*/
} // namespace Tools
