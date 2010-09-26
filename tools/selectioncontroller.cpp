#include "selectioncontroller.h"

// Sonic AWE
#include "renderview.h"
#include "filters/ellips.h"
#include "sawe/project.h"
#include "ui_mainwindow.h"
#include "ui/mainwindow.h"
#include "signal/operation-basic.h"
#include "support/operation-composite.h"

// Qt
#include <QMouseEvent>

namespace Tools
{
    SelectionController::SelectionController( SelectionView* view, RenderView* render_view)
        :   _view(view),
            _render_view(render_view),
            _worker(&render_view->model->project()->worker),
            selecting(false)
    {
        setupGui();
    }


    void SelectionController::
            setupGui()
    {
        Ui::MainWindow* ui = _render_view->model->project()->mainWindow()->getItems();
        connect(ui->actionActivateSelection, SIGNAL(toggled(bool)), SLOT(receiveToggleSelection(bool)));

        connect(_render_view, SIGNAL(painting()), _view, SLOT(draw()));

        connect(ui->actionActionAdd_selection, SIGNAL(triggered(bool)), SLOT(receiveAddSelection(bool)));
        connect(ui->actionActionRemove_selection, SIGNAL(triggered(bool)), SLOT(receiveAddClearSelection(bool)));
        connect(ui->actionCropSelection, SIGNAL(triggered()), SLOT(receiveCropSelection()));
        connect(ui->actionMoveSelection, SIGNAL(triggered(bool)), SLOT(receiveMoveSelection(bool)));
        connect(ui->actionMoveSelectionTime, SIGNAL(triggered(bool)), SLOT(receiveMoveSelectionInTime(bool)));

        // todo remove
        // connect(d, SIGNAL(setSelectionActive(bool)), ui->actionActivateSelection, SLOT(setChecked(bool)));
    }


    void SelectionController::
            mousePressEvent ( QMouseEvent * e )
    {
        if(isEnabled())
        {
            if( (e->button() & Qt::LeftButton) == Qt::LeftButton)
                selectionButton.press( e->x(), this->height() - e->y() );
        }

        _render_view->update();
    }


    void SelectionController::
            mouseReleaseEvent ( QMouseEvent * e )
    {
        switch ( e->button() )
        {
            case Qt::LeftButton:
                selectionButton.release();
                selecting = false;
                break;

            case Qt::MidButton:
                break;

            case Qt::RightButton:
                selectionButton.release();
                break;

            default:
                break;
        }
        _render_view->update();
    }


    void SelectionController::
            mouseMoveEvent ( QMouseEvent * e )
    {
        Tools::RenderView &r = *_render_view;
        r.makeCurrent();

        int x = e->x(), y = this->height() - e->y();
    //    TaskTimer tt("moving");

        if ( selectionButton.isDown() )
        {
            MyVector* selection = model()->selection;
            GLdouble p[2];
            if (selectionButton.worldPos(x, y, p[0], p[1], r.xscale))
            {
                if (!selecting) {
                    selection[0].x = selection[1].x = selectionStart.x = p[0];
                    selection[0].y = selection[1].y = selectionStart.y = 0;
                    selection[0].z = selection[1].z = selectionStart.z = p[1];
                    selecting = true;
                } else {
                    float rt = p[0]-selectionStart.x;
                    float rf = p[1]-selectionStart.z;
                    selection[0].x = selectionStart.x + .5f*rt;
                    selection[0].y = 0;
                    selection[0].z = selectionStart.z + .5f*rf;
                    selection[1].x = selection[0].x + .5f*sqrtf(2.f)*rt;
                    selection[1].y = 0;
                    selection[1].z = selection[0].z + .5f*sqrtf(2.f)*rf;

                    Signal::PostSink* postsink = model()->getPostSink();

                    Signal::pOperation newFilter( new Filters::Ellips(selection[0].x, selection[0].z, selection[1].x, selection[1].z, true ));
                    newFilter->source( model()->project->worker.source());
                    postsink->filter( newFilter );
                }
            }
        }

        //Updating the buttons
        selectionButton.update( x, y );

        _render_view->model->project()->worker.requested_fps(30);
        _render_view->update();
    }


    void SelectionController::
            changeEvent ( QEvent * event )
    {
        if (QEvent::EnabledChange == event->type())
            _view->enabled = isEnabled();
    }


    void SelectionController::
            receiveToggleSelection(bool active)
    {
        if (active)
        {
            _render_view->toolSelector()->setCurrentTool( this );
        }

        setEnabled(active);
    }


    void SelectionController::
            receiveAddSelection(bool active)
    {
        Signal::PostSink* postsink = model()->getPostSink();

        Tfr::Filter* f = dynamic_cast<Tfr::Filter*>(postsink->filter().get());
        if (!f) // No selection, nothing to do
            return;

        receiveAddClearSelection(active);
        f->enabled(false);
    }


    void SelectionController::
            receiveAddClearSelection(bool /*active*/)
    {
        Signal::PostSink* postsink = model()->getPostSink();

        if (!postsink->filter())
            return;

        { // If selection is an ellips, remove tfr data inside the ellips
            Filters::Ellips* ef = dynamic_cast<Filters::Ellips*>( postsink->filter().get() );
            if (ef)
                ef->_save_inside = false;
        }


        Signal::pOperation postsink_filter = postsink->filter();
        _worker->appendOperation( postsink_filter );
        model()->all_filters.push_back( postsink_filter );
    }


    void SelectionController::
            receiveCropSelection()
    {
        // Find out what to crop based on selection
        unsigned FS = _worker->source()->sample_rate();
        MyVector* selection = model()->selection;
        float radie = fabsf(selection[0].x - selection[1].x);
        unsigned start = std::max(0.f, selection[0].x - radie) * FS;
        unsigned end = (selection[0].x + radie) * FS;

        if (end<=start)
            return;

        // Create OperationRemoveSection to remove that section from the stream
        Signal::pOperation remove(new Signal::OperationRemoveSection( Signal::pOperation(), start, end-start ));
        _worker->appendOperation( remove );
    }


    void SelectionController::
            receiveMoveSelection(bool v)
    {
        MyVector* selection = model()->selection;
        MyVector* sourceSelection = model()->sourceSelection;

        if (true==v) { // Button pressed
            // Remember selection
            sourceSelection[0] = selection[0];
            sourceSelection[1] = selection[1];

        } else { // Button released
            Signal::pOperation filter(new Filters::Ellips(sourceSelection[0].x, sourceSelection[0].z, sourceSelection[1].x, sourceSelection[1].z, true ));

            unsigned FS = _worker->source()->sample_rate();
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
        MyVector* selection = model()->selection;
        MyVector* sourceSelection = model()->sourceSelection;

        if (true==v) { // Button pressed
            // Remember selection
            sourceSelection[0] = selection[0];
            sourceSelection[1] = selection[1];

        } else { // Button released

            // Create operation to move and merge selection,
            unsigned FS = _worker->source()->sample_rate();
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
    }


    void SelectionController::
            receiveCurrentSelection(int index, bool enabled)
    {
        setSelection(index, enabled);
    }


    void SelectionController::
            receiveFilterRemoval(int index)
    {
        removeFilter(index);
    }


    void SelectionController::
            setSelection(int index, bool enabled)
    {
        TaskTimer tt("Current selection: %d", index);

        Tfr::Filter* filter = dynamic_cast<Tfr::Filter*>( model()->all_filters[index].get() );

        MyVector* selection = model()->selection;
        Filters::Ellips *e = dynamic_cast<Filters::Ellips*>(filter);
        if (e)
        {
            selection[0].x = e->_t1;
            selection[0].z = e->_f1;
            selection[1].x = e->_t2;
            selection[1].z = e->_f2;

            Filters::Ellips *e2 = new Filters::Ellips(*e );
            e2->_save_inside = true;
            e2->enabled( true );

            Signal::pOperation selectionfilter( e2 );
            selectionfilter->source( Signal::pOperation() );

            model()->getPostSink()->filter( selectionfilter );
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

        Signal::pOperation next = model()->all_filters[index];

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
} // namespace Tools
