#include "selection.h"
#include "filters/rectangle.h"
#include "tfr/transformoperation.h"
#include "signal/processing/chain.h"
#include "log.h"

using namespace Signal;
using namespace Tfr;

Selection::Selection(QQuickItem *parent) :
    QQuickItem(parent)
{
    connect (this, SIGNAL(selectionChanged()), this, SLOT(onSelectionChanged()));
}

void Selection::
        setRenderOnHeightmap(Squircle*s)
{
    render_on_heightmap_=s;
    emit renderOnHeightmapChanged ();
    connect(s,SIGNAL(rendererChanged(SquircleRenderer*)),SLOT(onRendererChanged(SquircleRenderer*)), Qt::DirectConnection);
}

void Selection::
        setFilteredHeightmap(Squircle*s)
{
    auto olsselection = selection_.lock ();
    if (olsselection && filter_heightmap_) {
        Processing::Chain::ptr chain = filter_heightmap_->chain ()->chain ();
        chain->removeOperation(olsselection);
    }
    filter_heightmap_=s;
    emit filteredHeightmapChanged ();
}

void Selection::
        setT1(double v)
{
    if(t1_==v) return;
    t1_=v;emit selectionChanged();
}

void Selection::
        setT2(double v)
{
    if(t2_==v) return;
    t2_=v;emit selectionChanged();
}

void Selection::
        setF1(double v)
{
    if(f1_==v) return;
    f1_=v;emit selectionChanged();
}

void Selection::
        setF2(double v)
{
    if(f2_==v) return;
    f2_=v;emit selectionChanged();
}

bool Selection::
        valid() const
{
    // see onSelectionChanged
    return t1_!=t2_ && f1_!=f2_;
}


void Selection::
        discardSelection()
{
    t1_=t2_=f1_=f2_=0;
    emit onSelectionChanged ();
}


void Selection::
        onSelectionChanged()
{
    if (selection_renderer_)
        selection_renderer_->setSelection (t1_, f1_, t2_, f2_);

    if (valid_ != valid())
    {
        valid_ = valid();
        emit validChanged ();
    }

    if (!filter_heightmap_)
        return;

    OperationDesc::ptr selection = selection_.lock ();
    Processing::Chain::ptr chain = filter_heightmap_->chain ()->chain ();

    if (valid_)
    {
        Processing::TargetMarker::ptr target = filter_heightmap_->renderModel ()->target_marker ();
        Signal::OperationDesc::Extent x = chain->extent(target);

        if (x.sample_rate.is_initialized ())
        {
            float fs = x.sample_rate.get ();

            if (!selection)
            {
                ChunkFilterDesc::ptr cfd{new Filters::Rectangle(0, 0, 0, 0, true)};
                selection.reset (new TransformOperationDesc(cfd));

                selection_ = selection;
                chain->addOperationAt(selection, target);
            }

            {
                // convoluted: inline lock (selection), typecast, get new lock (chunk_filter) and
                // unlock (selection.write() out of scope)
                auto cf = ((TransformOperationDesc*)selection.write ().get ())->chunk_filter ();

                Filters::Rectangle* r = dynamic_cast<Filters::Rectangle*>(cf.get ());
                r->_s1 = std::min(t1_,t2_)*fs;
                r->_f1 = std::min(f1_,f2_);
                r->_s2 = std::max(t1_,t2_)*fs;
                r->_f2 = std::max(f1_,f2_);
                r->_save_inside = true;
            }

            auto I = selection->getInvalidator();
            I->deprecateCache(Signal::Interval::Interval_ALL);
        }
    }
    else if (selection)
    {
        chain->removeOperation(selection);
    }
}

void Selection::
        onRendererChanged(SquircleRenderer* renderer)
{
    // SquircleRenderer owns renderer
    selection_renderer_ = new SelectionRenderer(renderer);
    selection_renderer_->setSelection (t1_, f1_, t2_, f2_);
}
