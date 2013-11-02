#include "selectionviewinfo.h"
#include "ui_selectionviewinfo.h"

#include "selectionmodel.h"

#include "sawe/project.h"
#include "support/computerms.h"
#include "ui/mainwindow.h"
#include "ui_mainwindow.h"
#include "signal/sinksource.h"
#include "signal/operation-basic.h"
#include "signal/oldoperationwrapper.h"
#include "tfr/stft.h"

// gpumisc
#include "neat_math.h"

using namespace Signal;

namespace Tools {

SelectionViewInfo::
        SelectionViewInfo(Sawe::Project* project, SelectionModel* model)
            :
    QDockWidget(project->mainWindow()),
    project_(project),
    model_(model),
    ui_(new Ui::SelectionViewInfo)
{
    EXCEPTION_ASSERTX(false, "not implemented: target_");

    ui_->setupUi(this);

    connect(model, SIGNAL(selectionChanged()), SLOT(selectionChanged()));

    selectionChanged();

    hide();
    project->mainWindow()->addDockWidget(Qt::RightDockWidgetArea, this);

    connect(ui_->actionSelection_Info, SIGNAL(toggled(bool)), this, SLOT(setVisible(bool)));
    connect(ui_->actionSelection_Info, SIGNAL(triggered()), this, SLOT(raise()));
    connect(this, SIGNAL(visibilityChanged(bool)), SLOT(setVisible(bool)));
    ui_->actionSelection_Info->setChecked( false );

    project->mainWindow()->getItems()->menu_Windows->addAction(ui_->actionSelection_Info);
}


SelectionViewInfo::
        ~SelectionViewInfo()
{
    delete ui_;
}


void SelectionViewInfo::
        setText(QString text)
{
    ui_->textEdit->setText( text );
}


void SelectionViewInfo::
        append(QString text)
{
    ui_->textEdit->append(text);
}


void SelectionViewInfo::
        selectionChanged()
{
/*
//Use Signal::Processing namespace
    if (target_)
        project_->targets.erase( target_ );

    if (!model_->current_selection() || !isVisibleTo(parentWidget()))
    {
        target_.reset();
        return;
    }

    Signal::pOperation selection = model_->current_selection_copy();
    target_.reset( new Target(&model_->project()->layers, "SelectionViewInfo", true, false) );
    target_->addLayerHead(project_->head);

    std::vector<Signal::pOperation> svso;
    // A target must have a post sink with a sink that can tell what to work on.
    // A target is defined by the sinks that request data to be computed.
    Signal::pOperation infoOperation(new SelectionViewInfoSink(this));
    svso.push_back( infoOperation );
    target_->post_sink()->sinks( svso );
    target_->post_sink()->filter(selection);
    target_->post_sink()->invalidate_samples(~selection->zeroed_samples_recursive());

    project_->targets.insert( target_ );
*/
    if (target_marker_)
        target_marker_.reset ();

    if (!model_->current_selection() || !isVisibleTo(parentWidget()))
    {
        target_marker_.reset ();
        return;
    }

    Signal::pOperation infoOperation(new SelectionViewInfoSink(this));
    Signal::pOperation selection = model_->current_selection_copy();
    Signal::OperationDesc::Ptr info_desc( new Signal::OldOperationDescWrapper(infoOperation));
    Signal::OperationDesc::Ptr selection_desc( new Signal::OldOperationDescWrapper(selection));

    Signal::Processing::Chain::WritePtr chain( project_->processing_chain () );
    target_marker_ = chain->addTarget(info_desc, project_->default_target());
    chain->addOperationAt(selection_desc, target_marker_);
}


void SelectionViewInfo::
        setVisible(bool visible)
{
    selectionChanged();
    QDockWidget::setVisible(visible);
}


SelectionViewInfoSink::
        SelectionViewInfoSink( SelectionViewInfo* info )
            :
            info_(info),
            searchingformaximas_(false)
{
    mixchannels_.reset(new OperationSuperpositionChannels(pOperation()));
    rms_.reset(new Support::ComputeRms(mixchannels_));
    DeprecatedOperation::source(rms_);
}


pBuffer SelectionViewInfoSink::
        read( const Interval& I )
{
    pBuffer b;

    if (!searchingformaximas_)
    {
        b = DeprecatedOperation::read(I); // note: Operation::source() == rms_
        Support::ComputeRms* rms = dynamic_cast<Support::ComputeRms*>(rms_.get());
        Intervals all = this->getInterval() - this->zeroed_samples_recursive();
        Intervals processed = rms->rms_I;
        Intervals not_processed = all - processed;
        double P0 = 1;
        double P = rms->rms;
        double db = 10*log10(P/P0);
        QString text;
        text += QString("Mean intensity: %1 db").arg(db,0,'f',1);
        if (not_processed)
            text += QString(" (%1%)")
                           .arg((1.f - not_processed.count()/(float)all.count())*100,0,'f',0);
        text +="\n";
        text += QString("Selection length: %1").arg(QString::fromStdString(SourceBase::lengthLongFormat(all.count()/sample_rate())));
        text += "\n";
        text += QString("Total signal length: %1").arg(QString::fromStdString(lengthLongFormat()));

        missing_ -= b->getInterval();

        if (missing_.empty())
        {
            text +="\n";
            searchingformaximas_ = true;
            missing_ = all;
        }

        info_->setText(text);
    }

    if (searchingformaximas_)
    {
        Interval f = missing_.fetchInterval( sample_rate() );
        Interval centerInterval = f;
        missing_-=f;
        f = Intervals(f).enlarge(sample_rate()/2).spannedInterval() & getInterval();
        f.last = f.first + Tfr::Fft().lChunkSizeS( f.count() + 1, 4 );

        Tfr::StftDesc stft;
        stft.setWindow(Tfr::StftDesc::WindowType_Hann, 0.5);
        stft.set_approximate_chunk_size( f.count() );
        stft.compute_redundant(false);
        EXCEPTION_ASSERT(stft.chunk_size() == f.count());

        b = DeprecatedOperation::source()->readFixedLength(f);

        // Only check the first channel
        // TODO check other channels
        Tfr::pChunk c = (Tfr::Stft(stft))( b->getChannel (0));
        Tfr::ChunkElement* p = c->transform_data->getCpuMemory();
        EXCEPTION_ASSERT( 1 == c->nSamples() );
        EXCEPTION_ASSERT( c->nScales() == f.count()/2+1 );

        unsigned N = c->nScales();
        std::vector<float> absValues(N);
        float* q = &absValues[0];
        for (unsigned i=0; i<N; ++i)
            q[i] = norm(p[i]);

        unsigned max_i = 0;
        float max_v = 0;
        // use i=1 to skip DC component of ft
        for (unsigned i=1; i<N; ++i)
        {
            if (q[i]>max_v)
            {
                max_v = q[i];
                max_i = i;
            }
        }

        QString text;
        if (0 == max_i)
        {
            text += QString("[%1 s, %2) s, peak n/a").arg(f.first/sample_rate(), 0, 'f', 2).arg(f.last/sample_rate(), 0, 'f', 2);
        }
        else
        {
            float interpolated_i = 0;
            quad_interpol(max_i, q, N, 1, &interpolated_i);
            float hz = c->freqAxis.getFrequency( interpolated_i );
            float hz2 = c->freqAxis.getFrequency( max_i + 1);
            float hz1 = c->freqAxis.getFrequency( max_i - 1);
            float dhz = (hz2 - hz1)*.5; // distance between bins
            dhz = sqrtf(1.f/12)*dhz;
            text += QString("[%1, %2) s, peak %3 %4 %5 Hz")
                    .arg(centerInterval.first/sample_rate(), 0, 'f', 2)
                    .arg(centerInterval.last/sample_rate(), 0, 'f', 2)
                    .arg(hz, 0, 'f', 2)
                    .arg(QChar(0xB1))
                    .arg(dhz, 0, 'f', 2);
        }

        info_->append(text);
    }


    return b;
}


void SelectionViewInfoSink::
        source(pOperation v)
{
    mixchannels_->source(v);
}


void SelectionViewInfoSink::
        invalidate_samples(const Intervals& I)
{
    missing_ |= I;
}


Intervals SelectionViewInfoSink::
        invalid_samples()
{
    return missing_;
}

} // namespace Tools
