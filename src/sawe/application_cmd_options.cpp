#include "application.h"

#include "configuration.h"

// heightmap
#include "heightmap/renderer.h"
#include "heightmap/collection.h"

// tfr
#include "tfr/cwt.h"

// adapters
#include "adapters/csv.h"
#include "adapters/hdf5.h"
#include "adapters/playback.h"

// gpumisc
#include "redirectstdout.h"

// boost
#include <boost/foreach.hpp>

// Qt
#include <QGLContext>
#include <QMessageBox>
#include <QErrorMessage>

// std
#include <string>
#include <stdio.h>

using namespace std;

namespace Sawe {


void Application::
        execute_command_line_options()
{
    string message = Sawe::Configuration::parseCommandLineMessage();

    if (!message.empty())
    {
        cerr    << message << endl    // Want output in logfile
                << Sawe::Configuration::commandLineUsageString();
        this->rs.reset();
        cerr    << message << endl    // Want output in console window, if any
                << Sawe::Configuration::commandLineUsageString();
        QErrorMessage::qtHandler()->showMessage( QString::fromStdString( message ) );
        ::exit(1);
        //mb.setWindowModality( Qt::ApplicationModal );
        //mb.show();
        return;
    }


    Sawe::pProject p; // p will be owned by Application and released before a.exec()

    if (!Sawe::Configuration::input_file().empty())
        p = Sawe::Application::slotOpen_file( Sawe::Configuration::input_file() );

    if (!p)
        p = Sawe::Application::slotNew_recording( );

    if (!p)
        ::exit(3);

    Tools::RenderModel& render_model = p->tools().render_model;
    Tools::Support::TransformDescs::WritePtr td (render_model.transform_descs ());
    Tfr::Cwt& cwt = td->getParam<Tfr::Cwt>();
    //Signal::pOperation source = render_model.renderSignalTarget->post_sink()->source();
    Signal::OperationDesc::Extent extent = read1(p->processing_chain ())->extent(p->default_target ());
    Signal::IntervalType number_of_samples = extent.interval.get_value_or (Signal::Interval());
    float sample_rate = extent.sample_rate.get_value_or (1);
    unsigned samples_per_chunk_hint = Sawe::Configuration::samples_per_chunk_hint();
    unsigned total_samples_per_chunk = cwt.prev_good_size( 1<<samples_per_chunk_hint, sample_rate );

    bool sawe_exit = false;

/*
//Use Signal::Processing namespace
    unsigned get_csv = Sawe::Configuration::get_csv();
    if (get_csv != (unsigned)-1) {
        if (0==number_of_samples) {
            Sawe::Application::display_fatal_exception(std::invalid_argument("Can't extract CSV without input file."));
            ::exit(4);
        }

        Adapters::Csv csv(QString("sonicawe-%1.csv").arg(get_csv).toStdString());
        csv.source( source );
        csv.read( Signal::Interval( get_csv*total_samples_per_chunk, (get_csv+1)*total_samples_per_chunk ));
        TaskInfo("Samples per chunk = %u", total_samples_per_chunk);
        sawe_exit = true;
    }

    unsigned get_hdf = Sawe::Configuration::get_hdf();
    if (get_hdf != (unsigned)-1) {
        if (0==number_of_samples) {
            Sawe::Application::display_fatal_exception(std::invalid_argument("Can't extract HDF without input file."));
            ::exit(5);
        }

        Adapters::Hdf5Chunk hdf5(QString("sonicawe-%1.h5").arg(get_hdf).toStdString());
        hdf5.source( source );
        hdf5.read( Signal::Interval( get_hdf*total_samples_per_chunk, (get_hdf+1)*total_samples_per_chunk ));
        TaskInfo("Samples per chunk = %u", total_samples_per_chunk);
        sawe_exit = true;
    }
*/

    if (Sawe::Configuration::get_chunk_count()) {
        TaskInfo("number of samples = %u", number_of_samples);
        TaskInfo("samples per chunk = %u", total_samples_per_chunk);
        TaskInfo("chunk count = %u", (number_of_samples + total_samples_per_chunk-1) / total_samples_per_chunk);
        this->rs.reset();
        cout    << "number_of_samples = " << number_of_samples << endl
                << "samples_per_chunk = " << total_samples_per_chunk << endl
                << "chunk_count = " << (number_of_samples + total_samples_per_chunk-1) / total_samples_per_chunk << endl;
        sawe_exit = true;
    }

    if (sawe_exit)
    {
        ::exit(0);
    }
    else
    {
        // Ensures that an OpenGL context is created
        if( !QGLContext::currentContext() )
            QMessageBox::information(0,"Sonic AWE", "Sonic AWE couldn't start");
    }
}


void Application::
        apply_command_line_options( pProject p )
{
    {
        Tools::Support::TransformDescs::WritePtr td (p->tools().render_model.transform_descs ());
        Tfr::Cwt& cwt = td->getParam<Tfr::Cwt>();
        cwt.scales_per_octave( Sawe::Configuration::scales_per_octave() );
        cwt.set_wanted_min_hz( Sawe::Configuration::min_hz() );
        cwt.wavelet_time_support( Sawe::Configuration::wavelet_time_support() );
        cwt.wavelet_scale_support( Sawe::Configuration::wavelet_scale_support() );
    }

/*
//Use Signal::Processing namespace
#ifndef SAWE_NO_MUTEX
    if (Sawe::Configuration::feature("worker_thread"))
        p->worker.start();
#endif
*/

    Tools::ToolFactory &tools = p->tools();

    tools.playback_model.selection_filename  = Sawe::Configuration::selection_output_file();

    Heightmap::BlockSize newbc =
                Heightmap::BlockSize(
                    Sawe::Configuration::samples_per_block(),
                    Sawe::Configuration::scales_per_block ());

    tools.render_model.block_size ( newbc );

    tools.render_view()->emitTransformChanged();
}

} // namespace Sawe
