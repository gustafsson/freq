#include "csvfileopener.h"
#include "adapters/csvtimeseries.h"
#include <fstream>
#include <QInputDialog>

using namespace Adapters;

namespace Tools {
namespace Support {

OpenfileController::Patterns CsvfileOpener::
        patterns()
{
    QString filter1 = CsvTimeseries::getFileFormatsQtFilter(false).c_str ();
    QString filter2 = CsvTimeseries::getFileFormatsQtFilter(true).c_str ();

    OpenfileController::Patterns R;
    R.push_back(OpenfileController::Patterns::value_type(filter1, filter2));

    return R;
}


Signal::OperationDesc::ptr CsvfileOpener::
        reopen(QString url, Signal::OperationDesc::ptr prev)
{
    Signal::OperationDesc::ptr csvfile;
    try {
        CsvTimeseries* c;
        csvfile.reset (c = new CsvTimeseries(url.toStdString ()));

        c->setSampleRate ([&](){
            if (prev)
                return prev.read ()->extent().sample_rate.get();
            else
            {
                float fs = QInputDialog::getDouble (0, ".csv sample rate",
                                                       "Enter the sample rate for the csv data with one sample per row.\nMultiple values on the same row are interpreted as multiple channels.", 1);
                if (fs <= 0)
                    return 1.f;
                return fs;
            }
        }());

    } catch (const std::ifstream::failure&) {
    } catch (const std::exception& x) {
        TaskInfo(boost::format("CsvfileOpener: %s") % boost::diagnostic_information(x));
    }

    return csvfile;
}

} // namespace Support
} // namespace Tools


#include <QStandardPaths>
#include <QDir>
#include <QFile>
#include "test/randombuffer.h"
#include <fstream>

namespace Tools {
namespace Support {

void CsvfileOpener::
        test()
{
    // The AudiofileController class should open files supported by libsndfile.
    {
        // unit tests should not allocate resources or rely on external setups (such as a file at a specific location)
        QDir tmplocation = QStandardPaths::writableLocation(QStandardPaths::TempLocation);
        std::string filename = tmplocation.filePath("dummywav.csv").toStdString();
        Signal::pBuffer buffer = Test::RandomBuffer::randomBuffer (Signal::Interval(0,15),31,3);

        std::ofstream writefile(filename);
        for (int s=0; s<buffer->number_of_samples (); s++)
        {
            // Make a messy csv file, random spaces and random delimiters
            for (unsigned c=0; c<buffer->number_of_channels (); c++) {
                Signal::pMonoBuffer mono = buffer->getChannel (c);
                float *p = mono->waveform_data ()->getCpuMemory ();
                if ((s+c)%3)
                    writefile << " ";
                writefile << p[s];
                if ((s+c)%5)
                    writefile << " ";
                switch((s+c)%4) {
                case 0: writefile << ","; break;
                case 1: writefile << ";"; break;
                case 2: writefile << "\t"; break;
                case 3: writefile << " "; break;
                }
            }
            writefile << std::endl;
        }
        writefile.close ();

        OpenfileController openfile;
        openfile.registerOpener (new CsvfileOpener);

        Signal::OperationDesc::ptr od;
        od = openfile.open ("blaj");
        EXCEPTION_ASSERT(!od);

        Signal::OperationDesc::ptr prev(new Signal::BufferSource(
                Test::RandomBuffer::randomBuffer (Signal::Interval(0,1),31,19)
            ));
        od = openfile.reopen (filename.c_str (), prev);
        EXCEPTION_ASSERT(od);
        EXCEPTION_ASSERT(dynamic_cast<CsvTimeseries*>(od.raw ()));
        EXCEPTION_ASSERT_EQUALS(od.read ()->toString().toStdString(), filename);

        {
            Signal::Operation::ptr o = od.read ()->createOperation(0);
            EXCEPTION_ASSERT(o);
            auto op = o.write ();
            Signal::OperationDesc::Extent x = od.read ()->extent();
            Signal::pBuffer b(new Signal::Buffer(0, x.interval.get().count(), x.sample_rate.get(), x.number_of_channels.get()));
            Signal::pBuffer b2 = op->process(b);

            EXCEPTION_ASSERT(*b2 == *buffer);
            EXCEPTION_ASSERT_EQUALS(buffer->number_of_channels (), b2->number_of_channels ());
            EXCEPTION_ASSERT_EQUALS(buffer->number_of_samples (), b2->number_of_samples ());

            float maxdiff = 0;
            for (unsigned c=0; c<buffer->number_of_channels (); c++)
            {
                Signal::pMonoBuffer mono1 = buffer->getChannel (c);
                Signal::pMonoBuffer mono2 = b2->getChannel (c);
                float *p1 = mono1->waveform_data ()->getCpuMemory ();
                float *p2 = mono2->waveform_data ()->getCpuMemory ();
                for (int s=0; s<mono1->number_of_samples (); s++)
                    maxdiff = std::max(maxdiff, std::fabs(p1[s] - p2[s]));
            }

            EXCEPTION_ASSERT_LESS(maxdiff, 1.f/(1<<15));
        }

        QFile(filename.c_str ()).remove ();
    }
}

} // namespace Support
} // namespace Tools
