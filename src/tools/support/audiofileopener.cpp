#include "audiofileopener.h"
#include "adapters/audiofile.h"

using namespace Adapters;

namespace Tools {
namespace Support {

OpenfileController::Patterns AudiofileOpener::
        patterns()
{
    QString filter1 = Audiofile::getFileFormatsQtFilter(false).c_str ();
    QString filter2 = Audiofile::getFileFormatsQtFilter(true).c_str ();

    OpenfileController::Patterns R;
    R.push_back(OpenfileController::Patterns::value_type(filter1, filter2));

    return R;
}


Signal::OperationDesc::ptr AudiofileOpener::
        reopen(QString url, Signal::OperationDesc::ptr)
{
    boost::shared_ptr<Audiofile> audiofile;
    try {
        audiofile.reset (new Audiofile(url.toStdString ()));
    } catch (const std::exception&) {}

    if (!audiofile)
        return Signal::OperationDesc::ptr();

    return Signal::OperationDesc::ptr(new AudiofileDesc(audiofile));
}

} // namespace Support
} // namespace Tools


#include "adapters/writewav.h"

#include <QStandardPaths>
#include <QDir>
#include <QFile>


namespace Tools {
namespace Support {

void AudiofileOpener::
        test()
{
    // The AudiofileController class should open files supported by libsndfile.
    {
        // unit tests should not allocate resources or rely on external setups (such as a file at a specific location)
        QDir tmplocation = QStandardPaths::writableLocation(QStandardPaths::TempLocation);
        std::string filename = tmplocation.filePath("dummywav.wav").toStdString();
        Signal::pBuffer buffer(new Signal::Buffer(2, 10, 10, 3));
        for (unsigned c=0; c<buffer->number_of_channels (); c++) {
            Signal::pMonoBuffer mono = buffer->getChannel (c);
            float *p = mono->waveform_data ()->getCpuMemory ();
            for (int s=0; s<mono->number_of_samples (); s++)
                p[s] = (c*100 + s)*0.001;
        }
        Adapters::WriteWav::writeToDisk(filename, buffer, false);

        OpenfileController openfile;
        openfile.registerOpener (new AudiofileOpener);

        Signal::OperationDesc::ptr od;
        od = openfile.open ("blaj");
        EXCEPTION_ASSERT(!od);

        od = openfile.open (filename.c_str ());
        EXCEPTION_ASSERT(od);
        EXCEPTION_ASSERT(dynamic_cast<AudiofileDesc*>(od.raw ()));
        EXCEPTION_ASSERT_EQUALS(od.read ()->toString().toStdString(), filename);

        {
            Signal::Operation::ptr o = od.read ()->createOperation(0);
            EXCEPTION_ASSERT(o);
            auto op = o.write ();
            Signal::OperationDesc::Extent x = od.read ()->extent();
            Signal::pBuffer b(new Signal::Buffer(0, x.interval.get().count(), x.sample_rate.get(), x.number_of_channels.get()));
            Signal::pBuffer b2 = op->process(b);

            EXCEPTION_ASSERT_EQUALS(buffer->number_of_channels (), b2->number_of_channels ());
            EXCEPTION_ASSERT_EQUALS(buffer->number_of_samples (), b2->number_of_samples ());

            float maxdiff = 0;
            for (unsigned c=0; c<buffer->number_of_channels (); c++) {
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
