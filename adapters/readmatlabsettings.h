#ifndef READMATLABSETTINGS_H
#define READMATLABSETTINGS_H

#include "matlabfunction.h"

#include <QScopedPointer>
#include <signal/source.h>

namespace Adapters {

class ReadMatlabSettings: public QObject
{
    Q_OBJECT
public:
    enum MetaData
    {
        MetaData_Settings,
        MetaData_Source
    };

    static void readSettingsAsync(QString filename, QObject *receiver, const char *member, const char *failedmember=0);
    static Signal::pBuffer TryReadBuffer(QString filename);

    ReadMatlabSettings( QString filename, MetaData );

    void readAsyncAndDeleteSelfWhenDone();

    Signal::pBuffer sourceBuffer();
    DefaultMatlabFunctionSettings settings;
    std::string iconpath();

signals:
    void sourceRead();
    void settingsRead( Adapters::DefaultMatlabFunctionSettings settings );
    void failed( QString filename, QString info );

private slots:
    void checkIfReady();

private:
    MetaData type_;
    bool deletethis_;
    Signal::pBuffer source_buffer_;
    std::string iconpath_;

    QScopedPointer<MatlabFunction> function_;

    void readSettings(std::string filename);
    void readSource(std::string filename);
};

} // namespace Adapters

#endif // READMATLABSETTINGS_H
