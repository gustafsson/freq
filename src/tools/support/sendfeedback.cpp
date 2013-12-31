#include "sendfeedback.h"

#include "buildhttppost.h"
#include "sawe/application.h"

#include <QDir>
#include <QSettings>

namespace Tools {
namespace Support {

SendFeedback::SendFeedback(QObject *parent) :
    QObject(parent),
    targetUrl("http://feedback.sonicawe.com/sendfeedback.php")
{
}


QString SendFeedback::
        sendLogFiles(QString email, QString message, QString extraFile)
{
    BuildHttpPost postdata;

    QString logdir = Sawe::Application::log_directory();
    unsigned count = 0;
    QFileInfoList filesToSend = QDir(logdir).entryInfoList();
    if (QFile::exists(extraFile))
        filesToSend.append(extraFile);

    QString omittedMessage;
    qint64 uploadLimit = 7 << 20;
    foreach(QFileInfo f, filesToSend)
    {
        if (f.size() > uploadLimit)
        {
            omittedMessage += QString("File %1 (%2) was omitted\n")
                              .arg(f.fileName())
                              .arg(DataStorageVoid::getMemorySizeText(f.size()).c_str());
            continue;
        }
        if (postdata.addFile(f))
            count++;
    }

    if (!omittedMessage.isEmpty())
    {
        message = omittedMessage + "\n" + message;
    }

    postdata.addKeyValue( "email", email );
    postdata.addKeyValue( "message", message );
    postdata.addKeyValue( "value", QSettings().value("value").toString() );

    QByteArray feedbackdata = postdata;

    unsigned N = feedbackdata.size();
    TaskInfo ti("SendFeedback sends %s in %u files",
             DataStorageVoid::getMemorySizeText(N).c_str(),
             count);

    manager.reset( new QNetworkAccessManager(this) );
    connect(manager.data(), SIGNAL(finished(QNetworkReply*)),
            this, SLOT(finished(QNetworkReply*)));
    postdata.send( manager.data(), targetUrl );

    return omittedMessage;
}

} // namespace Support
} // namespace Tools
