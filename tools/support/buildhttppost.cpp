#include "buildhttppost.h"

#include <QNetworkRequest>
#include <QNetworkAccessManager>

namespace Tools {
namespace Support {

BuildHttpPost::
        BuildHttpPost()
            :
            bound( "---------------------------7d935033608e2" )
{
    crlf = 0x0d;
    crlf += 0x0a;
}


BuildHttpPost::
        ~BuildHttpPost()
{

}


bool BuildHttpPost::
        addFile(QFileInfo f )
{
    if (!f.isFile())
        return false;

    QString p = f.absoluteFilePath();
    QFile mfile(p);
    mfile.open(QIODevice::ReadOnly);
    if (!mfile.isOpen())
        return false;

    QByteArray file = mfile.readAll();
    mfile.close();

    addFormItem(f.fileName(), file, true);

    return true;
}


void BuildHttpPost::
        addKeyValue(QString key, QString value )
{
    addFormItem(key, value.toUtf8(), false);
}


BuildHttpPost::
        operator QByteArray()
{
    QByteArray data = feedbackdata;
    data += "--" + bound + "--" + crlf;
    return data;
}


QNetworkReply *BuildHttpPost::
        send(QNetworkAccessManager* manager, QUrl url)
{
    QNetworkRequest feedbackrequest(url);
    feedbackrequest.setHeader(QNetworkRequest::ContentTypeHeader,"multipart/form-data; boundary=" + bound);

    return manager->post(feedbackrequest, *this);
}


void BuildHttpPost::
        addFormItem(QString key, QByteArray value, bool isfile )
{
    // thanks abeinoe; http://www.qtcentre.org/threads/18452-How-to-upload-file-to-HTTP-server-(POST-Method)
    feedbackdata += "--" + bound + crlf;
    feedbackdata += "Content-Disposition: form-data; name=\"" + QUrl::toPercentEncoding(key) + "\";";
    if (isfile)
        feedbackdata += "filename=\"" + QUrl::toPercentEncoding(key) + "\";";

    feedbackdata += "size=" + QString("%1").arg(value.size()) + "";

    feedbackdata += crlf + "Content-Type: application/octet" + crlf + crlf;
    feedbackdata += value;
    feedbackdata += crlf;
}



} // namespace Support
} // namespace Tools
