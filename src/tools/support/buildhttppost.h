#ifndef BUILDHTTPPOST_H
#define BUILDHTTPPOST_H

#include <QByteArray>
#include <QFileInfo>
#include <QUrl>

class QNetworkAccessManager;
class QNetworkReply;

namespace Tools {
namespace Support {

class BuildHttpPost final
{
public:
    BuildHttpPost();
    ~BuildHttpPost();


    /**
      Appends a file.
      @return false if 'file' could not be opened.
      */
    bool addFile(QFileInfo file );

    void addKeyValue(QString key, QString value );

    QString boundary() { return bound; }

    operator QByteArray();

    QNetworkReply* send(QNetworkAccessManager*, QUrl url);

private:
    void addFormItem(QString key, QByteArray value, bool isfile );

    QString crlf, bound;
    QByteArray feedbackdata;
};

} // namespace Support
} // namespace Tools

#endif // BUILDHTTPPOST_H
