#ifndef QTAUDIOFILE_H
#define QTAUDIOFILE_H

#include "signal/operation.h"
#include <QUrl>
#include <QAudioDecoder>

/**
 * @brief The QtAudiofile class should decode audio files.
 *
 * QtAudiofile relies on Qt Multimedia backends through QAudioDecoder which is
 * only supported on windows (mediafoundation backend) and linux (gstreamer
 * backend).
 */
class QtAudiofile : public QObject, public Signal::OperationDesc
{
    Q_OBJECT
public:
    QtAudiofile(QUrl file);

    Signal::Interval requiredInterval( const Signal::Interval& I, Signal::Interval* expectedOutput ) const override;
    Signal::Interval affectedInterval( const Signal::Interval& I ) const override;
    Signal::OperationDesc::ptr copy() const override;
    Signal::Operation::ptr createOperation(Signal::ComputingEngine* engine=0) const override;
    Extent extent() const override;
    QString toString() const override;
    bool operator==(const OperationDesc& d) const override;

public slots:
    void durationChanged(qint64 duration);

private:
    QUrl url;
    QAudioDecoder decoder;
    qint64 old_duration = 0;
};

#endif // QTAUDIOFILE_H
