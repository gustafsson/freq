#ifndef BRUSHFILTERSUPPORT_H
#define BRUSHFILTERSUPPORT_H

#include <QObject>

namespace Tools {
namespace Support {

class BrushFilter;

class BrushFilterSupport : public QObject
{
    Q_OBJECT
public:
    explicit BrushFilterSupport(BrushFilter*parent);

public slots:
    void release_resources();

private:
    class BrushFilter *bf_;

};

} // namespace Support
} // namespace Tools

#endif // BRUSHFILTERSUPPORT_H
