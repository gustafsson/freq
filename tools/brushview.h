#ifndef BRUSHVIEW_H
#define BRUSHVIEW_H

#include <QObject>
#include "heightmap/position.h"
#include "heightmap/reference.h"
#include "brushmodel.h"

namespace Tools {

class BrushView: public QObject
{
    Q_OBJECT
public:
    BrushView(BrushModel* model);
    ~BrushView();

    bool enabled;
    Gauss gauss;

public slots:
    /// Connected in BrushController
    virtual void draw();

private:
    friend class BrushController;
    BrushModel* model_;

    void drawCircle();
};

} // namespace Tools

#endif // BRUSHVIEW_H
