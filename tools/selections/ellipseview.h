#ifndef ELLIPSEVIEW_H
#define ELLIPSEVIEW_H

#include <QObject>

namespace Tools { namespace Selections
{

class EllipseModel;

class EllipseView: public QObject
{
    Q_OBJECT
public:
    EllipseView(EllipseModel* model);
    ~EllipseView();

    void drawSelectionCircle();

    bool visible, enabled;

public slots:
    /// Connected in EllipseController
    virtual void draw();

private:
    friend class EllipseController;
    EllipseModel* model_;
};

}} // namespace Tools::Selections

#endif // ELLIPSEVIEW_H
