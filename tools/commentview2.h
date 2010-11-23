#ifndef COMMENTVIEW_H
#define COMMENTVIEW_H

#include <QWidget>

namespace Tools {

class RenderView;

class CommentView2: public QWidget
{
    Q_OBJECT
public:
    CommentView2(RenderView* render_view);
    ~CommentView2();

    double qx, qy, qz; // position

public slots:
    /// Connected in CommentController
    virtual void updatePosition();

private:
    RenderView* render_view_;
};

} // namespace Tools

#endif // COMMENTVIEW_H
