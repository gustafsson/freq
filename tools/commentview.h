#ifndef COMMENTVIEW_H
#define COMMENTVIEW_H

#include <QWidget>

namespace Tools {

class RenderView;

class CommentView: public QWidget
{
    Q_OBJECT
public:
    CommentView(RenderView* render_view);
    ~CommentView();

    double qx, qy, qz; // position

public slots:
    /// Connected in CommentController
    virtual void updatePosition();

private:
    RenderView* render_view_;
};

} // namespace Tools

#endif // COMMENTVIEW_H
