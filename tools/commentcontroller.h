#ifndef COMMENTCONTROLLER_H
#define COMMENTCONTROLLER_H

#include <QWidget>

namespace Tools
{
    class RenderView;
    class CommentView;


class CommentController: public QWidget
{
    Q_OBJECT
public:
    CommentController(RenderView* view);
    ~CommentController();

signals:
    void enabledChanged(bool active);

private slots:
    void enableCommentAdder(bool active);
    void showComments(bool active);

private:
    void changeEvent ( QEvent * event );
    void mouseMoveEvent ( QMouseEvent * e );

    class CommentView* createNewComment();

    void setupGui();

    RenderView* view_;

    CommentView* comment_;
    QList<CommentView*> comments_;
};

} // namespace Tools

#endif // COMMENTCONTROLLER_H
