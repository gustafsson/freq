#ifndef COMMENTCONTROLLER_H
#define COMMENTCONTROLLER_H

#include <QObject>

namespace Tools
{
    class RenderView;


class CommentController: public QObject
{
    Q_OBJECT
public:
    CommentController(RenderView* view);
    ~CommentController();

private slots:
    void receiveAddComment();

private:
    void setupGui();

    RenderView* view_;
};

} // namespace Tools

#endif // COMMENTCONTROLLER_H
