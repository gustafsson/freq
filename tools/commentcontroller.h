#ifndef COMMENTCONTROLLER_H
#define COMMENTCONTROLLER_H

#include <QWidget>
#include "toolmodel.h"
#include "heightmap/position.h"

namespace Tools
{
    class RenderView;
    class CommentView;


class CommentController: public ToolController
{
    Q_OBJECT
public:
    CommentController(RenderView* view);
    ~CommentController();

    virtual void createView( ToolModel* model, Sawe::Project* p, RenderView* r );

    void setComment( Heightmap::Position p, std::string text, CommentView** view = 0 );

signals:
    void enabledChanged(bool active);

private slots:
    void enableCommentAdder(bool active);
    void showComments(bool active);

private:
    void changeEvent ( QEvent * event );
    void mouseMoveEvent ( QMouseEvent * e );

    CommentView* createNewComment();

    void setupGui();

    RenderView* view_;

    CommentView* comment_;
    QList<CommentView*> comments_;
};

} // namespace Tools

#endif // COMMENTCONTROLLER_H
