#ifndef COMMENTCONTROLLER_H
#define COMMENTCONTROLLER_H

#include <QWidget>
#include <QPointer>
#include <QGraphicsScene>
#include "sawe/toolmodel.h"
#include "heightmap/position.h"
#include "sawe/project.h"

namespace Tools
{
    class RenderView;
    class CommentView;
    class CommentModel;
    class GraphicsView;


class CommentController: public ToolController
{
    Q_OBJECT
public:
    CommentController(QGraphicsScene* graphicsscene, RenderView* view, Sawe::Project* project, Support::ToolSelector* tool_selector, GraphicsView* graphicsview);
    ~CommentController();

    virtual void createView( ToolModelP model, ToolRepo* repo, Sawe::Project* p );

    void setComment( Heightmap::Position p, std::string text, QPointer<CommentView>* view = 0 );

    CommentView* findView( ToolModelP model );

signals:
    void enabledChanged(bool active);

private slots:
    void enableCommentAdder(bool active);
    void showComments(bool active);

private:
    virtual void changeEvent ( QEvent * event );
    virtual void mouseMoveEvent ( QMouseEvent * e );
    virtual void mousePressEvent ( QMouseEvent * e );

    CommentView* createNewComment();

    void setupGui();

    QGraphicsScene* graphicsscene_;
    RenderView* view_;
    Sawe::Project* project_;
    Support::ToolSelector* tool_selector_;
    GraphicsView* graphicsview_;

    CommentView* comment_;
    QList<QPointer<CommentView> > comments_;
};

} // namespace Tools

#endif // COMMENTCONTROLLER_H
