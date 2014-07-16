#ifndef COMMENTCONTROLLER_H
#define COMMENTCONTROLLER_H

#include <QWidget>
#include <QPointer>
#include <QGraphicsScene>
#include "sawe/toolmodel.h"
#include "heightmap/position.h"

namespace Tools
{
    class RenderView;
    class CommentView;
    class CommentModel;


class CommentController: public ToolController
{
    Q_OBJECT
public:
    CommentController(QGraphicsScene* graphicsscene, RenderView* view);
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

    CommentView* comment_;
    QList<QPointer<CommentView> > comments_;
};

} // namespace Tools

#endif // COMMENTCONTROLLER_H
