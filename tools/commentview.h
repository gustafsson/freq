#ifndef COMMENTVIEW_H
#define COMMENTVIEW_H

#include <QObject>

namespace Tools {

class CommentModel;

class CommentView: public QObject
{
    Q_OBJECT
public:
    CommentView(CommentModel* model);
    ~CommentView();

    bool enabled;

public slots:
    /// Connected in CommentController
    virtual void draw();

private:
    friend class CommentController;
    CommentModel* model_;
};

} // namespace Tools

#endif // COMMENTVIEW_H
