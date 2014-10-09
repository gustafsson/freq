import QtQuick 2.0
import OpenGLUnderQML 1.0

Squircle {
    id: squircle
    property Selection selection

    TouchNavigation {
        anchors.fill: parent
        squircle: squircle
        selection: squircle.selection

        signal touch(real x1, real y1, bool p1, real x2, real y2, bool p2, real x3, real y3, bool p3)
        signal mouseMove(real x1, real y1, bool p1)

        onIsHoldChanged: mousearea.cursorShape = isHold ? Qt.IBeamCursor : Qt.ArrowCursor

        MouseArea {
            id: mousearea
            anchors.fill: parent
            hoverEnabled: false
//            enabled: false // visible but disable; mousearea.cursorShape has effect

            onPressed: {
                console.log(("" + new Date()) + parent.parent.objectName + ": mouse press");
                mousearea.cursorShape = Qt.ClosedHandCursor;
                parent.mouseMove(mouseX, mouseY, pressed);
            }
            onPositionChanged: {
                parent.mouseMove(mouseX, mouseY, pressed);
            }
            onReleased: {
                mousearea.cursorShape = Qt.ArrowCursor;
                parent.mouseMove(mouseX, mouseY, false);
            }

//            onPressAndHold: {
//                console.log(("" + new Date()) + parent.parent.objectName + ": mouse long touch");
//            }
        }

        MultiPointTouchArea {
            anchors.fill: parent
            minimumTouchPoints: squircle.isIOS ? 1 : 2 // MouseArea handles single touch on desktops
            maximumTouchPoints: 2 // an extra just to check that it isn't used
            mouseEnabled: squircle.isIOS

            touchPoints: [
                TouchPoint { id: point1 },
                TouchPoint { id: point2 },
                TouchPoint { id: point3 }
            ]

            onGestureStarted: {
//                console.log(("" + new Date()) + parent.parent.objectName + ": touch gestures " + gesture.touchPoints.length);
//                if (gesture.touchPoints.length===2)
//                gesture.grab();
            }

            onCanceled: {
                console.log(("" + new Date()) + parent.parent.objectName + ": touch canceled " + touchPoints.length);
            }

            onPressed: {
                console.log(("" + new Date()) + parent.parent.objectName + ": touch pressed " + touchPoints.length);
            }

            onReleased: {
                console.log(("" + new Date()) + parent.parent.objectName + ": touch released " + touchPoints.length);
            }

            onTouchUpdated: {
//                console.log(("" + new Date()) + parent.parent.objectName + ": touch updated " + touchPoints.length);

                if (point3.pressed)
                {
                    parent.touch(point1.x, point1.y, point1.pressed,
                                  point2.x, point2.y, false,
                                  point3.x, point3.y, false)
                    return;
                }
                if (mousearea.pressed)
                    return;

                if (touchPoints.length === 2)
                    mousearea.cursorShape = Qt.OpenHandCursor;
                else
                    mousearea.cursorShape = Qt.ArrowCursor;

                parent.touch(point1.x, point1.y, point1.pressed,
                              point2.x, point2.y, point2.pressed,
                              point3.x, point3.y, point3.pressed)
            }
        }
    }
}
