import QtQuick 2.0
import OpenGLUnderQML 1.0

Squircle {
    id: squircle

//    property Chain chain: squircle.chain
//    property real t: squircle.t

    signal touch(real x1, real y1, bool p1, real x2, real y2, bool p2, real x3, real y3, bool p3)
    signal mouseMove(real x1, real y1, bool p1)

    MouseArea {
        anchors.fill: parent
        onMouseYChanged: onMouseXChanged
        onMouseXChanged: squircle.mouseMove(mouseX, mouseY, pressed)
    }

    MultiPointTouchArea {
        anchors.fill: parent
        minimumTouchPoints: 1
        maximumTouchPoints: 3 // an extra just to check that it isn't used
        mouseEnabled: false

        touchPoints: [
            TouchPoint { id: point1 },
            TouchPoint { id: point2 },
            TouchPoint { id: point3 }
        ]

        onTouchUpdated: {
            if (point3.pressed)
                return false;
            return squircle.touch(point1.sceneX, point1.sceneY, point1.pressed,
                                  point2.sceneX, point2.sceneY, point2.pressed,
                                  point3.sceneX, point3.sceneY, point3.pressed)
        }
    }
}
