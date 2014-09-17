import QtQuick 2.2
//import QtQuick.Controls 1.1
import OpenGLUnderQML 1.0

Item {
    visible: true

    width: 320
    height: 480

    Squircle {
        id: squircle

        signal touch(real x1, real y1, bool p1, real x2, real y2, bool p2, real x3, real y3, bool p3)
        signal mouseMove(real x1, real y1, bool p1)

        SequentialAnimation on t {
            NumberAnimation { to: 1; duration: 2500; easing.type: Easing.InQuad }
            NumberAnimation { to: 0; duration: 2500; easing.type: Easing.OutQuad }
            loops: Animation.Infinite
            running: false
        }
    }

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

    Rectangle {
        color: Qt.rgba(1, 1, 1, 0.7)
        radius: 10
        border.width: 1
        border.color: "white"
        anchors.fill: label
        anchors.margins: -10
    }

    Text {
        id: label
        color: "black"
        wrapMode: Text.WordWrap
        text: "Scroll with one finger, rotate with two fingers together, zoom with two fingers in different directions. http://freq.consulting"
        anchors.right: parent.right
        anchors.left: parent.left
        anchors.bottom: parent.bottom
        anchors.margins: 20
    }
}
