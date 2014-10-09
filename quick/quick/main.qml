import QtQuick 2.2
import QtQuick.Controls 1.2
import OpenGLUnderQML 1.0
import QtQuick.Layouts 1.1

ApplicationWindow {
    objectName: "root item "
    statusBar: StatusBar {
            visible: false
            RowLayout { Label { text: heightmap1.displayedTransformDetails } }
        }

    width: 320
    height: 480

    Chain {
        id: chain
        // common chain, the app only opens one source file at a time
        //
        // property: Chain::ptr chain
        // property: Target::ptr target, this is where modifications common for all targets is applied, such
        // as cropping

        // There might be multiple render targets (Squircle) but all showing the same point in time, open the
        // same file twice to shoe different points in time simultaneously

        // The filters selected in one squircle can be applied on another squircle
    }

    ColumnLayout {
        objectName: "row layout"
        anchors.fill: parent
        spacing: 0

        Heightmap {
            id: heightmap1
            objectName: "heightmap1"
            chain: chain
            selection: selection
            timepos: heightmap2.timepos
            xscale: heightmap2.xscale/5
            Layout.fillWidth: true
            Layout.fillHeight: true
            height: 5
        }

        Rectangle {
            id: divider
            Layout.fillWidth: true
            height: 2
            opacity: 0.0
            color: "black"

            Drag.active: dragArea.drag.active

            onYChanged: {
                heightmap1.height = y+height/2
                heightmap2.y = y+height/2
                heightmap2.height = parent.height-y-height/2
            }

            Rectangle {
                anchors.fill: parent
                anchors.topMargin: -30
                anchors.bottomMargin: -30
                opacity: 0.0

                MouseArea {
                    id: dragArea
                    anchors.fill: parent

                    drag.target: divider
                    drag.axis: Drag.YAxis
                    drag.minimumY: 10
                    drag.maximumY: divider.parent.height-10
                    hoverEnabled: true

                    onEntered: {cursorShape = Qt.SizeVerCursor;divider.opacity=0.5;}
                    onExited: {cursorShape = Qt.ArrowCursor;divider.opacity=0.0;}
                }
            }
        }

        Heightmap {
            id: heightmap2
            objectName: "heightmap2"
            chain: chain
            timepos: heightmap1.timepos
            xscale: 5*heightmap1.xscale
            displayedTransform: "waveform"
            Layout.fillWidth: true
            Layout.fillHeight: true
            visible: !heightmap2.isIOS
            height: 1
        }
    }
    /*RowLayout {
        spacing: 2
        anchors.fill: parent
        Rectangle { Layout.fillWidth: true; Layout.fillHeight: true; color: "red" }
        Rectangle { Layout.fillWidth: true; Layout.fillHeight: true; color: "green" }
        Rectangle { color: "blue"; width: 50; height: 20 }
    }*/

    DropArea {
        id: dropArea
        anchors.fill: parent

        onEntered: {
            drag.accept (Qt.CopyAction);
        }
        onDropped: {
            if (drop.hasUrls && 1===drop.urls.length)
            {
                openurl.openUrl(drop.urls[0]);
                drop.accept (Qt.CopyAction);
            }
        }
        onExited: {
        }

        OpenUrl {
            id: openurl
            chain: chain
            signal openUrl(url p)
        }
    }

    ColumnLayout {
        anchors.left: parent.left
        anchors.top : parent.top
        anchors.right : parent.right
        anchors.margins: 20

        spacing: 30

        Text {
            Layout.fillWidth: true

            visible: true
            id: text
            color: "black"
            wrapMode: Text.WordWrap
            text: "Scroll by dragging, rotate with two fingers together, zoom with two fingers in different directions. http://freq.consulting"

            SequentialAnimation on opacity {
                running: true
                NumberAnimation { to: 1; duration: 15000; easing.type: Easing.InQuad }
                NumberAnimation { to: 0; duration: 5000; easing.type: Easing.OutQuad }
            }
            SequentialAnimation on visible {
                running: true
                NumberAnimation { to: 1; duration: 20000 }
                NumberAnimation { to: 0; duration: 0 }
            }

            Rectangle {
                color: Qt.rgba(1, 1, 1, 1)
                radius: 10
                border.width: 1
                border.color: "black"
                anchors.fill: parent
                anchors.margins: -10
                z: -1

                SequentialAnimation on radius {
                    running: false // super annoying
                    NumberAnimation { to: 20; duration: 1000; easing.type: Easing.InQuad }
                    NumberAnimation { to: 10; duration: 1000; easing.type: Easing.OutQuad }
                    loops: Animation.Infinite
                }
            }
        }

        Text {
            Layout.fillWidth: true

            id: opacity_text
            visible: true
            color: "black"
            wrapMode: Text.WordWrap
            text: "Transform: " + heightmap1.displayedTransformDetails

            onTextChanged: {opacity_animation.restart();}

            SequentialAnimation on opacity {
                id: opacity_animation
                NumberAnimation { to: 1; duration: 100; easing.type: Easing.InQuad }
                NumberAnimation { to: 1; duration: 5000; easing.type: Easing.InQuad }
                NumberAnimation { to: 0; duration: 1000; easing.type: Easing.OutQuad }
            }

            Rectangle {
                color: Qt.rgba(1, 1, 1, 1)
                radius: 10
                border.width: 1
                border.color: "black"
                anchors.fill: parent
                anchors.margins: -10
                z: -1
            }
        }
    }

    Selection {
        id: selection
        filteredHeightmap: heightmap2
        renderOnHeightmap: heightmap1
    }
}
