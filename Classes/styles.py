class Styles:
    def __init__(self):
        self.colors = {
            "gray": "#333333",
            "lightGray": "#777777",
            "borderGray": "#999999",
            "secondaryGreen": "#1ECF55",
            "green": "#13FF00",
        }
        self.styleSheet = """
        * {
            background-color: @gray; 
            color: white;
            border: none;
            font-size: 16px;
        }

        #totalBackground {
            background-color: @secondaryGray;
        }

        QTabWidget::pane { /* The tab widget frame */
            border-top: 1px solid @borderGray;
        }

        QTabBar {
            background-color: @secondaryGray;
        }

        QTabBar::tab {
            background-color: @gray;
            border: 1px solid @borderGray;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            min-width: 8ex;
            padding: 2px;
        }

        QTabBar::tab:selected, QTabBar::tab:hover {
            background: @lightGray;
        }

        QTabBar::tab:selected {
            border-color: @borderGray; 
        }

        QLabel {
            font-size: 16px; 
            padding-bottom: 3px; 
            padding-top: 3px;
        } 

        QCheckBox::indicator {
            width: 18px;
            height: 18px;
            border-radius: 2px;
            border: 1px solid @borderGray;
            background: @gray;
        }

        QCheckBox::indicator:hover {
            background: @lightGray;
        }

        QCheckBox::indicator:checked {
            background: @secondaryGreen;
        }

        QRadioButton::indicator {
            width: 18px;
            height: 18px;
            border-radius: 9px;
            border: 1px solid @borderGray;
            background: @gray;
        }

        QRadioButton::indicator:hover {
            background: @lightGray;
        }

        QRadioButton::indicator:checked {
            background: @secondaryGreen;
        }

        QPushButton {
            border-radius: 10px; 
            border: 1px solid @lightGray;
            width: 250px;
            height: 50px;
            background-color: @gray;
        }

        QPushButton:hover {
            background-color: @lightGray;
        }

        QLineEdit{
            padding: 5px;
            border: 1px solid @borderGray;
            border-radius: 5px;
        }

        QSlider::handle {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #b4b4b4, stop:1 #8f8f8f);
            border: 1px solid #5c5c5c;
            width: 18px;
            margin: -2px 0; /* handle is placed by default on the contents rect of the groove. Expand outside the groove */
            border-radius: 3px;
        }

        QSlider::handle:hover {
            background: @lightGray;
        }
        
        #fingerJointSelector {
            border: 1px solid @borderGray;
            border-radius: 5px;
        }

        QDoubleSpinBox{
            padding: 7px;
            border: 1px solid @borderGray;
            border-radius: 5px;
        }

        QDoubleSpinBox::up-button{
            margin: 3px;
        }

        QDoubleSpinBox::down-button{
            margin: 3px;
        }

        QDoubleSpinBox::up-button:hover {
            background-color: @lightGray;
        }

        QDoubleSpinBox::up-button:pressed {
            background-color: @lightGray;
        }

        QDoubleSpinBox::down-button:hover {
            background-color: @lightGray;
        }

        QDoubleSpinBox::down-button:pressed {
            background-color: @lightGray;
        }
        """

        self.genStyles()

    def genStyles(self):
        for key, val in self.colors.items():
            self.styleSheet = self.styleSheet.replace(f"@{key}", val)

    def getStyles(self):
        return self.styleSheet
