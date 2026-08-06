"""Estilo visual centralizado do Dither App."""

DARK_STYLESHEET = """
QMainWindow, QWidget { background: #1e1f22; color: #e6e6e6; font: 14px 'Segoe UI'; }
QPushButton { background: #3a6ea5; border: 1px solid #4b82ba; border-radius: 5px; padding: 8px 14px; color: white; }
QPushButton:hover { background: #477fb5; }
QPushButton:pressed { background: #315d8a; }
QPushButton:disabled { background: #35383d; border-color: #35383d; color: #8a8d91; }
QComboBox, QSpinBox, QSlider { background: #202124; border: 1px solid #50535a; border-radius: 4px; padding: 5px; }
QComboBox:focus, QSpinBox:focus { border: 1px solid #5b9bd5; }
QCheckBox { spacing: 8px; padding: 3px 0; }
QCheckBox::indicator { width: 16px; height: 16px; }
QCheckBox::indicator:checked { background: #3a6ea5; border: 1px solid #5b9bd5; }
QSlider::groove:horizontal { height: 5px; background: #3b3e44; border-radius: 2px; }
QSlider::handle:horizontal { width: 15px; margin: -5px 0; background: #5b9bd5; border-radius: 7px; }
QStatusBar { background: #292b2f; border-top: 1px solid #3b3e44; }
QProgressBar { border: 1px solid #50535a; border-radius: 3px; background: #202124; }
QProgressBar::chunk { background: #5b9bd5; }
"""
