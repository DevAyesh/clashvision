"""
ClashVision Presentation Subsystem — View 7: Screenshot Dataset Manager.
Provides screenshot library and data annotation tools:
- Screenshot browser / annotation list
- GameState labeling dropdown (all 12 states)
- Bounding box annotation manager (add/remove ROIs)
- JSON dataset export for offline training and evaluation
Derived from Requirement R5 and Feature 35.
"""

from __future__ import annotations
import json
import uuid
from typing import Any, Dict, List, Optional

from clashvision.core.constants import GameState
from clashvision.core.models import DatasetAnnotation
from clashvision.storage.repository import Repository
from clashvision.ui.qt_compat import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QComboBox, QLineEdit, QTableWidget,
    QTableWidgetItem, QHeaderView, Qt
)
from clashvision.ui.styles import COLOR_NEON_GREEN, COLOR_BORDER


class DatasetView(QWidget):
    """
    Screenshot labeling and training dataset curator view.
    """

    def __init__(self, repo: Optional[Repository] = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("dataset_view")
        self.repo = repo
        self._current_boxes: List[Dict[str, Any]] = []
        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(14)

        # Header with Export Button
        header_row = QHBoxLayout()
        title_label = QLabel("Screenshot Dataset Manager & Labeling Studio")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #f1f2f6;")
        header_row.addWidget(title_label)

        header_row.addStretch()

        self.export_btn = QPushButton("💾 Export Dataset JSON")
        self.export_btn.setProperty("class", "btn_primary")
        self.export_btn.setStyleSheet(f"background-color: {COLOR_NEON_GREEN}; color: #0b1a10; font-weight: bold;")
        self.export_btn.clicked.connect(self._on_export_dataset)
        header_row.addWidget(self.export_btn)

        main_layout.addLayout(header_row)

        # Content Split Layout
        split_layout = QHBoxLayout()
        split_layout.setSpacing(16)

        # Left: Annotation List & Image Canvas Preview
        left_box = QFrame()
        left_box.setStyleSheet(
            f"background-color: #1a1c23; border: 1px solid {COLOR_BORDER}; border-radius: 8px; padding: 12px;"
        )
        left_layout = QVBoxLayout(left_box)

        left_title = QLabel("STORED DATASET ANNOTATIONS")
        left_title.setStyleSheet("font-size: 11px; font-weight: bold; color: #a4b0be;")
        left_layout.addWidget(left_title)

        self.annotations_table = QTableWidget(0, 3)
        self.annotations_table.setHorizontalHeaderLabels(["ID", "State", "Boxes Count"])
        self.annotations_table.horizontalHeader().setStretchLastSection(True)
        left_layout.addWidget(self.annotations_table, stretch=1)

        split_layout.addWidget(left_box, stretch=2)

        # Right: Annotation Editor Panel
        editor_box = QFrame()
        editor_box.setStyleSheet(
            f"background-color: #1a1c23; border: 1px solid {COLOR_BORDER}; border-radius: 8px; padding: 14px;"
        )
        editor_layout = QVBoxLayout(editor_box)
        editor_layout.setSpacing(12)

        editor_title = QLabel("ANNOTATION EDITOR")
        editor_title.setStyleSheet("font-size: 12px; font-weight: bold; color: #a4b0be;")
        editor_layout.addWidget(editor_title)

        # Image Path Input
        editor_layout.addWidget(QLabel("Image File Path:"))
        self.image_path_input = QLineEdit()
        self.image_path_input.setPlaceholderText("/path/to/screenshot.png")
        editor_layout.addWidget(self.image_path_input)

        # GameState Label Selector (12 States)
        editor_layout.addWidget(QLabel("Ground-Truth Game State:"))
        self.state_combo = QComboBox()
        for st in GameState:
            self.state_combo.addItem(st.value)
        editor_layout.addWidget(self.state_combo)

        # Bounding Box Controls
        editor_layout.addWidget(QLabel("Bounding Boxes (ROIs):"))
        self.boxes_table = QTableWidget(0, 5)
        self.boxes_table.setHorizontalHeaderLabels(["Label", "X", "Y", "Width", "Height"])
        editor_layout.addWidget(self.boxes_table)

        box_btn_row = QHBoxLayout()
        self.add_box_btn = QPushButton("+ Add Box")
        self.add_box_btn.clicked.connect(self._on_add_bounding_box)
        box_btn_row.addWidget(self.add_box_btn)

        self.clear_boxes_btn = QPushButton("Clear Boxes")
        self.clear_boxes_btn.clicked.connect(self._on_clear_boxes)
        box_btn_row.addWidget(self.clear_boxes_btn)
        editor_layout.addLayout(box_btn_row)

        # Save Annotation Button
        self.save_annotation_btn = QPushButton("Save Annotation to DB")
        self.save_annotation_btn.setStyleSheet("margin-top: 10px; font-weight: bold;")
        self.save_annotation_btn.clicked.connect(self._on_save_annotation)
        editor_layout.addWidget(self.save_annotation_btn)

        self.editor_status_label = QLabel("Ready.")
        self.editor_status_label.setStyleSheet("color: #747d8c; font-size: 11px;")
        editor_layout.addWidget(self.editor_status_label)

        editor_layout.addStretch()
        split_layout.addWidget(editor_box, stretch=2)

        main_layout.addLayout(split_layout)

    def set_repository(self, repo: Repository) -> None:
        self.repo = repo
        self.refresh_annotations()

    def _on_add_bounding_box(self) -> None:
        row = self.boxes_table.rowCount()
        self.boxes_table.setRowCount(row + 1)
        self.boxes_table.setItem(row, 0, QTableWidgetItem("resource_gold"))
        self.boxes_table.setItem(row, 1, QTableWidgetItem("100"))
        self.boxes_table.setItem(row, 2, QTableWidgetItem("120"))
        self.boxes_table.setItem(row, 3, QTableWidgetItem("150"))
        self.boxes_table.setItem(row, 4, QTableWidgetItem("40"))

    def _on_clear_boxes(self) -> None:
        self.boxes_table.setRowCount(0)

    def _on_save_annotation(self) -> None:
        img_path = self.image_path_input.text().strip() or f"/data/sample_{uuid.uuid4().hex[:6]}.png"
        st_val = self.state_combo.currentText()
        try:
            st_enum = GameState(st_val)
        except Exception:
            st_enum = GameState.UNKNOWN

        bboxes = []
        for r in range(self.boxes_table.rowCount()):
            lbl_item = self.boxes_table.item(r, 0)
            x_item = self.boxes_table.item(r, 1)
            y_item = self.boxes_table.item(r, 2)
            w_item = self.boxes_table.item(r, 3)
            h_item = self.boxes_table.item(r, 4)
            if lbl_item and x_item and y_item and w_item and h_item:
                try:
                    bboxes.append({
                        "label": lbl_item.text(),
                        "x": int(x_item.text()),
                        "y": int(y_item.text()),
                        "width": int(w_item.text()),
                        "height": int(h_item.text()),
                    })
                except ValueError:
                    pass

        ann = DatasetAnnotation(
            annotation_id=f"ann_{uuid.uuid4().hex[:8]}",
            image_path=img_path,
            state=st_enum,
            labels={"verified": True},
            bounding_boxes=bboxes,
        )

        if self.repo:
            self.repo.save_annotation(ann)
            self.editor_status_label.setText(f"Annotation {ann.annotation_id} saved successfully.")
            self.refresh_annotations()
        else:
            self.editor_status_label.setText(f"Annotation {ann.annotation_id} prepared (offline).")

    def _on_export_dataset(self) -> None:
        if self.repo:
            json_str = self.repo.export_dataset_json(output_path="dataset_export.json")
            self.editor_status_label.setText("Exported dataset to dataset_export.json")
        else:
            self.editor_status_label.setText("No database repository connected.")

    def refresh_annotations(self) -> None:
        if not self.repo:
            return
        items = self.repo.list_annotations(limit=100)
        self.annotations_table.setRowCount(len(items))
        for row, a in enumerate(items):
            self.annotations_table.setItem(row, 0, QTableWidgetItem(a.annotation_id))
            st_val = a.state.value if hasattr(a.state, "value") else str(a.state)
            self.annotations_table.setItem(row, 1, QTableWidgetItem(st_val))
            self.annotations_table.setItem(row, 2, QTableWidgetItem(str(len(a.bounding_boxes))))
