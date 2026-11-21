
import sys
import os
import math
import random
import logging
import faulthandler

if __name__ == "__main__":
    log_directory = os.path.join(
        os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
        "CollageMaker"
    )
    os.makedirs(log_directory, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_directory, "application.log"),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    crash_log = open(os.path.join(log_directory, "crash.log"), "a")
    faulthandler.enable(crash_log)

    def report_unhandled_exception(exception_type, exception, traceback):
        logging.critical(
            "Unhandled exception",
            exc_info=(exception_type, exception, traceback)
        )
        sys.__excepthook__(exception_type, exception, traceback)

    sys.excepthook = report_unhandled_exception

import cv2
import numpy as np

from PIL import Image

from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGroupBox,
    QPushButton,
    QLineEdit,
    QFileDialog,
    QLabel,
    QMessageBox,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsRectItem,
    QGraphicsPixmapItem,
    QCheckBox,
    QDoubleSpinBox,
    QSpinBox,
    QScrollArea,
    QToolButton,
    QMenu,
    QWidgetAction,
    QColorDialog,
)

from PyQt6.QtGui import (
    QBrush,
    QColor,
    QPen,
    QPixmap,
    QPainter,
    QImage,
    QImageReader,
    QTransform,
)

from PyQt6.QtCore import QRectF, Qt, QThread, QSize


# ============================================================
# CONFIGURATION AND DEFAULTS
# ============================================================

DEFAULT_DPI = 300
PREVIEW_MAX_SIZE = 1024

DEFAULT_LAYOUT_SETTINGS = {
    "equal_row_heights": True,
    "ratio_weight": 1.0,
    "count_weight": 0.8,
    "row_search_range": 2,
    "randomize_images": True,
}

SUPPORTED_IMAGES = (
    "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"
)


# ============================================================
# FACE DETECTION
# ============================================================

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# ============================================================
# IMAGE MODEL
# ============================================================

class ImgBox:

    def __init__(self, filepath):

        self.path = filepath

        with Image.open(filepath) as im:
            self.w, self.h = im.size
            im.draft("RGB", (800, 800))
            im.thumbnail((800, 800))
            self.img = im.convert("RGB").copy()

        if self.w <= 0 or self.h <= 0:
            raise ValueError(
                f"Invalid image: {filepath}"
            )

        self.ratio = self.w / self.h


# ============================================================
# SMART FACE-AWARE CROPPING
# ============================================================

def smart_crop_fill(
    img: Image.Image,
    target_w: int,
    target_h: int
):

    crop_coords = compute_crop_coords(img, target_w, target_h)
    resized = img.crop(crop_coords).resize(
        (target_w, target_h),
        Image.Resampling.LANCZOS
    )
    return resized, crop_coords


def compute_crop_coords(img, target_w, target_h):

    if target_w <= 0 or target_h <= 0:
        raise ValueError("Invalid crop dimensions")

    w, h = img.size
    detection_image = img.copy()
    detection_image.thumbnail((800, 800))
    img_np = np.asarray(detection_image.convert("RGB"))
    detection_scale_x = w / detection_image.width
    detection_scale_y = h / detection_image.height

    target_ratio = target_w / target_h

    gray = cv2.cvtColor(
        img_np,
        cv2.COLOR_RGB2GRAY
    )

    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(max(20, round(60 / detection_scale_x)),
             max(20, round(60 / detection_scale_y)))
    )

    # Find crop center
    if len(faces) > 0:

        x_min = min(x for x, y, fw, fh in faces)
        y_min = min(y for x, y, fw, fh in faces)

        x_max = max(x + fw for x, y, fw, fh in faces)
        y_max = max(y + fh for x, y, fw, fh in faces)

        cx = round((x_min + x_max) / 2 * detection_scale_x)
        cy = round((y_min + y_max) / 2 * detection_scale_y)

    else:

        cx = w // 2
        cy = h // 2

    # Determine crop size
    if w / h > target_ratio:

        crop_h = h
        crop_w = max(1, int(h * target_ratio))

    else:

        crop_w = w
        crop_h = max(1, int(w / target_ratio))

    # Clamp crop position
    left = max(
        0,
        min(cx - crop_w // 2, w - crop_w)
    )

    top = max(
        0,
        min(cy - crop_h // 2, h - crop_h)
    )

    right = left + crop_w
    bottom = top + crop_h

    crop_coords = (
        left,
        top,
        right,
        bottom
    )

    return crop_coords


# ============================================================
# DYNAMIC PROGRAMMING: BALANCED ROW DISTRIBUTION
# ============================================================

def compute_smart_rows(
    boxes,
    canvas_w,
    canvas_h,
    settings=None
):

    settings = settings or DEFAULT_LAYOUT_SETTINGS

    ratio_weight = settings["ratio_weight"]
    count_weight = settings["count_weight"]
    search_range = settings["row_search_range"]

    n = len(boxes)

    if n == 0:
        return [], []

    if n == 1:
        return [boxes], [canvas_h]

    ratios = np.array(
        [b.ratio for b in boxes],
        dtype=float
    )

    # --------------------------------------------------------
    # 1. Estimate ideal row count
    # --------------------------------------------------------

    total_ratio = ratios.sum()

    ideal_rows = math.sqrt(
        total_ratio * canvas_h / canvas_w
    )

    center_rows = max(
        1,
        min(n, round(ideal_rows))
    )

    min_rows = max(
        1,
        center_rows - search_range
    )

    max_rows = min(
        n,
        center_rows + search_range
    )

    # Prefix sum
    prefix = np.concatenate(
        ([0.0], np.cumsum(ratios))
    )

    best_solution = None
    best_cost = float("inf")

    # --------------------------------------------------------
    # 2. Evaluate possible row counts
    # --------------------------------------------------------

    for num_rows in range(
        min_rows,
        max_rows + 1
    ):

        target_count = n / num_rows

        target_ratio = (
            canvas_w * num_rows / canvas_h
        )

        dp = np.full(
            (num_rows + 1, n + 1),
            np.inf
        )

        parent = np.full(
            (num_rows + 1, n + 1),
            -1,
            dtype=int
        )

        dp[0, 0] = 0.0

        # Dynamic programming
        for r in range(1, num_rows + 1):

            for j in range(r, n + 1):

                for i in range(r - 1, j):

                    count = j - i

                    row_ratio = (
                        prefix[j] - prefix[i]
                    )

                    if row_ratio <= 0:
                        continue

                    # Row height consistency
                    ratio_error = math.log(
                        row_ratio / target_ratio
                    ) ** 2

                    # Image count balance
                    count_error = (
                        (count - target_count) /
                        target_count
                    ) ** 2

                    cost = (
                        ratio_weight * ratio_error +
                        count_weight * count_error
                    )

                    total_cost = (
                        dp[r - 1, i] + cost
                    )

                    if total_cost < dp[r, j]:

                        dp[r, j] = total_cost
                        parent[r, j] = i

        final_cost = dp[num_rows, n]

        if not np.isfinite(final_cost):
            continue

        normalized_cost = (
            final_cost / num_rows
        )

        # Slight preference for estimated row count
        row_count_penalty = (
            0.15 *
            math.log(num_rows / ideal_rows) ** 2
        )

        total_cost = (
            normalized_cost +
            row_count_penalty
        )

        # ----------------------------------------------------
        # 3. Reconstruct rows
        # ----------------------------------------------------

        if total_cost < best_cost:

            groups = []

            j = n

            for r in range(num_rows, 0, -1):

                i = parent[r, j]

                if i < 0:
                    groups = []
                    break

                groups.append(
                    boxes[i:j]
                )

                j = i

            if not groups:
                continue

            groups.reverse()

            best_cost = total_cost
            best_solution = groups

    if best_solution is None:

        best_solution = [
            [b] for b in boxes
        ]

    rows = best_solution

    # --------------------------------------------------------
    # 4. Calculate natural row heights
    # --------------------------------------------------------

    natural_heights = []

    for row in rows:

        row_ratio_sum = sum(
            b.ratio for b in row
        )

        natural_heights.append(
            canvas_w / row_ratio_sum
        )

    natural_heights = np.array(
        natural_heights,
        dtype=float
    )

    # --------------------------------------------------------
    # 5. Equal-height or natural-height mode
    # --------------------------------------------------------

    if settings["equal_row_heights"]:

        row_heights = np.full(
            len(rows),
            canvas_h / len(rows),
            dtype=float
        )

    else:

        scale = (
            canvas_h / natural_heights.sum()
        )

        row_heights = (
            natural_heights * scale
        )

    # --------------------------------------------------------
    # 6. Exact integer height correction
    # --------------------------------------------------------

    row_heights = np.floor(
        row_heights
    ).astype(int)

    row_heights[-1] += (
        canvas_h - int(row_heights.sum())
    )

    return rows, row_heights.tolist()


# ============================================================
# COLLAGE GENERATION
# ============================================================

def make_intelligent_collage(
    image_paths,
    output_path,
    width_in,
    height_in,
    dpi=300,
    save_file=False,
    settings=None
):

    settings = settings or DEFAULT_LAYOUT_SETTINGS

    if not image_paths:
        raise ValueError("No images selected")

    if width_in <= 0 or height_in <= 0 or dpi <= 0:
        raise ValueError("Invalid canvas dimensions")

    canvas_w = int(width_in * dpi)
    canvas_h = int(height_in * dpi)

    if canvas_w < 1 or canvas_h < 1:
        raise ValueError("Canvas dimensions must be at least one pixel")

    collage = Image.new(
        "RGB",
        (canvas_w, canvas_h),
        "white"
    ) if save_file and output_path else None

    # Load images
    boxes = [
        ImgBox(p) for p in image_paths
    ]

    # KEEP RANDOM SHUFFLE AS REQUESTED
    if settings["randomize_images"]:
        random.shuffle(boxes)

    # Calculate optimized rows
    rows, row_heights = compute_smart_rows(
        boxes,
        canvas_w,
        canvas_h,
        settings
    )

    collage_details = []

    y = 0

    # --------------------------------------------------------
    # Place images row by row
    # --------------------------------------------------------

    for row_index, (row_imgs, row_h) in enumerate(
        zip(rows, row_heights)
    ):

        total_ratio = sum(
            b.ratio for b in row_imgs
        )

        x = 0

        for i, b in enumerate(row_imgs):

            w = int(
                canvas_w *
                (b.ratio / total_ratio)
            )

            # Exact row width
            if i == len(row_imgs) - 1:
                w = canvas_w - x

            if collage is not None:
                with Image.open(b.path) as original:
                    cropped_img, crop_coords = smart_crop_fill(
                        original.convert("RGB"), w, row_h
                    )
                collage.paste(cropped_img, (x, y))
            else:
                preview_crop = compute_crop_coords(b.img, w, row_h)
                crop_coords = (
                    round(preview_crop[0] * b.w / b.img.width),
                    round(preview_crop[1] * b.h / b.img.height),
                    round(preview_crop[2] * b.w / b.img.width),
                    round(preview_crop[3] * b.h / b.img.height)
                )

            collage_details.append({

                "image_path": b.path,

                "image": b.img,
                "original_size": (b.w, b.h),

                "canvas_position": (
                    x,
                    y,
                    x + w,
                    y + row_h
                ),

                "crop_position": crop_coords,

                "row_index": row_index

            })

            x += w

        y += row_h

    # Optional direct save
    if collage is not None:

        collage.save(
            output_path,
            dpi=(dpi, dpi),
            quality=95
        )

    return collage_details, rows, row_heights


# ============================================================
# MOVABLE IMAGE ITEM
# ============================================================

class ImageBox(QGraphicsPixmapItem):

    def __init__(
        self,
        pixmap,
        position,
        scale_factor
    ):

        super().__init__(pixmap)

        self.setFlags(
            QGraphicsPixmapItem.GraphicsItemFlag.ItemIsMovable |
            QGraphicsPixmapItem.GraphicsItemFlag.ItemIsSelectable |
            QGraphicsPixmapItem.GraphicsItemFlag.ItemIsFocusable
        )

        self.setTransformationMode(
            Qt.TransformationMode.SmoothTransformation
        )

        self.setPos(
            position[0],
            position[1]
        )

        self.setScale(scale_factor)


# ============================================================
# COLLAGE VIEWER
# ============================================================

class CanvasViewer(QGraphicsView):

    def __init__(
        self,
        width,
        height,
        boxes
    ):

        super().__init__()

        self.setWindowTitle(
            "Intelligent Collage Editor"
        )

        self.scene = QGraphicsScene(self)

        self.setScene(self.scene)

        self.scene.setBackgroundBrush(
            QBrush(QColor("white"))
        )

        self.scene.setSceneRect(
            0,
            0,
            width,
            height
        )

        self._zoom = 1.0
        self.image_items = []
        self.border_items = []

        # ----------------------------------------------------
        # Create image boxes
        # ----------------------------------------------------

        for item in boxes:

            x1, y1, x2, y2 = (
                item["canvas_position"]
            )

            grid_width = x2 - x1
            grid_height = y2 - y1

            # Grid clipping rectangle
            grid = QGraphicsRectItem(
                QRectF(
                    x1,
                    y1,
                    grid_width,
                    grid_height
                )
            )

            grid.setPen(
                QPen(Qt.PenStyle.NoPen)
            )

            grid.setBrush(
                QBrush(QColor("white"))
            )

            grid.setFlag(
                QGraphicsRectItem.GraphicsItemFlag.ItemClipsChildrenToShape,
                True
            )

            self.scene.addItem(grid)

            # Crop coordinates
            crop_x1, crop_y1, crop_x2, crop_y2 = (
                item["crop_position"]
            )

            crop_width = crop_x2 - crop_x1
            crop_height = crop_y2 - crop_y1

            original_width, original_height = item.get(
                "original_size", item["image"].size
            )

            # Scale original image to fill grid
            scale = max(
                grid_width / crop_width,
                grid_height / crop_height
            )

            # Position in scene coordinates
            offset_x = x1 - crop_x1 * scale
            offset_y = y1 - crop_y1 * scale

            # Grid item sits at pos (0,0) with a scene-coordinate rect,
            # so child coordinates are scene coordinates.
            local_x = offset_x
            local_y = offset_y

            reader = QImageReader(item["image_path"])
            preview_size = QSize(original_width, original_height)
            if max(original_width, original_height) > PREVIEW_MAX_SIZE:
                preview_size.scale(
                    PREVIEW_MAX_SIZE, PREVIEW_MAX_SIZE,
                    Qt.AspectRatioMode.KeepAspectRatio
                )
            reader.setScaledSize(preview_size)
            preview = reader.read()
            if preview.isNull():
                raise ValueError(
                    f"Cannot load image: {item['image_path']}: {reader.errorString()}"
                )
            pixmap = QPixmap.fromImage(preview)

            img_item = ImageBox(
                pixmap,
                (local_x, local_y),
                scale
            )

            img_item.setParentItem(grid)
            img_item.setZValue(1)
            img_item.setTransform(QTransform.fromScale(
                original_width / pixmap.width(),
                original_height / pixmap.height()
            ))
            img_item.source_path = item["image_path"]
            self.image_items.append(img_item)

            # Border overlay
            border = QGraphicsRectItem(
                QRectF(
                    x1,
                    y1,
                    grid_width,
                    grid_height
                )
            )

            pen = QPen(
                QColor("red")
            )

            pen.setWidth(2)

            border.setPen(pen)

            border.setBrush(
                QBrush(Qt.BrushStyle.NoBrush)
            )

            border.setZValue(10)
            border.setVisible(False)
            border.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

            self.scene.addItem(border)
            self.border_items.append(border)

        # Rendering settings
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing |
            QPainter.RenderHint.SmoothPixmapTransform
        )

        self.setDragMode(
            QGraphicsView.DragMode.RubberBandDrag
        )

        self.setViewportUpdateMode(
            QGraphicsView.ViewportUpdateMode.FullViewportUpdate
        )

    # --------------------------------------------------------
    # Initial fit and resize
    # --------------------------------------------------------

    def showEvent(self, event):

        super().showEvent(event)

        if self._zoom == 1.0:

            self.fitInView(
                self.scene.sceneRect(),
                Qt.AspectRatioMode.KeepAspectRatio
            )

    def resizeEvent(self, event):

        super().resizeEvent(event)

        if self._zoom == 1.0:

            self.fitInView(
                self.scene.sceneRect(),
                Qt.AspectRatioMode.KeepAspectRatio
            )

    # --------------------------------------------------------
    # Mouse wheel zoom
    # --------------------------------------------------------

    def wheelEvent(self, event):

        zoom_in = event.angleDelta().y() > 0

        factor = 1.1 if zoom_in else 0.9

        # Ctrl + Wheel: Zoom selected images
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:

            selected_items = (
                self.scene.selectedItems()
            )

            image_items = [
                item for item in selected_items
                if isinstance(item, ImageBox)
            ]

            if image_items:

                for item in image_items:

                    item.setScale(
                        item.scale() * factor
                    )

                event.accept()
                return

            super().wheelEvent(event)
            return

        # Normal wheel: Zoom canvas
        self._zoom *= factor

        self.scale(
            factor,
            factor
        )

        event.accept()

    # --------------------------------------------------------
    # Reset zoom
    # --------------------------------------------------------

    def set_cell_lines(self, visible, thickness, color):
        pen = QPen(QColor(color))
        pen.setWidthF(thickness)
        for border in self.border_items:
            border.setPen(pen)
            border.setVisible(visible)

    def reset_zoom(self):

        self.resetTransform()

        self._zoom = 1.0

        self.fitInView(
            self.scene.sceneRect(),
            Qt.AspectRatioMode.KeepAspectRatio
        )

    # --------------------------------------------------------
    # Save current edited scene
    # --------------------------------------------------------

    def save_scene_to_jpg(
        self,
        output_path,
        dpi=300
    ):

        rect = self.scene.sceneRect()

        width = int(rect.width())
        height = int(rect.height())

        image = QImage(
            width,
            height,
            QImage.Format.Format_RGB32
        )

        image.fill(
            Qt.GlobalColor.white
        )

        painter = QPainter(image)

        painter.setRenderHint(
            QPainter.RenderHint.Antialiasing
        )

        painter.setRenderHint(
            QPainter.RenderHint.SmoothPixmapTransform
        )

        try:
            for item in self.image_items:
                source = QImage(item.source_path)
                if source.isNull():
                    raise ValueError(f"Cannot load original image: {item.source_path}")
                painter.save()
                painter.setClipRect(item.parentItem().sceneBoundingRect())
                painter.setWorldTransform(item.sceneTransform())
                painter.drawImage(
                    QRectF(0, 0, item.pixmap().width(), item.pixmap().height()),
                    source
                )
                painter.restore()
                del source

            for border in self.border_items:
                if not border.isVisible():
                    continue
                painter.setPen(border.pen())
                painter.setBrush(border.brush())
                painter.drawRect(border.rect())
        finally:
            painter.end()

        rgba_image = image.convertToFormat(
            QImage.Format.Format_RGBA8888
        )

        ptr = rgba_image.bits()

        ptr.setsize(
            rgba_image.sizeInBytes()
        )

        pil_img = Image.frombytes(
            "RGBA",
            (width, height),
            bytes(ptr)
        ).convert("RGB")

        pil_img.save(
            output_path,
            "JPEG",
            quality=100,
            subsampling=0,
            dpi=(dpi, dpi)
        )


# ============================================================
# MAIN COLLAGE EDITOR WINDOW
# ============================================================

class CollageGenerationWorker(QThread):

    def __init__(self, image_paths, width_in, height_in, dpi, settings, parent):
        super().__init__(parent)
        self.image_paths = list(image_paths)
        self.width_in = width_in
        self.height_in = height_in
        self.dpi = dpi
        self.settings = dict(settings)
        self.result = None
        self.error = None

    def run(self):
        try:
            self.result = make_intelligent_collage(
                self.image_paths, "", self.width_in, self.height_in,
                dpi=self.dpi, settings=self.settings
            )
        except Exception as exception:
            logging.exception("Collage generation failed")
            self.error = str(exception)


class CollageEditorWindow(QWidget):

    def __init__(self):

        super().__init__()

        self.dpi = DEFAULT_DPI

        self.image_paths = []

        self.viewer = None

        self.current_rows = []
        self.current_row_heights = []
        self.generation_worker = None

        self.setup_ui()

    # ========================================================
    # UI SETUP
    # ========================================================

    def setup_ui(self):

        self.setWindowTitle(
            "Intelligent Photo Collage Editor"
        )

        self.resize(1400, 950)

        main_layout = QVBoxLayout(self)

        # ----------------------------------------------------
        # TOP CONTROLS
        # ----------------------------------------------------

        controls = QHBoxLayout()

        self.select_btn = QPushButton(
            "Select Images"
        )

        self.select_btn.clicked.connect(
            self.select_images
        )

        self.width_input = QLineEdit("12")
        self.width_input.setFixedWidth(75)

        self.height_input = QLineEdit("8")
        self.height_input.setFixedWidth(75)

        self.create_btn = QPushButton(
            "Generate Collage"
        )

        self.create_btn.clicked.connect(
            self.generate_collage
        )

        self.reset_btn = QPushButton(
            "Reset Zoom"
        )

        self.reset_btn.clicked.connect(
            self.reset_view
        )

        self.save_btn = QPushButton(
            "Save JPEG"
        )

        self.save_btn.clicked.connect(
            self.save_collage
        )

        controls.addWidget(self.select_btn)

        controls.addWidget(
            QLabel("Width (in):")
        )

        controls.addWidget(self.width_input)

        controls.addWidget(
            QLabel("Height (in):")
        )

        controls.addWidget(self.height_input)

        controls.addWidget(self.create_btn)
        controls.addWidget(self.reset_btn)
        controls.addWidget(self.save_btn)

        main_layout.addLayout(controls)

        # ----------------------------------------------------
        # IMAGE INFORMATION
        # ----------------------------------------------------

        self.info_label = QLabel(
            "No images selected"
        )

        main_layout.addWidget(
            self.info_label
        )

        # ----------------------------------------------------
        # ADVANCED LAYOUT SETTINGS
        # ----------------------------------------------------

        settings_group = QGroupBox(
            "Advanced Layout Settings"
        )

        settings_layout = QFormLayout(
            settings_group
        )

        # Equal row heights
        self.equal_height_check = QCheckBox()

        self.equal_height_check.setChecked(
            DEFAULT_LAYOUT_SETTINGS["equal_row_heights"]
        )

        self.equal_height_check.setToolTip(
            "ON: All rows have the same height.\n"
            "OFF: Row heights follow their natural image proportions."
        )

        settings_layout.addRow(
            "Equal row heights:",
            self.equal_height_check
        )

        # Row height consistency weight
        self.ratio_weight_input = QDoubleSpinBox()

        self.ratio_weight_input.setRange(
            0.0,
            10.0
        )

        self.ratio_weight_input.setSingleStep(0.1)
        self.ratio_weight_input.setDecimals(2)

        self.ratio_weight_input.setValue(
            DEFAULT_LAYOUT_SETTINGS["ratio_weight"]
        )

        self.ratio_weight_input.setToolTip(
            "Higher value prioritizes consistent natural row heights."
        )

        settings_layout.addRow(
            "Row height consistency weight:",
            self.ratio_weight_input
        )

        # Image count balance weight
        self.count_weight_input = QDoubleSpinBox()

        self.count_weight_input.setRange(
            0.0,
            10.0
        )

        self.count_weight_input.setSingleStep(0.1)
        self.count_weight_input.setDecimals(2)

        self.count_weight_input.setValue(
            DEFAULT_LAYOUT_SETTINGS["count_weight"]
        )

        self.count_weight_input.setToolTip(
            "Higher value prioritizes equal image counts per row."
        )

        settings_layout.addRow(
            "Image count balance weight:",
            self.count_weight_input
        )

        # Row search range
        self.search_range_input = QSpinBox()

        self.search_range_input.setRange(
            0,
            10
        )

        self.search_range_input.setValue(
            DEFAULT_LAYOUT_SETTINGS["row_search_range"]
        )

        self.search_range_input.setToolTip(
            "Number of row counts above and below the estimated "
            "ideal row count to evaluate."
        )

        settings_layout.addRow(
            "Row search range:",
            self.search_range_input
        )

        # Randomize image order
        self.randomize_check = QCheckBox()

        self.randomize_check.setChecked(
            DEFAULT_LAYOUT_SETTINGS["randomize_images"]
        )

        self.randomize_check.setToolTip(
            "Shuffle images before arranging them."
        )

        settings_layout.addRow(
            "Randomize image order:",
            self.randomize_check
        )

        self.cell_lines_check = QCheckBox()
        self.cell_lines_check.setChecked(False)
        settings_layout.addRow("Show cell lines:", self.cell_lines_check)

        self.cell_line_thickness_input = QDoubleSpinBox()
        self.cell_line_thickness_input.setRange(0.5, 50.0)
        self.cell_line_thickness_input.setSingleStep(0.5)
        self.cell_line_thickness_input.setDecimals(1)
        self.cell_line_thickness_input.setSuffix(" px")
        self.cell_line_thickness_input.setValue(2.0)
        settings_layout.addRow("Cell line thickness:", self.cell_line_thickness_input)

        self.cell_line_color = QColor("red")
        self.cell_line_color_btn = QToolButton()
        self.cell_line_color_btn.setFixedSize(28, 28)
        self.cell_line_color_btn.setAccessibleName("Cell line color")
        self.cell_line_color_btn.clicked.connect(self.choose_cell_line_color)
        settings_layout.addRow("Cell line color:", self.cell_line_color_btn)

        self.cell_lines_check.toggled.connect(self.update_cell_lines)
        self.cell_line_thickness_input.valueChanged.connect(self.update_cell_lines)
        self.update_cell_lines()

        # Reset settings
        self.default_settings_btn = QPushButton(
            "Restore Default Settings"
        )

        self.default_settings_btn.clicked.connect(
            self.restore_default_settings
        )

        settings_layout.addRow(
            self.default_settings_btn
        )

        # ----------------------------------------------------
        # SETTINGS HELP TEXT
        # ----------------------------------------------------

        self.settings_help = QLabel(
            "Tip: Increase image count weight for balanced rows. "
            "Increase row height weight for more consistent image sizes."
        )

        self.settings_help.setWordWrap(True)

        settings_layout.addRow(
            self.settings_help
        )

        # Settings area
        settings_scroll = QScrollArea()

        settings_scroll.setWidgetResizable(True)

        settings_scroll.setWidget(
            settings_group
        )

        settings_scroll.setMinimumSize(440, 340)
        self.settings_menu = QMenu(self)
        settings_action = QWidgetAction(self.settings_menu)
        settings_action.setDefaultWidget(settings_scroll)
        self.settings_menu.addAction(settings_action)

        self.settings_toggle_btn = QToolButton(self)
        self.settings_toggle_btn.setText("Settings")
        self.settings_toggle_btn.setToolTip("Advanced layout settings")
        self.settings_toggle_btn.setPopupMode(
            QToolButton.ToolButtonPopupMode.InstantPopup
        )
        self.settings_toggle_btn.setMenu(self.settings_menu)
        controls.addWidget(self.settings_toggle_btn)

        # ----------------------------------------------------
        # LIVE LAYOUT STATISTICS
        # ----------------------------------------------------

        stats_group = QGroupBox(
            "Layout Statistics"
        )

        stats_layout = QVBoxLayout(
            stats_group
        )

        self.stats_label = QLabel(
            "Generate a collage to see layout statistics."
        )

        self.stats_label.setWordWrap(True)

        self.stats_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        stats_layout.addWidget(
            self.stats_label
        )

        stats_group.setMaximumHeight(180)

        main_layout.addWidget(
            stats_group
        )

        # ----------------------------------------------------
        # COLLAGE VIEWER
        # ----------------------------------------------------

        self.viewer_container = QVBoxLayout()

        main_layout.addLayout(
            self.viewer_container,
            stretch=1
        )

    # ========================================================
    # READ SETTINGS FROM UI
    # ========================================================

    def choose_cell_line_color(self):
        color = QColorDialog.getColor(self.cell_line_color, self, "Cell line color")
        if color.isValid():
            self.cell_line_color = color
            self.update_cell_lines()

    def update_cell_lines(self):
        visible = self.cell_lines_check.isChecked()
        self.cell_line_thickness_input.setEnabled(visible)
        self.cell_line_color_btn.setEnabled(visible)
        self.cell_line_color_btn.setStyleSheet(
            f"background-color: {self.cell_line_color.name()}; border: 1px solid #777777;"
        )
        self.cell_line_color_btn.setToolTip(
            f"Cell line color: {self.cell_line_color.name()}"
        )
        if self.viewer is not None:
            self.viewer.set_cell_lines(
                visible, self.cell_line_thickness_input.value(), self.cell_line_color
            )

    def get_layout_settings(self):

        return {

            "equal_row_heights":
                self.equal_height_check.isChecked(),

            "ratio_weight":
                self.ratio_weight_input.value(),

            "count_weight":
                self.count_weight_input.value(),

            "row_search_range":
                self.search_range_input.value(),

            "randomize_images":
                self.randomize_check.isChecked(),

        }

    # ========================================================
    # RESTORE DEFAULT SETTINGS
    # ========================================================

    def restore_default_settings(self):

        self.equal_height_check.setChecked(
            DEFAULT_LAYOUT_SETTINGS["equal_row_heights"]
        )

        self.ratio_weight_input.setValue(
            DEFAULT_LAYOUT_SETTINGS["ratio_weight"]
        )

        self.count_weight_input.setValue(
            DEFAULT_LAYOUT_SETTINGS["count_weight"]
        )

        self.search_range_input.setValue(
            DEFAULT_LAYOUT_SETTINGS["row_search_range"]
        )

        self.randomize_check.setChecked(
            DEFAULT_LAYOUT_SETTINGS["randomize_images"]
        )

        self.cell_lines_check.setChecked(False)
        self.cell_line_thickness_input.setValue(2.0)
        self.cell_line_color = QColor("red")
        self.update_cell_lines()

    # ========================================================
    # SELECT IMAGES
    # ========================================================

    def select_images(self):

        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Images",
            "",
            f"Images ({SUPPORTED_IMAGES})"
        )

        if files:

            self.image_paths = files

            self.info_label.setText(
                f"{len(files)} images selected"
            )

            self.stats_label.setText(
                "Generate a collage to see layout statistics."
            )

    # ========================================================
    # LIVE LAYOUT STATISTICS
    # ========================================================

    def update_layout_statistics(
        self,
        rows,
        row_heights,
        width_in,
        height_in
    ):

        image_count = sum(
            len(row) for row in rows
        )

        row_count = len(rows)

        counts = [
            len(row) for row in rows
        ]

        average_count = (
            image_count / row_count
            if row_count else 0
        )

        min_count = min(counts) if counts else 0
        max_count = max(counts) if counts else 0

        min_height = min(row_heights) if row_heights else 0
        max_height = max(row_heights) if row_heights else 0

        canvas_w = int(width_in * self.dpi)
        canvas_h = int(height_in * self.dpi)

        settings = self.get_layout_settings()

        mode = (
            "Equal row heights"
            if settings["equal_row_heights"]
            else "Natural row heights"
        )

        lines = [

            f"Total images: {image_count}",

            f"Total rows: {row_count}",

            f"Average images per row: {average_count:.2f}",

            f"Minimum / Maximum images per row: "
            f"{min_count} / {max_count}",

            f"Row height range: {min_height} - {max_height} px",

            f"Canvas: {canvas_w} x {canvas_h} px "
            f"({width_in} x {height_in} inches)",

            f"Resolution: {self.dpi} DPI",

            f"Layout mode: {mode}",

            "",

            "ROW DISTRIBUTION:"

        ]

        for index, (row, height) in enumerate(
            zip(rows, row_heights),
            start=1
        ):

            lines.append(
                f"Row {index}: {len(row)} images | "
                f"Height: {height} px"
            )

        self.stats_label.setText(
            "\n".join(lines)
        )

    # ========================================================
    # GENERATE COLLAGE
    # ========================================================

    def generate_collage(self):

        if self.generation_worker is not None:
            return

        if not self.image_paths:

            QMessageBox.warning(
                self,
                "No Images",
                "Please select images first."
            )

            return

        try:

            width_in = float(
                self.width_input.text()
            )

            height_in = float(
                self.height_input.text()
            )

            if width_in <= 0 or height_in <= 0:

                raise ValueError(
                    "Width and height must be positive."
                )

            canvas_width = int(
                width_in * self.dpi
            )

            canvas_height = int(
                height_in * self.dpi
            )

            settings = self.get_layout_settings()

            if (
                settings["ratio_weight"] == 0 and
                settings["count_weight"] == 0
            ):

                QMessageBox.warning(
                    self,
                    "Optimization Weights",
                    "At least one optimization weight "
                    "must be greater than zero."
                )

                return

            self.generation_worker = CollageGenerationWorker(
                self.image_paths, width_in, height_in, self.dpi,
                settings, self
            )
            self.generation_worker.finished.connect(self.generation_finished)
            self.set_generation_busy(True)
            self.info_label.setText("Generating collage...")
            self.generation_worker.start()

        except Exception as exception:
            logging.exception("Could not start collage generation")
            QMessageBox.critical(self, "Collage Generation Error", str(exception))

    def set_generation_busy(self, busy):
        for control in (
            self.select_btn, self.create_btn, self.save_btn,
            self.width_input, self.height_input, self.settings_toggle_btn
        ):
            control.setEnabled(not busy)

    def generation_finished(self):
        worker = self.generation_worker
        self.generation_worker = None
        self.set_generation_busy(False)

        try:
            if worker.error is not None:
                raise RuntimeError(worker.error)

            collage_details, rows, row_heights = worker.result
            width_in, height_in = worker.width_in, worker.height_in
            canvas_width = int(width_in * worker.dpi)
            canvas_height = int(height_in * worker.dpi)

            # Remove old viewer
            if self.viewer:

                self.viewer_container.removeWidget(
                    self.viewer
                )

                self.viewer.setParent(None)

                self.viewer.deleteLater()

                self.viewer = None

            # Create new viewer
            self.viewer = CanvasViewer(
                canvas_width,
                canvas_height,
                collage_details
            )
            self.update_cell_lines()

            self.viewer_container.addWidget(
                self.viewer
            )

            # Store layout information
            self.current_rows = rows

            self.current_row_heights = row_heights

            # Update statistics
            self.update_layout_statistics(
                rows,
                row_heights,
                width_in,
                height_in
            )

            self.info_label.setText(
                f"{len(self.image_paths)} images | "
                f"{width_in} x {height_in} inches | "
                f"{self.dpi} DPI"
            )

            self.showMaximized()

        except Exception as e:

            logging.exception("Could not display collage")
            self.info_label.setText("Collage generation failed")
            QMessageBox.critical(
                self,
                "Collage Generation Error",
                str(e)
            )

        finally:
            worker.deleteLater()

    def closeEvent(self, event):
        if self.generation_worker is not None:
            self.info_label.setText("Finishing collage generation before closing...")
            event.ignore()
            return
        super().closeEvent(event)

    # ========================================================
    # RESET VIEW
    # ========================================================

    def reset_view(self):

        if self.viewer:

            self.viewer.reset_zoom()

    # ========================================================
    # SAVE COLLAGE
    # ========================================================

    def save_collage(self):

        if not self.viewer:

            QMessageBox.warning(
                self,
                "No Collage",
                "Please generate a collage first."
            )

            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Collage",
            "collage.jpg",
            "JPEG (*.jpg *.jpeg)"
        )

        if not path:
            return

        if not path.lower().endswith(
            (".jpg", ".jpeg")
        ):

            path += ".jpg"

        try:

            self.viewer.save_scene_to_jpg(
                path,
                dpi=self.dpi
            )

            QMessageBox.information(
                self,
                "Saved Successfully",
                f"Collage saved successfully:\n{path}"
            )

        except Exception as e:

            QMessageBox.critical(
                self,
                "Save Error",
                str(e)
            )


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":

    app = QApplication(sys.argv)

    window = CollageEditorWindow()

    window.show()

    sys.exit(
        app.exec()
    )
