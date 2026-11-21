import sys
from PyQt6.QtWidgets import QApplication, QGraphicsScene, QGraphicsView, QGraphicsRectItem, QGraphicsPixmapItem
from PyQt6.QtGui import QBrush, QColor, QPen, QWheelEvent, QPixmap, QPainter, QImage
from PyQt6.QtCore import QRectF, Qt
from PIL import Image
import os
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLineEdit, QFileDialog, QLabel, QHBoxLayout

import cv2
import numpy as np
from PIL import Image

import math
import os
import random

# Load OpenCV Haar cascade (comes with cv2)
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def smart_crop_fill(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )

    h, w = img_np.shape[:2]
    target_ratio = target_w / target_h

    if len(faces) > 0:
        # ---- Use face-aware bounding box ----
        x_min = min(x for x, y, fw, fh in faces)
        y_min = min(y for x, y, fw, fh in faces)
        x_max = max(x + fw for x, y, fw, fh in faces)
        y_max = max(y + fh for x, y, fw, fh in faces)

        cx = (x_min + x_max) // 2
        cy = (y_min + y_max) // 2
    else:
        # ---- Fallback: center crop ----
        cx, cy = w // 2, h // 2

    # Determine crop size that fills target
    if w / h > target_ratio:
        crop_h = h
        crop_w = int(h * target_ratio)
    else:
        crop_w = w
        crop_h = int(w / target_ratio)

    left = max(0, cx - crop_w // 2)
    top = max(0, cy - crop_h // 2)

    right = min(w, left + crop_w)
    bottom = min(h, top + crop_h)

    cropped = img.crop((left, top, right, bottom))
    crop_coords = (left, top, right, bottom)

    return cropped.resize((target_w, target_h), Image.LANCZOS), crop_coords

class ImgBox:
    def __init__(self, filepath):
        self.img = Image.open(filepath).convert("RGB")
        self.path = filepath
        self.w, self.h = self.img.size
        self.ratio = self.w / self.h

def layout(images, rect):
    """
    images: list of ImgBox
    rect: (x, y, w, h)
    returns list of (img, rect)
    """
    if not images:
        return []

    if len(images) == 1:
        return [(images[0], rect)]

    x, y, w, h = rect
    canvas_ratio = w / h

    # pick image whose ratio is closest to the available space
    images.sort(key=lambda i: abs(i.ratio - canvas_ratio))
    img = images.pop(0)

    placements = [(img, rect)]

    # decide split direction
    if canvas_ratio > img.ratio:
        # vertical split
        split_w = int(w * (img.ratio / canvas_ratio))
        remaining_rect = (x + split_w, y, w - split_w, h)
    else:
        # horizontal split
        split_h = int(h * (canvas_ratio / img.ratio))
        remaining_rect = (x, y + split_h, w, h - split_h)

    return placements + layout(images, remaining_rect)

def compute_dynamic_row_heights(
    boxes,
    width_in,
    height_in,
    dpi,
    randomness=0.25
):
    """
    boxes      : list of ImgBox (must have .ratio)
    width_in   : output width in inches
    height_in  : output height in inches
    dpi        : DPI
    randomness : 0.0 (no randomness) to 0.5 (very organic)

    returns    : raw_heights (list[int]) that sum exactly to canvas height
    """

    # --- Canvas size ---
    canvas_w = int(width_in * dpi)
    canvas_h = int(height_in * dpi)
    canvas_ratio = canvas_w / canvas_h

    n = len(boxes)
    ratios = np.array([b.ratio for b in boxes])

    # --- Image distribution ---
    landscape_count = np.sum(ratios > 1.2)
    portrait_count = np.sum(ratios < 0.8)

    # --- Decide number of rows ---
    base_rows = round(math.sqrt(n * canvas_h / canvas_w))

    if portrait_count > landscape_count:
        rows = base_rows + 1
    elif landscape_count > portrait_count:
        rows = max(1, base_rows - 1)
    else:
        rows = base_rows

    rows = max(1, rows)

    # --- Estimate natural row height from images ---
    est_row_heights = canvas_w / ratios
    median_row_height = np.median(est_row_heights)

    # --- Dynamic min / max row height ---
    min_row_height_px = int(median_row_height * (1.0 - randomness))
    max_row_height_px = int(median_row_height * (1.0 + randomness))

    # Safety clamps
    min_row_height_px = max(200, min_row_height_px)
    max_row_height_px = min(canvas_h, max_row_height_px)

    # --- Generate row heights ---
    raw = np.random.uniform(
        min_row_height_px,
        max_row_height_px,
        size=rows
    )

    # Normalize to fill canvas height
    raw = raw / raw.sum()
    raw_heights = [int(h * canvas_h) for h in raw]

    # Fix rounding error
    raw_heights[-1] += canvas_h - sum(raw_heights)

    #return np.array(raw_heights)
    # --- Distribute images across rows proportionally ---
    imgs_per_row = [n // rows] * rows
    remainder = n % rows
    for i in range(remainder):
        imgs_per_row[i] += 1  # distribute leftover images

    # --- Return both heights and image counts ---
    return np.array(raw_heights), imgs_per_row   # <-- change

def compute_smart_rows(boxes, canvas_w, canvas_h):
    """
    Returns list of rows
    Each row is a list of ImgBox
    """

    ratios = [b.ratio for b in boxes]
    n = len(boxes)

    avg_ratio = np.mean(ratios)
    target_row_height = canvas_w / (avg_ratio * math.sqrt(n))

    rows = []
    current_row = []
    current_width = 0

    for b in boxes:
        w = target_row_height * b.ratio

        if current_row and current_width + w > canvas_w * 1.05:
            rows.append(current_row)
            current_row = [b]
            current_width = w
        else:
            current_row.append(b)
            current_width += w

    if current_row:
        rows.append(current_row)

    # --- compute natural row heights ---
    natural_heights = []
    for row in rows:
        row_ratio_sum = sum(b.ratio for b in row)
        natural_heights.append(canvas_w / row_ratio_sum)

    # --- balance heights ---
    median_h = np.median(natural_heights)

    min_h = median_h * 0.85
    max_h = median_h * 1.15

    balanced = [
        min(max(h, min_h), max_h)
        for h in natural_heights
    ]

    # --- normalize to canvas height ---
    scale = canvas_h / sum(balanced)
    row_heights = [int(h * scale) for h in balanced]
    row_heights[-1] += canvas_h - sum(row_heights)


    return rows, row_heights

def make_intelligent_collage(
    image_paths,
    output_path,
    width_in,
    height_in,
    dpi=300,
    save_file=False
):
    collage_details = []
    canvas_w = int(width_in * dpi)
    canvas_h = int(height_in * dpi)

    collage = Image.new("RGB", (canvas_w, canvas_h))
    
    boxes = [ImgBox(p) for p in image_paths]
    random.shuffle(boxes)

    rows, row_heights = compute_smart_rows(boxes, canvas_w, canvas_h)
    
    y = 0
    for row_imgs, row_h in zip(rows, row_heights):
        total_ratio = sum(b.ratio for b in row_imgs)
        x = 0

        for i, b in enumerate(row_imgs):
            w = int(canvas_w * (b.ratio / total_ratio))
            if i == len(row_imgs) - 1:
                w = canvas_w - x

            img, crop_coords = smart_crop_fill(b.img, w, row_h)
            collage.paste(img, (x, y))
            collage_details.append({
                "image_path": b.path,
                "image": b.img,
                "canvas_position": (x, y, x + w, y + row_h),
                "crop_position": crop_coords
            })
            x += w

        y += row_h

    

    if save_file:
        collage.save(output_path, dpi=(dpi, dpi))

    return collage_details


class ImageBox(QGraphicsPixmapItem):
    def __init__(self, pixmap, rect: QRectF):
        super().__init__(pixmap)
        self.setFlags(
            QGraphicsPixmapItem.GraphicsItemFlag.ItemIsMovable |
            QGraphicsPixmapItem.GraphicsItemFlag.ItemIsSelectable |
            QGraphicsPixmapItem.GraphicsItemFlag.ItemIsFocusable
        )
        self.setTransformationMode(Qt.TransformationMode.SmoothTransformation)

        # Use QRectF directly
        box_width = rect.width()
        box_height = rect.height()

        # Scale image to fit the box initially
        self.setPos(rect.x(), rect.y())
        if pixmap.width() == 0 or pixmap.height() == 0:
            print("Invalid pixmap size")
            self.scale_factor = 1  # fallback
        else:
            self.scale_factor = min(box_width / pixmap.width(),
                                    box_height / pixmap.height())
        #self.scale_factor = min(box_width / pixmap.width(), box_height / pixmap.height())
        self.setScale(self.scale_factor)

        self.box_rect = rect


class CanvasViewer(QGraphicsView):
    
    def __init__(self, width, height, boxes):
        super().__init__()
        self.setWindowTitle("Collage Grid Viewer")

        self.scene = QGraphicsScene()
        self.setScene(self.scene)

        # White canvas background
        self.scene.setBackgroundBrush(QBrush(QColor("white")))
        self.scene.setSceneRect(0, 0, width, height)

        # Draw grid boxes
        pen = QPen(QColor("red"))
        pen.setWidth(2)
        
        for item in boxes:
           
            x1, y1, x2, y2 = item["canvas_position"]
            #rect = QGraphicsRectItem(QRectF(x1, y1, x2 - x1, y2 - y1))
            rect = QRectF(x1, y1, x2 - x1, y2 - y1)
         
            # Draw grid box
            grid = QGraphicsRectItem(rect)
            grid.setPen(pen)
            grid.setBrush(QBrush(QColor(255,255,255,0)))
            grid.setFlag(grid.GraphicsItemFlag.ItemClipsChildrenToShape, True)  #
            self.scene.addItem(grid)

            # Crop image according to crop_position
            crop_x1, crop_y1, crop_x2, crop_y2 = item["crop_position"]
            crop_width = crop_x2 - crop_x1
            crop_height = crop_y2 - crop_y1

            # Compute scale factor to make crop area fit the grid
            grid_width = x2 - x1
            grid_height = y2 - y1
            scale = max(grid_width / crop_width, grid_height / crop_height)

             # Compute scaled image size
            img_width = item["image"].width * scale
            img_height = item["image"].height * scale

            # Compute offset to align crop area with grid
            offset_x = x1 - crop_x1 * scale
            offset_y = y1 - crop_y1 * scale


            # Load image
            pixmap = QPixmap(item["image_path"])
            pixmap = pixmap.scaled(int(img_width), int(img_height),
                           Qt.AspectRatioMode.IgnoreAspectRatio,
                           Qt.TransformationMode.SmoothTransformation)
            
            # Place image inside grid

            #img_item = ImageBox(pixmap, rect)
            img_item = ImageBox(pixmap, QRectF(offset_x, offset_y, img_width, img_height))

            img_item.setParentItem(grid)
            img_item.setZValue(1)
            grid.setZValue(2)  # Keep grid on top

        # Fit canvas to view
        self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

        # Correct render hints
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing |
            QPainter.RenderHint.SmoothPixmapTransform
        )

        # Zoom factor
        self._zoom = 1

        def save_scene_to_jpg(self, output_path, dpi=300):
            """
            Save the current state of the QGraphicsScene as a JPG image.
            This flattens all layers, including user adjustments.
            """
            # Get scene size
            rect = self.scene.sceneRect()
            width = int(rect.width())
            height = int(rect.height())

            # Create QImage with same size
            image = QImage(width, height, QImage.Format.Format_RGB32)
            image.fill(Qt.GlobalColor.white)

            # Render scene onto QImage
            painter = QPainter(image)
            self.scene.render(painter)
            painter.end()

            # Convert QImage to PIL Image for saving with DPI
            ptr = image.bits()
            ptr.setsize(image.byteCount())
            pil_img = Image.frombytes("RGBA", (image.width(), image.height()), bytes(ptr))
            pil_img = pil_img.convert("RGB")  # remove alpha

            # Save to file with DPI
            pil_img.save(output_path, "JPEG", dpi=(dpi, dpi))
            print(f"Scene saved to {output_path}")
    
    def wheelEvent(self, event: QWheelEvent):
        """Zoom the canvas if Ctrl is NOT pressed"""
        if event.modifiers() != Qt.KeyboardModifier.ControlModifier:
            zoom_in = event.angleDelta().y() > 0
            factor = 1.1 if zoom_in else 0.9
            self._zoom *= factor
            self.scale(factor, factor)
        else:
            # If Ctrl pressed, zoom the selected image only
            selected_items = self.scene.selectedItems()
            if selected_items:
                for item in selected_items:
                    if isinstance(item, ImageBox):
                        zoom_in = event.angleDelta().y() > 0
                        factor = 1.1 if zoom_in else 0.9
                        item.setScale(item.scale() * factor)
            else:
                super().wheelEvent(event)

    def save_scene_to_jpg(self, output_path, dpi=300):
        rect = self.scene.sceneRect()
        width = int(rect.width())
        height = int(rect.height())

        image = QImage(width, height, QImage.Format.Format_RGB32)
        image.fill(Qt.GlobalColor.white)

        painter = QPainter(image)
        self.scene.render(painter)
        painter.end()

        ptr = image.bits()
        ptr.setsize(image.bytesPerLine() * image.height())

        # ✅ FIX COLOR CHANNELS (BGRA → RGB)
        pil_img = Image.frombuffer(
            "RGBA",
            (image.width(), image.height()),
            bytes(ptr),
            "raw",
            "BGRA",
            0,
            1
        ).convert("RGB")

        pil_img.save(output_path, "JPEG", dpi=(dpi, dpi))
        print(f"Scene saved to {output_path}")



#####################################

class CollageEditorWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.dpi = 300
        self.image_paths = []
        self.viewer = None
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("Collage Editor")

        main_layout = QVBoxLayout(self)

        # --- Controls ---
        controls = QHBoxLayout()

        self.select_btn = QPushButton("Select Images")
        self.select_btn.clicked.connect(self.select_images)

        self.width_input = QLineEdit("12")
        self.width_input.setPlaceholderText("Width (inches)")
        self.width_input.setFixedWidth(100)

        self.height_input = QLineEdit("8")
        self.height_input.setPlaceholderText("Height (inches)")
        self.height_input.setFixedWidth(100)

        self.create_btn = QPushButton("Create Collage")
        self.create_btn.clicked.connect(self.generate_collage)

        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self.save_collage)

        controls.addWidget(self.select_btn)
        controls.addWidget(QLabel("Width"))
        controls.addWidget(self.width_input)
        controls.addWidget(QLabel("Height"))
        controls.addWidget(self.height_input)
        controls.addWidget(self.create_btn)
        controls.addWidget(self.save_btn)

        main_layout.addLayout(controls)

        # Placeholder for viewer
        self.viewer_container = QVBoxLayout()
        main_layout.addLayout(self.viewer_container)

    def select_images(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Images",
            "",
            "Images (*.jpg *.jpeg *.png)"
        )
        if files:
            self.image_paths = files

    def generate_collage(self):
        if not self.image_paths:
            return

        width_in = float(self.width_input.text())
        height_in = float(self.height_input.text())

        canvas_width = int(width_in * self.dpi)
        canvas_height = int(height_in * self.dpi)

        collage_details = make_intelligent_collage(
            image_paths=self.image_paths,
            output_path="",
            width_in=width_in,
            height_in=height_in,
            dpi=self.dpi,
            save_file=False
        )

        if self.viewer:
            self.viewer.setParent(None)
            self.viewer.deleteLater()

        self.viewer = CanvasViewer(canvas_width, canvas_height, collage_details)
        self.layout().addWidget(self.viewer)
        self.showMaximized()

    def save_collage(self):
        if not self.viewer:
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Collage",
            "collage.jpg",
            "JPEG (*.jpg)"
        )
        if path:
            self.viewer.save_scene_to_jpg(path, dpi=self.dpi)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = CollageEditorWindow()
    window.show()

    sys.exit(app.exec())
    