"""Main window for the zarr sequence viewer application."""

import matplotlib
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ...graph.zarr_loader import ZarrSegmentationLoader


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Zarr Sequence Viewer")
        self.setMinimumSize(1000, 700)

        # Data state
        self.loader = ZarrSegmentationLoader()
        self.zarr_path: str | None = None
        self.sequences: list[dict[str, int]] = []
        self.current_sequence_index = -1

        # Current sequence data
        self.data: np.ndarray | None = None  # (T, C, H, W)
        self.nuclei_masks: np.ndarray | None = None  # (T, H, W)
        self.cell_masks: np.ndarray | None = None  # (T, H, W)
        self.channel_names: list[str] = []

        # Display state
        self.normalized_data: np.ndarray | None = None
        self.current_frame = 0
        self.total_frames = 0
        self.current_channel = 0
        self.img_display = None
        self._last_mode_seg = False

        # Initialize UI
        self._init_ui()

        # Timer for autoplay
        self.timer = QTimer(self)
        self.timer.setInterval(100)  # 100ms between frames (10 fps)
        self.timer.timeout.connect(self._advance_frame)

    def _init_ui(self):
        """Initialize the user interface"""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # Left panel - Controls
        left_panel = QWidget()
        left_panel.setMaximumWidth(400)
        left_layout = QVBoxLayout(left_panel)

        # File Controls Group
        file_group = QGroupBox("Zarr Navigation")
        file_layout = QVBoxLayout()

        self.open_zarr_button = QPushButton("Open Zarr")
        self.open_zarr_button.clicked.connect(self._handle_open_zarr)
        file_layout.addWidget(self.open_zarr_button)

        nav_layout = QHBoxLayout()
        self.prev_button = QPushButton("◀ Previous")
        self.prev_button.setEnabled(False)
        self.prev_button.clicked.connect(self._handle_prev_sequence)
        nav_layout.addWidget(self.prev_button)

        self.next_button = QPushButton("Next ▶")
        self.next_button.setEnabled(False)
        self.next_button.clicked.connect(self._handle_next_sequence)
        nav_layout.addWidget(self.next_button)
        file_layout.addLayout(nav_layout)

        self.sequence_label = QLabel("No zarr loaded")
        self.sequence_label.setWordWrap(True)
        file_layout.addWidget(self.sequence_label)

        file_group.setLayout(file_layout)
        left_layout.addWidget(file_group)

        # Frame Controls Group
        frame_group = QGroupBox("Frame Controls")
        frame_layout = QVBoxLayout()

        slider_label = QLabel("Frame Navigation:")
        frame_layout.addWidget(slider_label)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._handle_slider_change)
        frame_layout.addWidget(self.slider)

        self.frame_label = QLabel("Frame: 0/0")
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        frame_layout.addWidget(self.frame_label)

        self.autoplay_button = QPushButton("▶ Play")
        self.autoplay_button.setCheckable(True)
        self.autoplay_button.setEnabled(False)
        self.autoplay_button.clicked.connect(self._handle_autoplay_toggle)
        frame_layout.addWidget(self.autoplay_button)

        channel_label = QLabel("Channel Selection:")
        frame_layout.addWidget(channel_label)

        self.channel_selector = QComboBox()
        self.channel_selector.setEnabled(False)
        self.channel_selector.currentIndexChanged.connect(self._handle_channel_change)
        frame_layout.addWidget(self.channel_selector)

        frame_group.setLayout(frame_layout)
        left_layout.addWidget(frame_group)

        left_layout.addStretch()

        # Right panel - Image viewer
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.figure = Figure(tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.axis("off")
        right_layout.addWidget(self.canvas)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)

    # Zarr operations
    def _handle_open_zarr(self):
        """Handle opening a zarr folder"""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Zarr Store", "", QFileDialog.Option.ShowDirsOnly
        )
        if folder_path:
            self._load_zarr(folder_path)

    def _load_zarr(self, zarr_path: str):
        """Load zarr store and list sequences"""
        try:
            self.sequences = self.loader.list_sequences(zarr_path)
            if not self.sequences:
                self.statusBar.showMessage("No sequences found in zarr store")
                self._reset_state()
                return

            self.zarr_path = zarr_path
            self.current_sequence_index = 0
            self._load_current_sequence()
            self._update_navigation_buttons()
            self.statusBar.showMessage(f"Loaded zarr: {zarr_path} ({len(self.sequences)} sequences)")
        except Exception as e:
            self.statusBar.showMessage(f"Error loading zarr: {e}")
            self._reset_state()

    def _load_current_sequence(self):
        """Load the current sequence data"""
        if self.current_sequence_index < 0 or self.current_sequence_index >= len(self.sequences):
            return

        seq = self.sequences[self.current_sequence_index]
        try:
            result = self.loader.load_cell_filter_data(
                self.zarr_path,
                seq["fov_idx"],
                seq["pattern_idx"],
            )

            self.data = result["data"]  # (T, C, H, W)
            self.nuclei_masks = result.get("nuclei_masks")  # (T, H, W) or None
            self.cell_masks = result.get("segmentation_masks")  # (T, H, W) or None

            # Build channel names
            if result.get("channels"):
                self.channel_names = list(result["channels"])
            else:
                self.channel_names = [f"channel_{i}" for i in range(self.data.shape[1])]

            # Add mask pseudo-channels
            if self.cell_masks is not None:
                self.channel_names.append("cell_masks")
            if self.nuclei_masks is not None:
                self.channel_names.append("nuclei_masks")

            # Normalize image data for display
            self.normalized_data = self._normalize_stack(self.data)

            self.total_frames = self.data.shape[0]
            self.current_frame = 0
            self.current_channel = 0
            self.img_display = None

            # Update UI
            self._update_channel_selector()
            self.slider.setRange(0, self.total_frames - 1)
            self.slider.setValue(0)
            self.slider.setEnabled(True)
            self.autoplay_button.setEnabled(True)
            self.channel_selector.setEnabled(True)

            # Update label
            self.sequence_label.setText(
                f"Sequence {self.current_sequence_index + 1}/{len(self.sequences)}\n"
                f"FOV: {seq['fov_idx']}, Cell: {seq['pattern_idx']}"
            )

            self._update_view()

        except Exception as e:
            self.statusBar.showMessage(f"Error loading sequence: {e}")

    def _normalize_stack(self, stack: np.ndarray) -> np.ndarray:
        """Normalize each channel using 1st and 99.9th percentiles."""
        normalized = np.zeros(stack.shape, dtype=np.uint8)
        num_channels = stack.shape[1]

        for c in range(num_channels):
            channel_data = stack[:, c, :, :]
            positive_mask = channel_data > 0

            if positive_mask.any():
                positive_values = channel_data[positive_mask]
                low = np.percentile(positive_values, 0.1)
                high = np.percentile(positive_values, 99.9)

                if high > low:
                    normalized_channel = (channel_data - low) / (high - low) * 255
                    normalized_channel = np.clip(normalized_channel, 0, 255)
                    normalized[:, c, :, :] = normalized_channel.astype(np.uint8)
                else:
                    normalized[:, c, :, :] = 128

        return normalized

    def _reset_state(self):
        """Reset all state"""
        self.zarr_path = None
        self.sequences = []
        self.current_sequence_index = -1
        self.data = None
        self.nuclei_masks = None
        self.cell_masks = None
        self.normalized_data = None
        self.channel_names = []
        self.img_display = None

        self.slider.setEnabled(False)
        self.slider.setRange(0, 0)
        self.autoplay_button.setEnabled(False)
        self.channel_selector.setEnabled(False)
        self.prev_button.setEnabled(False)
        self.next_button.setEnabled(False)
        self.frame_label.setText("Frame: 0/0")
        self.sequence_label.setText("No zarr loaded")

    def _update_navigation_buttons(self):
        """Update navigation button states"""
        self.prev_button.setEnabled(self.current_sequence_index > 0)
        self.next_button.setEnabled(self.current_sequence_index < len(self.sequences) - 1)

    def _handle_prev_sequence(self):
        """Navigate to previous sequence"""
        if self.current_sequence_index > 0:
            self.current_sequence_index -= 1
            self._load_current_sequence()
            self._update_navigation_buttons()

    def _handle_next_sequence(self):
        """Navigate to next sequence"""
        if self.current_sequence_index < len(self.sequences) - 1:
            self.current_sequence_index += 1
            self._load_current_sequence()
            self._update_navigation_buttons()

    # Frame operations
    def _handle_slider_change(self, value):
        """Handle slider value changes"""
        self.current_frame = value
        self._update_view()

    def _handle_autoplay_toggle(self, checked):
        """Handle autoplay button toggle"""
        if checked:
            self.timer.start()
            self.autoplay_button.setText("⏸ Pause")
        else:
            self.timer.stop()
            self.autoplay_button.setText("▶ Play")

    def _advance_frame(self):
        """Advance to the next frame"""
        if self.current_frame < self.total_frames - 1:
            self.current_frame += 1
        else:
            self.current_frame = 0
        self.slider.setValue(self.current_frame)
        self._update_view()

    def _update_channel_selector(self):
        """Update channel selector with available channels"""
        self.channel_selector.blockSignals(True)
        self.channel_selector.clear()

        for name in self.channel_names:
            self.channel_selector.addItem(name)

        self.channel_selector.setCurrentIndex(0)
        self.channel_selector.blockSignals(False)

    def _handle_channel_change(self, index):
        """Handle channel selection change"""
        self.current_channel = index
        self._update_view()
        if 0 <= index < len(self.channel_names):
            self.statusBar.showMessage(f"Channel: {self.channel_names[index]}")

    def _update_view(self):
        """Update the display with current frame and channel"""
        if self.data is None:
            return

        try:
            from matplotlib import cm, colors

            channel_name = self.channel_names[self.current_channel]
            is_mask = channel_name in ("cell_masks", "nuclei_masks")

            if is_mask:
                # Display mask with categorical colormap
                if channel_name == "cell_masks":
                    mask_data = self.cell_masks[self.current_frame]
                else:
                    mask_data = self.nuclei_masks[self.current_frame]

                mask_data = mask_data.astype(np.int64)
                max_label = int(mask_data.max()) if mask_data.size else 0
                n_classes = max(2, max_label + 1)

                base = cm.get_cmap("tab20", 20)
                colors_list = [(0.0, 0.0, 0.0, 1.0)]  # background black
                for i in range(1, n_classes):
                    colors_list.append(base((i - 1) % 20))
                cmap = matplotlib.colors.ListedColormap(colors_list, name="seg_cmap", N=n_classes)

                boundaries = np.arange(-0.5, max_label + 1.5, 1)
                norm = colors.BoundaryNorm(boundaries, ncolors=cmap.N, clip=True)

                recreate = self.img_display is None or not self._last_mode_seg
                if recreate:
                    self.ax.clear()
                    self.ax.axis("off")
                    self.img_display = self.ax.imshow(mask_data, cmap=cmap, norm=norm, interpolation="nearest")
                    self.figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
                else:
                    self.img_display.set_cmap(cmap)
                    self.img_display.set_norm(norm)
                    self.img_display.set_data(mask_data)
                self._last_mode_seg = True
            else:
                # Display normalized image channel
                display_image = self.normalized_data[self.current_frame, self.current_channel]

                recreate = self.img_display is None or self._last_mode_seg
                if recreate:
                    self.ax.clear()
                    self.ax.axis("off")
                    self.img_display = self.ax.imshow(display_image, cmap="gray", aspect="equal", vmin=0, vmax=255)
                    self.figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
                else:
                    self.img_display.set_data(display_image)
                self._last_mode_seg = False

            self.canvas.draw()
            self.frame_label.setText(f"Frame: {self.current_frame + 1}/{self.total_frames}")

        except Exception as e:
            self.statusBar.showMessage(f"Error updating view: {e}")

    def resizeEvent(self, event):  # noqa: N802
        """Handle window resize events"""
        super().resizeEvent(event)
        if self.img_display is not None:
            self.figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
            self.canvas.draw()
