"""Main window controller for the PySide6 ONNX segmentation app."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QObject, QPoint, Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QLabel,
    QLayout,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QToolTip,
    QWidget,
)

from constants import PREPROCESS_OPTIONS, TimingInfo
from inference import OnnxSegmentor
from media_source import MediaFrame, MediaSource
from model_registry import ModelEntry, scan_models
from overlay import (
    PredictionResult,
    class_summary,
    postprocess_prediction,
    render_overlay,
    tooltip_for_component,
)
from preprocess import apply_preprocess
from settings import AppSettings, SettingsStore


class ModelLoadWorker(QObject):
    loaded = Signal(int, str, object)
    failed = Signal(int, str, str)
    finished = Signal()

    def __init__(self, request_id: int, model_path: Path) -> None:
        super().__init__()
        self.request_id = request_id
        self.model_path = model_path

    @Slot()
    def run(self) -> None:
        try:
            segmentor = OnnxSegmentor()
            segmentor.load(self.model_path)
            h, w = segmentor.input_size
            segmentor.infer(np.zeros((h, w, 3), dtype=np.uint8))
            self.loaded.emit(self.request_id, str(self.model_path), segmentor)
        except Exception as exc:
            self.failed.emit(self.request_id, str(self.model_path), str(exc))
        finally:
            self.finished.emit()


class AppController(QObject):
    def __init__(self, ui_path: Path) -> None:
        super().__init__()
        loader = QUiLoader()
        self.window = loader.load(str(ui_path))
        if self.window is None:
            raise RuntimeError(f'Could not load UI file: {ui_path}')
        self.app_root = Path(__file__).resolve().parent.parent

        self.settings_store = SettingsStore()
        self.saved = self.settings_store.load()
        self.media_source = MediaSource()
        self.segmentor = OnnxSegmentor()
        self.models: list[ModelEntry] = []

        self.current_frame: MediaFrame | None = None
        self.original_image: np.ndarray | None = None
        self.preprocessed_image: np.ndarray | None = None
        self.current_base_mode = self.saved.preview_mode or 'original'
        self.prediction: PredictionResult | None = None
        self.timing = TimingInfo()
        self._current_pixmap_image: np.ndarray | None = None
        self._model_load_request_id = 0
        self._model_load_threads: list[QThread] = []
        self._model_load_workers: list[ModelLoadWorker] = []

        self._bind_widgets()
        self._configure_widgets()
        self._connect_signals()
        self.window.installEventFilter(self)
        self._restore_settings()
        self._update_status('就绪')

    def show(self) -> None:
        self.window.show()

    def __del__(self) -> None:
        try:
            self.wait_for_model_load_threads()
        except RuntimeError:
            pass

    def _bind_widgets(self) -> None:
        self.main_splitter = self._find(QSplitter, 'mainSplitter')
        self.settings_panel = self._find(QWidget, 'settingsPanel')
        self.image_panel = self._find(QWidget, 'imagePanel')
        self.class_panel = self._find(QWidget, 'classPanel')
        self.btn_open_model_dir = self._find(QPushButton, 'btnOpenModelDir')
        self.btn_reload_models = self._find(QPushButton, 'btnReloadModels')
        self.combo_model = self._find(QComboBox, 'comboModel')
        self.combo_preprocess = self._find(QComboBox, 'comboPreprocess')
        self.spin_threshold = self._find(QDoubleSpinBox, 'spinThreshold')
        self.check_fill_overlay = self._find(QCheckBox, 'checkFillOverlay')
        self.btn_open_image_dir = self._find(QPushButton, 'btnOpenImageDir')
        self.btn_open_image = self._find(QPushButton, 'btnOpenImage')
        self.btn_open_video = self._find(QPushButton, 'btnOpenVideo')
        self.btn_preprocess = self._find(QPushButton, 'btnPreprocess')
        self.btn_toggle_preview = self._find(QPushButton, 'btnTogglePreview')
        self.btn_infer = self._find(QPushButton, 'btnInfer')
        self.model_dir_label = self._find(QLabel, 'modelDirLabel')
        self.media_label = self._find(QLabel, 'mediaLabel')
        self.image_label = self._find(QLabel, 'imageLabel')
        self.class_list_widget = self._find(QWidget, 'classListWidget')
        self.class_list_layout = self._find(QLayout, 'classListLayout')
        self.statusbar = self._find(QStatusBar, 'statusbar')

    def _find(self, widget_type, name: str):
        widget = self.window.findChild(widget_type, name)
        if widget is None:
            raise RuntimeError(f'Missing UI object: {name}')
        return widget

    def _configure_widgets(self) -> None:
        self.settings_panel.setMinimumWidth(280)
        self.settings_panel.setMaximumWidth(360)
        self.class_panel.setMinimumWidth(260)
        self.class_panel.setMaximumWidth(360)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setStretchFactor(2, 0)
        self.main_splitter.setSizes([320, 780, 260])
        self.combo_preprocess.addItems(PREPROCESS_OPTIONS)
        self.spin_threshold.setRange(0.0, 1.0)
        self.spin_threshold.setSingleStep(0.05)
        self.spin_threshold.setValue(self.saved.threshold)
        self.check_fill_overlay.setChecked(self.saved.fill_overlay)
        self.btn_infer.setEnabled(False)
        self.image_label.setMouseTracking(True)
        self.image_label.installEventFilter(self)
        self.image_label.setScaledContents(False)
        self._install_shortcuts()

    def _install_shortcuts(self) -> None:
        for key in (Qt.Key_Left, Qt.Key_A):
            shortcut = QShortcut(QKeySequence(key), self.window)
            shortcut.activated.connect(self.previous_frame)
        for key in (Qt.Key_Right, Qt.Key_D):
            shortcut = QShortcut(QKeySequence(key), self.window)
            shortcut.activated.connect(self.next_frame)

    def _connect_signals(self) -> None:
        self.btn_open_model_dir.clicked.connect(self.open_model_dir)
        self.btn_reload_models.clicked.connect(self.reload_models)
        self.combo_model.currentIndexChanged.connect(self.select_model)
        self.combo_preprocess.currentTextChanged.connect(self.invalidate_preprocess)
        self.spin_threshold.valueChanged.connect(self.invalidate_prediction)
        self.check_fill_overlay.stateChanged.connect(lambda _state: self.refresh_image())
        self.btn_open_image_dir.clicked.connect(self.open_image_folder)
        self.btn_open_image.clicked.connect(self.open_image)
        self.btn_open_video.clicked.connect(self.open_video)
        self.btn_preprocess.clicked.connect(self.preprocess_current)
        self.btn_toggle_preview.clicked.connect(self.toggle_preview)
        self.btn_infer.clicked.connect(self.infer_current)

    def _restore_settings(self) -> None:
        if self.saved.geometry:
            self.window.restoreGeometry(self.saved.geometry)
        if self.saved.window_state:
            self.window.restoreState(self.saved.window_state)
        if self.saved.preprocess in PREPROCESS_OPTIONS:
            self.combo_preprocess.setCurrentText(self.saved.preprocess)
        self.current_base_mode = (
            self.saved.preview_mode
            if self.saved.preview_mode in {'original', 'preprocessed'}
            else 'original')
        if self.saved.model_dir:
            self.load_model_dir(Path(self.saved.model_dir), self.saved.model_path)
        self._restore_media()

    def _restore_media(self) -> None:
        try:
            if self.saved.media_kind == 'folder' and self.saved.image_dir:
                frame = self.media_source.open_folder(self.saved.image_dir)
                self.media_source.index = min(
                    max(self.saved.media_index, 0), self.media_source.total - 1)
                self.set_frame(self.media_source.current())
            elif self.saved.media_kind == 'image' and self.saved.image_path:
                self.set_frame(self.media_source.open_image(self.saved.image_path))
            elif self.saved.media_kind == 'video' and self.saved.video_path:
                frame = self.media_source.open_video(self.saved.video_path)
                self.media_source.index = min(
                    max(self.saved.media_index, 0), self.media_source.total - 1)
                self.set_frame(self.media_source.current())
        except Exception as exc:
            self._update_status(f'恢复媒体失败：{exc}')

    def open_model_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self.window, '打开模型文件夹')
        if folder:
            self.load_model_dir(Path(folder))

    def reload_models(self) -> None:
        folder = self.model_dir_label.property('model_dir') or self.saved.model_dir
        if folder:
            self.load_model_dir(Path(folder), self.current_model_path())

    def load_model_dir(self, folder: Path, selected_model: str = '') -> None:
        self.models = scan_models(folder)
        self.combo_model.blockSignals(True)
        self.combo_model.clear()
        for entry in self.models:
            self.combo_model.addItem(entry.display_name, str(entry.path))
        self.combo_model.blockSignals(False)
        display_folder = _display_path(folder, self.app_root)
        self.model_dir_label.setText(f'模型文件夹：{display_folder}')
        self.model_dir_label.setToolTip(str(folder))
        self.model_dir_label.setProperty('model_dir', str(folder))

        index = 0
        if selected_model:
            selected = str(Path(selected_model))
            for i, entry in enumerate(self.models):
                if str(entry.path) == selected:
                    index = i
                    break
        if self.models:
            self.combo_model.blockSignals(True)
            self.combo_model.setCurrentIndex(index)
            self.combo_model.blockSignals(False)
            self.select_model(index)
        else:
            self.btn_infer.setEnabled(False)
            self._update_status('模型文件夹中没有 .onnx 文件')

    def select_model(self, index: int) -> None:
        if index < 0 or index >= len(self.models):
            return
        entry = self.models[index]
        self.combo_preprocess.setCurrentText(entry.recommended_preprocess)
        self.start_model_load(entry)

    def start_model_load(self, entry: ModelEntry) -> None:
        self._model_load_request_id += 1
        request_id = self._model_load_request_id
        self.segmentor = OnnxSegmentor()
        self.btn_infer.setEnabled(False)

        thread = QThread()
        worker = ModelLoadWorker(request_id, entry.path)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.loaded.connect(self.finish_model_load)
        worker.failed.connect(self.fail_model_load)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(
            lambda thread=thread, worker=worker: self.forget_model_load_job(
                thread, worker))
        self._model_load_threads.append(thread)
        self._model_load_workers.append(worker)
        thread.start()
        self._update_status(f'后台加载并预热模型：{entry.network_name}')

    @Slot(int, str, object)
    def finish_model_load(self, request_id: int, model_path: str, segmentor: object) -> None:
        if request_id != self._model_load_request_id or model_path != self.current_model_path():
            return
        if not isinstance(segmentor, OnnxSegmentor):
            self._update_status('模型加载失败：加载器返回了无效对象')
            return
        self.segmentor = segmentor
        self.btn_infer.setEnabled(True)
        entry = self.current_model_entry()
        network = entry.network_name if entry else 'Unknown Network'
        self._update_status(
            f'模型已后台加载并预热：{network} | 设备: {self._device_label()}')

    @Slot(int, str, str)
    def fail_model_load(self, request_id: int, model_path: str, message: str) -> None:
        if request_id != self._model_load_request_id or model_path != self.current_model_path():
            return
        self.segmentor = OnnxSegmentor()
        self.btn_infer.setEnabled(False)
        self._update_status(f'模型加载失败：{message}')

    def forget_model_load_job(self, thread: QThread, worker: ModelLoadWorker) -> None:
        try:
            self._model_load_threads.remove(thread)
        except ValueError:
            pass
        try:
            self._model_load_workers.remove(worker)
        except ValueError:
            pass

    def current_model_entry(self) -> ModelEntry | None:
        index = self.combo_model.currentIndex()
        if 0 <= index < len(self.models):
            return self.models[index]
        return None

    def current_model_path(self) -> str:
        entry = self.current_model_entry()
        return str(entry.path) if entry else ''

    def open_image_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self.window, '打开图片文件夹')
        if not folder:
            return
        try:
            self.set_frame(self.media_source.open_folder(folder))
        except Exception as exc:
            self._show_error(str(exc))

    def open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self.window, '打开图片', '', 'Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)')
        if not path:
            return
        try:
            self.set_frame(self.media_source.open_image(path))
        except Exception as exc:
            self._show_error(str(exc))

    def open_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self.window, '打开视频', '', 'Videos (*.mp4 *.avi *.mov *.mkv *.wmv *.m4v)')
        if not path:
            return
        try:
            self.set_frame(self.media_source.open_video(path))
        except Exception as exc:
            self._show_error(str(exc))

    def set_frame(self, frame: MediaFrame) -> None:
        self.current_frame = frame
        self.original_image = frame.image_rgb
        self.preprocessed_image = None
        self.current_base_mode = 'original'
        self.prediction = None
        self.timing = TimingInfo()
        self.media_label.setText(
            f'媒体：{frame.label} ({frame.index + 1}/{frame.total})')
        self.clear_class_list()
        self.refresh_image()
        self._update_status('已加载媒体')

    def next_frame(self) -> None:
        try:
            self.set_frame(self.media_source.next())
        except Exception as exc:
            self._update_status(str(exc))

    def previous_frame(self) -> None:
        try:
            self.set_frame(self.media_source.previous())
        except Exception as exc:
            self._update_status(str(exc))

    def preprocess_current(self) -> None:
        if self.original_image is None:
            self._update_status('请先打开图片、图片文件夹或视频')
            return
        result = apply_preprocess(
            self.original_image, self.combo_preprocess.currentText())
        self.preprocessed_image = result.image
        self.current_base_mode = 'preprocessed'
        self.prediction = None
        self.timing = TimingInfo(preprocess_ms=result.elapsed_ms)
        self.clear_class_list()
        self.refresh_image()
        self._update_status('预处理完成')

    def toggle_preview(self) -> None:
        if self.original_image is None:
            return
        if self.current_base_mode == 'preprocessed':
            self.current_base_mode = 'original'
        elif self.preprocessed_image is not None:
            self.current_base_mode = 'preprocessed'
        else:
            self._update_status('当前帧还没有预处理图')
            return
        self.refresh_image()
        self._update_status('已切换预览')

    def infer_current(self) -> None:
        if self.original_image is None:
            self._update_status('请先打开图片、图片文件夹或视频')
            return
        if not self.segmentor.is_loaded():
            self._update_status('请先加载 ONNX 模型')
            return

        if self.preprocessed_image is None:
            preprocess_result = apply_preprocess(
                self.original_image, self.combo_preprocess.currentText())
            self.preprocessed_image = preprocess_result.image
            self.current_base_mode = 'preprocessed'
            preprocess_ms = preprocess_result.elapsed_ms
        else:
            preprocess_ms = self.timing.preprocess_ms

        try:
            output = self.segmentor.infer(self.preprocessed_image)
            target_h, target_w = self.preprocessed_image.shape[:2]
            prediction = postprocess_prediction(
                output.logits, (target_h, target_w), self.spin_threshold.value())
        except Exception as exc:
            self._show_error(str(exc))
            return

        self.prediction = prediction
        self.timing = TimingInfo(
            preprocess_ms=preprocess_ms,
            inference_ms=output.elapsed_ms,
            postprocess_ms=prediction.postprocess_ms,
        )
        self.populate_class_list()
        self.refresh_image()
        self._update_status('推理完成')

    def invalidate_preprocess(self) -> None:
        self.preprocessed_image = None
        self.prediction = None
        self.clear_class_list()
        self._update_status('预处理设置已改变，当前结果已过期')

    def invalidate_prediction(self) -> None:
        self.prediction = None
        self.clear_class_list()
        self.refresh_image()
        self._update_status('阈值已改变，请重新推理')

    def clear_class_list(self) -> None:
        while self.class_list_layout.count() > 1:
            item = self.class_list_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def populate_class_list(self) -> None:
        self.clear_class_list()
        if self.prediction is None:
            return
        for summary in class_summary(self.prediction):
            class_id = int(summary['class_id'])
            color = summary['color']
            checkbox = QCheckBox(
                f"{summary['class_name']}\n"
                f"区域 {summary['count']} | 面积 {summary['area']}\n"
                f"置信度 {summary['mean_confidence']:.3f}")
            checkbox.setChecked(class_id in self.prediction.visible_class_ids)
            checkbox.setMinimumWidth(230)
            checkbox.setToolTip(checkbox.text())
            checkbox.setStyleSheet(
                f'QCheckBox {{ color: rgb({color[0]}, {color[1]}, {color[2]}); }}')
            checkbox.stateChanged.connect(
                lambda state, cid=class_id: self.set_class_visible(
                    cid, _is_checked(state)))
            self.class_list_layout.insertWidget(
                max(self.class_list_layout.count() - 1, 0), checkbox)

    def set_class_visible(self, class_id: int, visible: bool) -> None:
        if self.prediction is None:
            return
        if visible:
            self.prediction.visible_class_ids.add(class_id)
        else:
            self.prediction.visible_class_ids.discard(class_id)
        self.refresh_image()

    def refresh_image(self) -> None:
        base = self.current_base_image()
        if base is None:
            self.image_label.clear()
            self.image_label.setText('打开图片、图片文件夹或视频')
            self._current_pixmap_image = None
            self._hide_hover_tooltip()
            return
        shown = render_overlay(
            base, self.prediction, self.check_fill_overlay.isChecked())
        self._current_pixmap_image = shown
        self._update_image_pixmap()

    def current_base_image(self) -> np.ndarray | None:
        if self.current_base_mode == 'preprocessed' and self.preprocessed_image is not None:
            return self.preprocessed_image
        return self.original_image

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        try:
            if watched is self.image_label and event.type() == QEvent.Type.MouseMove:
                self._handle_hover(event.position().toPoint())
            elif watched is self.image_label and event.type() == QEvent.Type.Resize:
                self._update_image_pixmap()
            elif watched is self.image_label and event.type() == QEvent.Type.Leave:
                self._hide_hover_tooltip()
            elif watched is self.window and event.type() == QEvent.Type.Close:
                self.save_settings()
                self.media_source.close()
                self.wait_for_model_load_threads()
            return super().eventFilter(watched, event)
        except RuntimeError:
            return False

    def _handle_hover(self, point: QPoint) -> None:
        if self.prediction is None or self._current_pixmap_image is None:
            self._hide_hover_tooltip()
            return
        coords = self._label_point_to_image(point)
        if coords is None:
            self._hide_hover_tooltip()
            return
        x, y = coords
        component = self.prediction.component_at(x, y)
        if component is None or component.class_id not in self.prediction.visible_class_ids:
            self._hide_hover_tooltip()
            return
        tooltip = tooltip_for_component(component)
        self.image_label.setToolTip(tooltip)
        QToolTip.showText(
            self.image_label.mapToGlobal(point + QPoint(12, 20)),
            tooltip,
            self.image_label,
        )

    def _hide_hover_tooltip(self) -> None:
        self.image_label.setToolTip('')
        QToolTip.hideText()

    def _label_point_to_image(self, point: QPoint) -> tuple[int, int] | None:
        pixmap = self.image_label.pixmap()
        image = self._current_pixmap_image
        if pixmap is None or image is None:
            return None
        label_w = self.image_label.width()
        label_h = self.image_label.height()
        pix_w = pixmap.width()
        pix_h = pixmap.height()
        offset_x = (label_w - pix_w) // 2
        offset_y = (label_h - pix_h) // 2
        local_x = point.x() - offset_x
        local_y = point.y() - offset_y
        if local_x < 0 or local_y < 0 or local_x >= pix_w or local_y >= pix_h:
            return None
        image_h, image_w = image.shape[:2]
        x = int(local_x * image_w / max(pix_w, 1))
        y = int(local_y * image_h / max(pix_h, 1))
        return x, y

    def _update_image_pixmap(self) -> None:
        if self._current_pixmap_image is None:
            return
        pixmap = _to_pixmap(self._current_pixmap_image)
        target_size = self.image_label.contentsRect().size()
        if target_size.isEmpty():
            target_size = self.image_label.size()
        self.image_label.setPixmap(pixmap.scaled(
            target_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        ))

    def _update_status(self, message: str = '') -> None:
        entry = self.current_model_entry()
        network = entry.network_name if entry else self.saved.network_name
        frame = self.current_frame
        position = f'{frame.index + 1}/{frame.total}' if frame else '-/-'
        filename = frame.label if frame else '未选择媒体'
        status = (
            f'{message} | 网络: {network} | '
            f'设备: {self._device_label()} | '
            f'预处理: {self.timing.preprocess_ms:.1f} ms | '
            f'推理: {self.timing.inference_ms:.1f} ms | '
            f'后处理: {self.timing.postprocess_ms:.1f} ms | '
            f'FPS: {self.timing.fps:.2f} | '
            f'{position} | {filename}')
        self.statusbar.showMessage(status)

    def _device_label(self) -> str:
        providers = self.segmentor.provider_names()
        if not providers:
            return '未加载'
        if providers[0] == 'CUDAExecutionProvider':
            return 'CUDA'
        if providers[0] == 'CPUExecutionProvider':
            return 'CPU'
        return providers[0].replace('ExecutionProvider', '')

    def _show_error(self, message: str) -> None:
        self._update_status(message)
        QMessageBox.critical(self.window, '错误', message)

    def wait_for_model_load_threads(self) -> None:
        for thread in list(self._model_load_threads):
            if thread.isRunning():
                thread.quit()
                thread.wait()

    def save_settings(self) -> None:
        frame = self.current_frame
        model_dir = self.model_dir_label.property('model_dir') or ''
        entry = self.current_model_entry()
        state = AppSettings(
            model_dir=str(model_dir),
            model_path=str(entry.path) if entry else '',
            network_name=entry.network_name if entry else 'Unknown Network',
            preprocess=self.combo_preprocess.currentText(),
            threshold=self.spin_threshold.value(),
            image_dir=str(self.media_source.paths[0].parent)
            if self.media_source.kind == 'folder' and self.media_source.paths else '',
            image_path=str(self.media_source.paths[0])
            if self.media_source.kind == 'image' and self.media_source.paths else '',
            video_path=str(self.media_source.video_path or ''),
            media_kind=self.media_source.kind,
            media_index=self.media_source.index if frame else 0,
            preview_mode=self.current_base_mode,
            fill_overlay=self.check_fill_overlay.isChecked(),
            geometry=self.window.saveGeometry(),
            window_state=self.window.saveState(),
        )
        self.settings_store.save(state)


def _to_pixmap(image_rgb: np.ndarray) -> QPixmap:
    contiguous = np.ascontiguousarray(image_rgb)
    h, w, ch = contiguous.shape
    image = QImage(contiguous.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(image.copy())


def _display_path(path: str | Path, root: Path) -> str:
    resolved = Path(path).expanduser().resolve()
    for base in (Path.cwd().resolve(), root.resolve(), Path.home().resolve()):
        try:
            relative = resolved.relative_to(base)
        except ValueError:
            continue
        if base == Path.home().resolve():
            return str(Path('~') / relative)
        return str(relative)
    return str(resolved)


def _is_checked(state: int) -> bool:
    try:
        state_value = state.value
    except AttributeError:
        state_value = int(state)
    return state_value == Qt.CheckState.Checked.value
