import os
import sys
import cv2
import faiss
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE
import torch
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError
import shutil
import subprocess
import joblib
import logging
import torch.nn.functional as F
import timm
import concurrent.futures
from pathlib import Path
from retrying import retry
from typing import List, Tuple, Dict, Optional, Callable
import random
import string

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QComboBox, QProgressBar,
                             QCheckBox, QFileDialog, QMessageBox, QStyleFactory)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QEvent, QSize
from PyQt5.QtGui import QDragEnterEvent, QDropEvent, QIcon

from PyQt5.QtWidgets import QGroupBox, QStyle  # 添加这行

# 配置日志记录
logging.basicConfig(
    filename='image_clustering.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# -------------------- 全局配置 --------------------
DEFAULT_CONFIG = {
    'sorting_model_path': 'weights/MobileNetV4.bin',
    'image_extensions': ['.jpg', '.jpeg', '.png', '.bmp'],
    'raw_extensions': ['.NEF', '.ARW', '.CR2', '.DNG'],
    'max_clusters_options': [5, 10, 15, 20],
    'niter_options': [100, 200, 300, 400],
    'default_max_clusters': 10,
    'default_niter': 300,
    'progress_stages': {
        'feature_extraction': (0, 30),
        'clustering': (30, 50),
        'copying': (50, 80),
        'sorting': (80, 100)
    }
}

# -------------------- 工具类 --------------------
class FileProcessor:
    @staticmethod
    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def safe_copy(src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        while dst.exists():
            base_name = dst.stem
            extension = dst.suffix
            random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=4))
            new_name = f"{base_name}_{random_str}{extension}"
            dst = dst.with_name(new_name)
        try:
            shutil.copy2(src, dst)
        except Exception as e:
            logging.error(f"复制失败 {src} -> {dst}: {e}")
            raise

    @classmethod
    def get_image_files(cls, folder: Path) -> List[Path]:
        return [f for f in folder.glob('*') if f.suffix.lower() in DEFAULT_CONFIG['image_extensions']]

# -------------------- 特征提取和策略类 --------------------
class FeatureExtractor:
    def __init__(self, device: torch.device):
        self.device = device
        self.model = self._load_feature_model()
        self.preprocessor = self._get_preprocessor()

    def _load_feature_model(self) -> torch.nn.Module:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        return torch.nn.Sequential(*list(model.children())[:-1]).to(self.device).eval()

    def _get_preprocessor(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(448),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomHorizontalFlip(p=0.5)
        ])

    def extract_batch(self, batch: List[Image.Image]) -> np.ndarray:
        tensors = torch.stack([self.preprocessor(img) for img in batch]).to(self.device)
        with torch.no_grad():
            features = self.model(tensors)
        return features.squeeze().cpu().numpy()

class FaissKMeansClustering:
    def __init__(self, max_clusters: int = 10, niter: int = 300):
        self.max_clusters = max_clusters
        self.niter = niter

    def cluster(self, features: np.ndarray) -> np.ndarray:
        features = features.astype('float32')
        faiss.normalize_L2(features)
        best_k = self._find_optimal_clusters(features)
        kmeans = faiss.Kmeans(features.shape[1], best_k, niter=self.niter, gpu=True)
        kmeans.train(features)
        _, cluster_ids = kmeans.index.search(features, 1)
        return cluster_ids.flatten()

    def _find_optimal_clusters(self, features: np.ndarray) -> int:
        distortions = []
        for k in range(1, self.max_clusters + 1):
            kmeans = faiss.Kmeans(features.shape[1], k, niter=self.niter, gpu=True)
            kmeans.train(features)
            distortions.append(kmeans.obj[-1])
        deltas = np.diff(distortions)
        best_k = np.argmax(deltas) + 2
        return max(2, min(best_k, self.max_clusters))

class SimilaritySorting:
    def sort(self, cluster_dir: Path, model: torch.nn.Module,
             transform: transforms.Compose, device: torch.device) -> None:
        img_files = [f for f in cluster_dir.iterdir() if f.suffix.lower() in DEFAULT_CONFIG['image_extensions']]
        features = []
        valid_files = []
        batch_size = 32
        for i in range(0, len(img_files), batch_size):
            batch = []
            for img_file in img_files[i:i + batch_size]:
                try:
                    img = Image.open(img_file).convert('RGB')
                    batch.append(img)
                    valid_files.append(img_file)
                except Exception as e:
                    logging.error(f"处理图片失败 {img_file}: {e}")
            if batch:
                tensors = torch.stack([transform(img) for img in batch]).to(device)
                with torch.no_grad():
                    batch_features = model(tensors)
                features.append(batch_features.cpu().numpy())
        if not features:
            return
        features = np.concatenate(features)
        features = torch.tensor(features, device=device)
        features = F.normalize(features, p=2, dim=1)
        sim_matrix = torch.mm(features, features.T)
        ordered_indices = self._optimized_ordering(sim_matrix)
        self._rename_files(cluster_dir, valid_files, ordered_indices)

    def _optimized_ordering(self, sim_matrix: torch.Tensor) -> List[int]:
        n = sim_matrix.size(0)
        ordered = [0]
        remaining = set(range(1, n))
        while remaining:
            current = ordered[-1]
            similarities = sim_matrix[current]
            best = max(remaining, key=lambda x: similarities[x].item())
            ordered.append(best)
            remaining.remove(best)
        return ordered

    def _rename_files(self, cluster_dir: Path, files: List[Path], order: List[int]) -> None:
        for idx, orig_idx in enumerate(order):
            old_path = files[orig_idx]
            new_name = f"{idx:05d}_{old_path.name}"
            new_path = cluster_dir / new_name
            old_path.rename(new_path)

# -------------------- 主逻辑类 --------------------
class ImageCluster(QThread):
    progress_updated = pyqtSignal(float, str)
    file_info_updated = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.cancel_flag = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.clustering_strategy = FaissKMeansClustering(
            max_clusters=config['max_clusters'],
            niter=config['niter']
        )
        self.sorting_strategy = SimilaritySorting()
        self.feature_extractor = FeatureExtractor(self.device)
        self.sorting_model = self._load_sorting_model()

    def _load_sorting_model(self) -> Tuple[torch.nn.Module, transforms.Compose]:
        model = timm.create_model('mobilenetv4_hybrid_large.e600_r384_in1k', pretrained=False, num_classes=0)
        checkpoint = torch.load(DEFAULT_CONFIG['sorting_model_path'], map_location=self.device)
        model.load_state_dict(self._filter_model_params(model.state_dict(), checkpoint))
        model = model.to(self.device).eval()
        data_config = timm.data.resolve_model_data_config(model)
        transform = timm.data.create_transform(**data_config, is_training=False)
        return model, transform

    @staticmethod
    def _filter_model_params(model_dict: Dict, pretrained_dict: Dict) -> Dict:
        return {k: v for k, v in pretrained_dict.items() if k in model_dict}

    def run(self):
        try:
            self._update_progress(0, "初始化...")
            input_dir = Path(self.config['input_dir'])
            output_dir = Path(self.config['output_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)

            # 特征提取
            self._update_progress(0, "特征提取")
            features, valid_files = self._extract_features(input_dir, output_dir)
            if self.cancel_flag:
                return

            # 聚类
            self._update_progress(30, "聚类分析")
            cluster_ids = self.clustering_strategy.cluster(features)
            if self.cancel_flag:
                return

            # 整理文件
            self._update_progress(50, "整理文件")
            self._organize_files(valid_files, cluster_ids, output_dir)
            if self.cancel_flag:
                return

            # 可视化
            self._visualize(features, cluster_ids, output_dir)

            # 排序
            if self.config['enable_sorting']:
                self._update_progress(80, "排序文件")
                self._sort_clusters(output_dir)

            self._update_progress(100, "完成")
            self.finished.emit(True, "处理完成")

        except Exception as e:
            logging.error(f"处理失败: {e}")
            self.finished.emit(False, str(e))

    def _extract_features(self, input_dir: Path, output_dir: Path) -> Tuple[np.ndarray, List[Path]]:
        cache_file = output_dir / 'features_cache.joblib'
        if cache_file.exists():
            try:
                return joblib.load(cache_file)
            except Exception as e:
                logging.warning(f"加载缓存失败: {e}")

        img_files = FileProcessor.get_image_files(input_dir)
        features = []
        valid_files = []
        batch_size = 32

        for i in range(0, len(img_files), batch_size):
            if self.cancel_flag:
                return None, []

            batch = []
            for img_file in img_files[i:i + batch_size]:
                try:
                    img = Image.open(img_file).convert('RGB')
                    batch.append(img)
                    valid_files.append(img_file)
                except Exception as e:
                    logging.error(f"处理图片失败 {img_file}: {e}")

            if batch:
                features.append(self.feature_extractor.extract_batch(batch))

            progress = (i / len(img_files)) * 30
            self._update_progress(progress, "特征提取")

        features = np.concatenate(features) if features else np.array([])
        joblib.dump((features, valid_files), cache_file)
        return features, valid_files

    def _organize_files(self, files: List[Path], cluster_ids: np.ndarray, output_dir: Path) -> None:
        clusters = {cid: output_dir / f'cluster_{cid}' for cid in np.unique(cluster_ids)}
        for cid in clusters.values():
            cid.mkdir(exist_ok=True)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for file, cid in zip(files, cluster_ids):
                dst = clusters[cid] / file.name
                futures.append(executor.submit(FileProcessor.safe_copy, file, dst))

                raw_file = self._find_raw_file(file)
                if raw_file:
                    futures.append(executor.submit(FileProcessor.safe_copy, raw_file, clusters[cid] / raw_file.name))

            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"文件操作失败: {e}")

    def _find_raw_file(self, img_path: Path) -> Optional[Path]:
        for ext in DEFAULT_CONFIG['raw_extensions']:
            raw_path = img_path.with_suffix(ext)
            if raw_path.exists():
                return raw_path
        return None

    def _sort_clusters(self, output_dir: Path) -> None:
        cluster_dirs = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith('cluster_')]
        for cluster_dir in cluster_dirs:
            if self.cancel_flag:
                return
            self.sorting_strategy.sort(cluster_dir, *self.sorting_model, self.device)

    def _visualize(self, features: np.ndarray, cluster_ids: np.ndarray, output_dir: Path) -> None:
        n_samples = features.shape[0]
        # 动态调整 perplexity 参数
        if n_samples > 1:
            perplexity = min(30, n_samples - 1)
        else:
            perplexity = 5  # 当样本数量为 1 时，设置一个较小的值

        tsne = TSNE(n_components=2, perplexity=perplexity)
        reduced = tsne.fit_transform(features)
        plt.scatter(reduced[:, 0], reduced[:, 1], c=cluster_ids, cmap='tab20', alpha=0.6)
        plt.savefig(output_dir / 'clusters_visual.png')
        plt.close()

    def _update_progress(self, value: float, stage: str):
        self.progress_updated.emit(value, stage)

    def cancel(self):
        self.cancel_flag = True

# -------------------- PyQt5 GUI界面类 --------------------
class DragDropLineEdit(QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls and urls[0].isLocalFile():
            path = urls[0].toLocalFile()
            if os.path.isdir(path):
                self.setText(path)

# 在原有代码基础上更新ImageClusterGUI类的init_ui方法和其他相关样式
class ImageClusterGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能图像聚类工具")
        self.setWindowIcon(QIcon('icon.ico'))
        self.setGeometry(100, 100, 1000, 600)  # 增大窗口尺寸
        self.setup_styles()
        self.init_ui()

    def setup_styles(self):
        # 现代扁平化风格样式表
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8f9fa;
            }
            QWidget {
                font-family: 'Segoe UI', 'Microsoft YaHei';
                font-size: 13px;
            }
            QLabel {
                color: #495057;
                font-size: 13px;
            }
            QLineEdit {
                background: white;
                border: 1px solid #ced4da;
                border-radius: 4px;
                padding: 8px;
                font-size: 13px;
                selection-background-color: #4dabf7;
            }
            QLineEdit:hover {
                border: 1px solid #4dabf7;
            }
            QPushButton {
                background-color: #4dabf7;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-size: 13px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #339af0;
            }
            QPushButton:pressed {
                background-color: #1c7ed6;
            }
            QPushButton:disabled {
                background-color: #adb5bd;
            }
            QComboBox {
                background: white;
                border: 1px solid #ced4da;
                border-radius: 4px;
                padding: 6px;
                min-width: 100px;
            }
            QComboBox::drop-down {
                width: 20px;
                border-left: 1px solid #ced4da;
            }
            QProgressBar {
                border: 1px solid #ced4da;
                border-radius: 4px;
                background: white;
                text-align: center;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #69db7c;
                border-radius: 3px;
            }
            QCheckBox {
                spacing: 5px;
                color: #495057;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
            QGroupBox {
                border: 1px solid #dee2e6;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 15px;
                font-weight: bold;
                color: #343a40;
            }
        """)

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # 输入输出组
        io_group = QGroupBox("输入输出设置")
        io_layout = QVBoxLayout(io_group)
        io_layout.setContentsMargins(15, 15, 15, 15)
        io_layout.setSpacing(12)

        # 输入路径
        input_layout = QHBoxLayout()
        self.input_edit = DragDropLineEdit()
        self.input_edit.setPlaceholderText("拖放文件夹或点击选择...")
        input_btn = QPushButton()
        input_btn.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        input_btn.setToolTip("选择输入文件夹")
        input_btn.clicked.connect(self.select_input)
        input_layout.addWidget(QLabel("输入文件夹:"), stretch=0)
        input_layout.addWidget(self.input_edit, stretch=3)
        input_layout.addWidget(input_btn, stretch=0)

        # 输出路径
        output_layout = QHBoxLayout()
        self.output_edit = DragDropLineEdit()
        self.output_edit.setPlaceholderText("拖放文件夹或点击选择...")
        output_btn = QPushButton()
        output_btn.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        output_btn.setToolTip("选择输出文件夹")
        output_btn.clicked.connect(self.select_output)
        output_layout.addWidget(QLabel("输出文件夹:"), stretch=0)
        output_layout.addWidget(self.output_edit, stretch=3)
        output_layout.addWidget(output_btn, stretch=0)

        io_layout.addLayout(input_layout)
        io_layout.addLayout(output_layout)

        # 参数设置组
        param_group = QGroupBox("聚类参数")
        param_layout = QHBoxLayout(param_group)
        param_layout.setContentsMargins(15, 15, 15, 15)
        param_layout.setSpacing(20)

        self.max_clusters = QComboBox()
        self.max_clusters.addItems(map(str, DEFAULT_CONFIG['max_clusters_options']))
        self.max_clusters.setCurrentText(str(DEFAULT_CONFIG['default_max_clusters']))
        
        self.niter = QComboBox()
        self.niter.addItems(map(str, DEFAULT_CONFIG['niter_options']))
        self.niter.setCurrentText(str(DEFAULT_CONFIG['default_niter']))

        param_layout.addWidget(QLabel("最大聚类数:"), stretch=0)
        param_layout.addWidget(self.max_clusters, stretch=1)
        param_layout.addWidget(QLabel("迭代次数:"), stretch=0)
        param_layout.addWidget(self.niter, stretch=1)
        param_layout.addStretch(2)

        # 操作控制组
        control_group = QWidget()
        control_layout = QHBoxLayout(control_group)
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.setSpacing(15)

        self.start_btn = QPushButton("开始聚类")
        self.start_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.start_btn.clicked.connect(self.start_clustering)
        
        self.cancel_btn = QPushButton("取消操作")
        self.cancel_btn.setIcon(self.style().standardIcon(QStyle.SP_DialogCancelButton))
        self.cancel_btn.clicked.connect(self.cancel_clustering)
        self.cancel_btn.setEnabled(False)
        
        self.open_btn = QPushButton("打开结果")
        self.open_btn.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(self.open_output)

        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.cancel_btn)
        control_layout.addStretch(1)
        control_layout.addWidget(self.open_btn)

        # 进度显示
        progress_group = QGroupBox("处理进度")
        progress_layout = QVBoxLayout(progress_group)
        progress_layout.setContentsMargins(15, 15, 15, 15)
        progress_layout.setSpacing(10)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p% (%v/%m)")
        
        self.stage_label = QLabel("准备就绪")
        self.stage_label.setAlignment(Qt.AlignCenter)
        self.stage_label.setStyleSheet("color: #868e96; font-style: italic;")

        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.stage_label)

        # 选项设置
        option_group = QGroupBox("高级选项")
        option_layout = QHBoxLayout(option_group)
        option_layout.setContentsMargins(15, 15, 15, 15)
        option_layout.setSpacing(20)

        self.sort_check = QCheckBox("智能排序")
        self.sort_check.setChecked(True)
        self.sort_check.setToolTip("启用基于相似性的智能文件排序")
        
        self.delete_check = QCheckBox("清理源文件夹")
        self.delete_check.setToolTip("处理完成后自动删除源文件夹内容")
        self.delete_warning = QLabel("⚠️ 此操作不可恢复！")
        self.delete_warning.setStyleSheet("color: #fa5252; font-weight: bold;")

        option_layout.addWidget(self.sort_check)
        option_layout.addWidget(self.delete_check)
        option_layout.addWidget(self.delete_warning)
        option_layout.addStretch()

        # 文件信息
        self.file_info = QLabel()
        self.file_info.setAlignment(Qt.AlignCenter)
        self.file_info.setStyleSheet("""
            background-color: #e9ecef;
            border-radius: 4px;
            padding: 12px;
            color: #495057;
        """)

        # 组装主界面
        layout.addWidget(io_group)
        layout.addWidget(param_group)
        layout.addWidget(option_group)
        layout.addWidget(progress_group)
        layout.addWidget(control_group)
        layout.addWidget(self.file_info)

        # 初始化文件统计
        self.input_edit.textChanged.connect(self.update_file_count)
        self.update_file_count()

    def select_input(self):
        path = QFileDialog.getExistingDirectory(self, "选择输入文件夹")
        if path:
            self.input_edit.setText(path)
            self.update_file_count()

    def select_output(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if path:
            self.output_edit.setText(path)

    def update_file_count(self):
        input_path = self.input_edit.text()
        if input_path:
            try:
                count = len(FileProcessor.get_image_files(Path(input_path)))
                self.file_info.setText(f"检测到 {count} 张待处理图片")
            except Exception as e:
                self.file_info.setText(f"文件统计失败: {str(e)}")

    def start_clustering(self):
        input_dir = self.input_edit.text()
        output_dir = self.output_edit.text()

        if not input_dir or not output_dir:
            QMessageBox.critical(self, "错误", "请先选择输入和输出文件夹")
            return

        # 准备配置参数
        config = {
            'input_dir': input_dir,
            'output_dir': output_dir,
            'max_clusters': int(self.max_clusters.currentText()),
            'niter': int(self.niter.currentText()),
            'enable_sorting': self.sort_check.isChecked()
        }

        # 初始化工作线程
        self.worker = ImageCluster(config)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.file_info_updated.connect(self.file_info.setText)
        self.worker.finished.connect(self.handle_finished)

        # 更新界面状态
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.open_btn.setEnabled(False)
        self.worker.start()

    def update_progress(self, value: float, stage: str):
        self.progress_bar.setValue(int(value))
        self.stage_label.setText(f"{stage} ({value:.1f}%)")

    def cancel_clustering(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.cancel_btn.setEnabled(False)
            self.file_info.setText("正在停止任务...")
            QMessageBox.information(self, "提示", "已发送取消请求，正在停止当前任务")

    def handle_finished(self, success: bool, message: str):
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

        if success:
            self.open_btn.setEnabled(True)
            QMessageBox.information(self, "完成", message)
            if self.delete_check.isChecked():
                self.delete_input_folder()
        else:
            QMessageBox.critical(self, "错误", message)

    def delete_input_folder(self):
        input_path = Path(self.input_edit.text())
        try:
            if input_path.exists():
                shutil.rmtree(input_path)
                QMessageBox.information(self, "提示", "输入文件夹已成功删除")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法删除输入文件夹:\n{str(e)}")

    def open_output(self):
        output_dir = self.output_edit.text()
        if not output_dir:
            return

        try:
            if sys.platform == 'win32':
                os.startfile(output_dir)
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', output_dir])
            else:
                subprocess.Popen(['xdg-open', output_dir])
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法打开文件夹:\n{str(e)}")

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self, '后台任务运行中',
                '聚类任务仍在运行，确定要退出吗？',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.worker.cancel()
                self.worker.wait(2000)
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create('Fusion'))
    window = ImageClusterGUI()
    window.show()
    sys.exit(app.exec_())