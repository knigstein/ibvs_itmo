"""
Предобработка изображения для YOLO:
- Подавление теней
- Усиление контраста объекта
- Сохранение цвета для YOLO
"""
import cv2
import numpy as np
from typing import Tuple, Optional


def preprocess_image_for_yolo(
    bgr: np.ndarray,
    hsv_lower: Tuple[int, int, int] = (5, 40, 50),
    hsv_upper: Tuple[int, int, int] = (35, 255, 255),
    suppress_shadow: bool = True,
    enhance_contrast: bool = True,
    blur_background: bool = True,
    debug: bool = False
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Предобработка изображения перед подачей в YOLO.
    
    Args:
        bgr: исходное изображение (H, W, 3)
        hsv_lower: нижняя граница HSV для объекта
        hsv_upper: верхняя граница HSV для объекта
        suppress_shadow: подавлять тени (работает с V-каналом)
        enhance_contrast: усиливать контраст объекта
        blur_background: размывать фон (помогает YOLO сфокусироваться)
        debug: если True, возвращает маску для отладки
    
    Returns:
        processed_img: обработанное изображение для YOLO
        mask: бинарная маска объекта (если debug=True)
    """
    
    # ========== 1. HSV-сегментация ==========
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    
    # Бинарная маска по цвету
    mask = cv2.inRange(hsv, np.array(hsv_lower, dtype=np.uint8), 
                            np.array(hsv_upper, dtype=np.uint8))
    
    # Морфология для удаления шума
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # ========== 2. Подавление теней ==========
    if suppress_shadow:
        # Извлекаем V-канал (яркость)
        v_channel = hsv[:, :, 2]
        
        # Тень = тёмные области, но НЕ чёрные (чёрные — это фон)
        # Создаём маску тени: яркость 20-80 (примерно)
        shadow_mask = cv2.inRange(v_channel, 20, 80)
        
        # Убираем из маски тени объекты (чтобы не стереть сам объект)
        shadow_mask = cv2.bitwise_and(shadow_mask, shadow_mask, mask=cv2.bitwise_not(mask))
        
        # Морфология для тени
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Осветляем тени на оригинальном изображении
        # Коэффициент осветления: чем темнее тень, тем сильнее осветляем
        v_float = v_channel.astype(np.float32)
        v_enhanced = v_float.copy()
        
        # Где тень — увеличиваем яркость
        shadow_regions = shadow_mask > 0
        v_enhanced[shadow_regions] = np.minimum(v_enhanced[shadow_regions] * 1.5, 255)
        
        # Заменяем V-канал
        hsv_enhanced = hsv.copy()
        hsv_enhanced[:, :, 2] = np.clip(v_enhanced, 0, 255).astype(np.uint8)
        
        # Конвертируем обратно в BGR
        bgr_processed = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
    else:
        bgr_processed = bgr.copy()
    
    # ========== 3. Усиление контраста объекта ==========
    if enhance_contrast:
        # Применяем маску к изображению
        masked_img = cv2.bitwise_and(bgr_processed, bgr_processed, mask=mask)
        
        # Для областей внутри маски — усиливаем контраст
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(masked_img, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # Применяем CLAHE только внутри маски
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel)
        
        # Смешиваем усиленный L-канал с оригиналом по маске
        lab[:, :, 0] = np.where(mask > 0, l_enhanced, l_channel)
        bgr_processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # ========== 4. Размытие фона ==========
    if blur_background:
        # Создаём инвертированную маску (фон)
        bg_mask = cv2.bitwise_not(mask)
        
        # Размываем всё изображение
        blurred = cv2.GaussianBlur(bgr_processed, (15, 15), 0)
        
        # Смешиваем: объект чёткий, фон размытый
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        bgr_processed = np.where(
            mask_3ch > 0,
            bgr_processed,
            blurred
        ).astype(np.uint8)
    
    # ========== 5. Дополнительная постобработка ==========
    # Убираем возможные артефакты (шум после обработки)
    bgr_processed = cv2.bilateralFilter(bgr_processed, 5, 75, 75)
    
    if debug:
        return bgr_processed, mask
    else:
        return bgr_processed, None


# ========== Интеграция с YOLO-детектором ==========

class YOLOFeatureDetectorPreprocessed:
    """
    YOLO-детектор с предобработкой изображения.
    Подавляет тени, усиливает контраст объекта.
    """
    
    def __init__(self, config: dict):
        from ultralytics import YOLO
        
        self._config = config
        self._model = YOLO(config.get("weights_path", "yolov8n.pt"))
        self._min_conf = float(config.get("min_conf", 0.5))
        self._ema_alpha = float(config.get("ema_alpha", 0.35))
        self._corners_ema = None
        
        # Параметры HSV для предобработки
        self._hsv_lower = tuple(config.get("hsv_lower", [5, 40, 50]))
        self._hsv_upper = tuple(config.get("hsv_upper", [35, 255, 255]))
        self._use_preprocessing = config.get("use_preprocessing", True)
    
    def detect(self, bgr: np.ndarray) -> 'CubeSegmentationResult':
        from dataclasses import dataclass
        from typing import Optional, Dict, Any
        
        @dataclass
        class CubeSegmentationResult:
            corners: Optional[np.ndarray]
            ok: bool
            meta: Dict[str, Any]
        
        if bgr is None or bgr.size == 0:
            return CubeSegmentationResult(None, False, {"reason": "empty_frame"})
        
        # ========== ПРЕДОБРАБОТКА ==========
        if self._use_preprocessing:
            img_for_yolo, mask = preprocess_image_for_yolo(
                bgr,
                hsv_lower=self._hsv_lower,
                hsv_upper=self._hsv_upper,
                suppress_shadow=True,
                enhance_contrast=True,
                blur_background=True,
                debug=True
            )
        else:
            img_for_yolo = bgr
            mask = None
        
        # ========== ДЕТЕКЦИЯ YOLO ==========
        results = self._model(img_for_yolo, verbose=False, conf=self._min_conf)
        
        if len(results[0].boxes) == 0:
            # Если не сработало — пробуем на оригинальном изображении
            if self._use_preprocessing:
                results = self._model(bgr, verbose=False, conf=self._min_conf * 0.8)
                if len(results[0].boxes) == 0:
                    return CubeSegmentationResult(None, False, {"n_detections": 0})
            else:
                return CubeSegmentationResult(None, False, {"n_detections": 0})
        
        # Извлекаем бокс
        box = results[0].boxes.xyxy[0].cpu().numpy()
        conf = float(results[0].boxes.conf[0])
        cls_id = int(results[0].boxes.cls[0])
        
        x1, y1, x2, y2 = box
        corners = np.array([
            [x1, y1], [x2, y1],
            [x2, y2], [x1, y2]
        ], dtype=np.float32)
        
        # Порядок углов
        corners = self._order_corners(corners)
        
        # EMA-сглаживание
        if self._ema_alpha > 0.0 and self._corners_ema is not None:
            corners = self._ema_alpha * corners + (1.0 - self._ema_alpha) * self._corners_ema
        self._corners_ema = corners.copy()
        
        meta = {
            "n_detections": len(results[0].boxes),
            "confidence": conf,
            "class_id": cls_id,
            "center": corners.mean(axis=0).tolist(),
            "area": float((x2 - x1) * (y2 - y1)),
            "bbox": [x1, y1, x2, y2],
            "preprocessing_used": self._use_preprocessing,
        }
        
        if mask is not None:
            meta["mask_area"] = int(cv2.countNonZero(mask))
        
        return CubeSegmentationResult(corners, True, meta)
    
    def _order_corners(self, pts: np.ndarray) -> np.ndarray:
        """Приводит 4 точки к порядку: TL, TR, BR, BL"""
        pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
        s = pts.sum(axis=1)
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        rem = [i for i in range(4) if not np.allclose(pts[i], tl) and not np.allclose(pts[i], br)]
        others = pts[rem]
        if others.shape[0] != 2:
            return pts
        tr, bl = (others[0], others[1]) if others[0, 0] < others[1, 0] else (others[1], others[0])
        return np.stack([tl, tr, br, bl], axis=0)