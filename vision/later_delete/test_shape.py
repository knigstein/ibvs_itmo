"""test_local.py — Быстрая проверка GeneralShapeSegmenter на твоих фото."""
import cv2
import numpy as np
import argparse
from pathlib import Path
import sys

from shape_segmenter import GeneralShapeSegmenter, ShapeType

def main():
    parser = argparse.ArgumentParser(description="Тест сегментатора")
    parser.add_argument("image", help="Путь к картинке")
    parser.add_argument("--mode", choices=["color", "contrast", "auto"], default="auto",
                        help="Режим: color (только HSV), contrast (только контраст), auto (гибрид)")
    parser.add_argument("--hsv_l", nargs=3, type=int, default=[5, 50, 40])
    parser.add_argument("--hsv_u", nargs=3, type=int, default=[35, 255, 255])
    parser.add_argument("--min_area", type=float, default=300.0)
    parser.add_argument("--debug", action="store_true", help="Сохранять промежуточные маски")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"❌ Файл не найден: {img_path}")
        sys.exit(1)

    print("📥 Загрузка изображения...")
    img = cv2.imread(str(img_path))
    if img is None:
        print(" Не удалось прочитать файл")
        sys.exit(1)

    # Настройка конфига под выбранный режим
    config = {
        "hsv_lower": args.hsv_l,
        "hsv_upper": args.hsv_u,
        "min_area": args.min_area,
        "use_contrast_fallback": args.mode != "color",
        "shadow_v_threshold": 40,
        "ema_alpha": 0.35
    }

    print(f"⚙️ Запуск в режиме: {args.mode.upper()}")
    detector = GeneralShapeSegmenter(config)
    result = detector.detect(img)

    if not result.ok:
        print("⚠️ Детекция не прошла")
        print(f"   Причина: {result.meta.get('reason', 'unknown')}")
        print(f"   Метод: {result.meta.get('detection_method')}")
        print(f"   Контуров найдено: {result.meta.get('n_contours', 0)}")
        return

    print("\n✅ Успешная детекция!")
    print(f"    Форма: {result.meta['shape_type'].upper()} (уверенность: {result.meta['confidence']:.2f})")
    print(f"   🟢 Метод: {result.meta['detection_method']}")
    print(f"   📏 Площадь: {result.meta['area']:.0f} px² | Solidity: {result.meta['solidity']:.2f}")
    print(f"    Центр: ({result.meta['center_x']:.1f}, {result.meta['center_y']:.1f})")
    print(f"   🔄 Углы [TL, TR, BR, BL]:\n{result.corners.astype(int)}")

    # === ВИЗУАЛИЗАЦИЯ ===
    vis = img.copy()
    pts = result.corners.astype(int)
    cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
    for i, (x, y) in enumerate(pts):
        cv2.circle(vis, (x, y), 5, (255, 0, 0), -1)
        cv2.putText(vis, f"P{i}", (x+8, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    label = f"{result.meta['shape_type'].upper()} ({result.meta['confidence']:.2f})"
    cv2.putText(vis, label, (pts[0][0], pts[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    out_dir = Path("test_output")
    out_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(out_dir / "result.jpg"), vis)
    print(f"\n💾 Результат: {out_dir / 'result.jpg'}")

    if args.debug:
        # Пересоздаём маску для отладки (класс её не возвращает)
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask_color = cv2.inRange(hsv, np.array(args.hsv_l, dtype=np.uint8), np.array(args.hsv_u, dtype=np.uint8))
        mask_contrast = cv2.adaptiveThreshold(cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY), 255,
                                              cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        cv2.imwrite(str(out_dir / "mask_color.jpg"), mask_color)
        cv2.imwrite(str(out_dir / "mask_contrast.jpg"), mask_contrast)
        print(f"🔍 Маски сохранены: mask_color.jpg, mask_contrast.jpg")

    print("\n🎉 Готово. Открой result.jpg и проверь, что рамка на объекте.")

if __name__ == "__main__":
    main()