import json
from pathlib import Path
from typing import Tuple, Union

import numpy as np


class IBVS:
    '''Класс визуального сервоуправления по изображению.'''

    def __init__(
        self,
        camera_config: dict,
        active_directions: Union[list, np.ndarray] = None,
        control_coefficient: Union[list, np.ndarray] = None,
        exit_treshhold: float = 0.013,
        damping: float = 1e-4,
    ):
        '''Создаёт экземпляр IBVS.

        Параметры
        ----------
        camera_config : dict
            Словарь параметров камеры: ``focal_length``, ``principal_point``, ``features``
            (эталонные признаки в пикселях).

        active_directions : list или np.ndarray
            Индексы разрешённых направлений движения «камеры» в декартовом пространстве:
            всего 6 осей — три поступательных и три вращательных: x, y, z, Rx, Ry, Rz
            (индексы 0 … 5). По умолчанию активны все шесть.
            Пример: ``[0, 1, 5]`` — только x, y и Rz. Элементы не должны повторяться.

        control_coefficient : list или np.ndarray
            Коэффициенты усиления по закону управления: ровно 6 чисел — по одному на каждое
            направление (x … Rz). Для активных осей множители берутся из соответствующих позиций.

        damping : float
            Параметр тихоновского демпфирования при псевдообращении усечённой матрицы
            взаимодействия (устойчивость при плохой обусловленности).

        '''
        self._features_desired_positions = np.asarray(camera_config["features"], dtype=float)
        self._principal_point = np.asarray(camera_config["principal_point"], dtype=float).reshape(2)
        self._focal_length = np.asarray(camera_config["focal_length"], dtype=float).reshape(2)
        if active_directions is None:
            active_directions = [0, 1, 2, 3, 4, 5]
        if control_coefficient is None:
            control_coefficient = [0.3, 0.3, 0.3, 0.5, 0.5, 0.5]
        self._control_coefficient = np.asarray(control_coefficient, dtype=float).reshape(6)
        self._active_directions = np.sort(np.asarray(active_directions, dtype=int))
        self._exit_treshhold = exit_treshhold
        self._damping = damping

        if len(active_directions) > 6:
            raise Exception("Число активных направлений не может быть больше 6")

        if len(set(active_directions)) != len(active_directions):
            raise Exception("Индексы в active_directions должны быть уникальными")

        if len(control_coefficient) > 6:
            raise Exception("Длина control_coefficient не может быть больше 6")

        for direction in active_directions:
            if direction >= 6:
                raise ValueError("Индекс в active_directions должен быть меньше 6")

        if self._active_directions.size == 0:
            raise ValueError("active_directions не должен быть пустым")

        self._s_desired = self._stack_normalized(self._features_desired_positions)

    def _stack_normalized(self, features_pixel: np.ndarray) -> np.ndarray:
        '''Собирает вектор длины 2n: для каждой точки пара нормализованных координат (x, y).'''
        features_pixel = np.asarray(features_pixel, dtype=float)
        if features_pixel.ndim == 1:
            features_pixel = features_pixel.reshape(1, -1)
        n = features_pixel.shape[0]
        s = np.zeros(2 * n, dtype=float)
        for i in range(n):
            x, y = self.normalize(features_pixel[i])
            s[2 * i] = x
            s[2 * i + 1] = y
        return s

    @property
    def active_directions(self) -> np.ndarray:
        '''Текущий набор индексов активных направлений.'''
        return self._active_directions

    @active_directions.setter
    def active_directions(self, new_active_directions: Union[list, np.ndarray]):
        '''Задаёт новый набор активных направлений.'''
        arr = np.asarray(new_active_directions, dtype=int)
        if arr.size == 0:
            raise ValueError("active_directions не должен быть пустым")
        self._active_directions = np.sort(arr)

    @property
    def control_coefficient(self) -> np.ndarray:
        '''Вектор коэффициентов усиления (6 элементов).'''
        return self._control_coefficient

    @control_coefficient.setter
    def control_coefficient(self, new_control_coefficient: Union[list, np.ndarray]):
        '''Задаёт коэффициенты усиления по шести направлениям.'''
        self._control_coefficient = np.asarray(new_control_coefficient, dtype=float).reshape(6)

    @property
    def s_desired(self) -> np.ndarray:
        '''Эталонный вектор признаков в нормализованных координатах изображения, форма (2n,).'''
        return self._s_desired

    @property
    def exit_threshold(self) -> float:
        '''Порог нормы ошибки по признакам для сходимости.'''
        return self._exit_treshhold

    def calculate_interaction_matrix(self, x, y, Z: float):
        '''Матрица взаимодействия L_i для одной точки в нормализованной плоскости изображения
        (фокусное расстояние сведено к масштабу 1).

        Точка задаётся нормализованными координатами (x, y); глубина Z > 0 в системе отсчёта
        камеры, метры.

        Порядок компонент винта: [vx, vy, vz, wx, wy, wz] в базисе камеры.
        '''
        if Z <= 0:
            raise ValueError("Z должна быть положительной")

        L = np.array(
            [
                [-1.0 / Z, 0.0, x / Z, x * y, -(1.0 + x * x), y],
                [0.0, -1.0 / Z, y / Z, 1.0 + y * y, -x * y, -x],
            ],
            dtype=float,
        )
        return L

    def normalize(self, pixel):
        '''Переводит пиксель (u, v) в нормализованные координаты (x, y).'''
        pixel = np.asarray(pixel, dtype=float).reshape(-1)
        if pixel.size < 2:
            raise ValueError("Ожидается пара (u, v)")
        u, v = pixel[0], pixel[1]
        fx, fy = self._focal_length[0], self._focal_length[1]
        cx, cy = self._principal_point[0], self._principal_point[1]
        x = (u - cx) / fx
        y = (v - cy) / fy
        return float(x), float(y)

    def calculate_error(self, features_current_positions: Union[list, np.ndarray]) -> np.ndarray:
        '''Ошибка по признакам e = s − s* в нормализованных координатах'''
        s = self._stack_normalized(features_current_positions)
        if s.shape != self._s_desired.shape:
            raise ValueError(
                f"Несовпадение числа признаков: текущий вектор {s.shape[0]} vs эталон {self._s_desired.shape[0]}"
            )
        return s - self._s_desired

    def get_jacobian(self, features_current_positions: Union[list, np.ndarray], Z) -> np.ndarray:
        '''
        водная матрица взаимодействия L размера (2n, 6).

        Z — скаляр или массив формы с глубиной каждой точки.
        '''
        features_current_positions = np.asarray(features_current_positions, dtype=float)
        if features_current_positions.ndim == 1:
            features_current_positions = features_current_positions.reshape(1, -1)
        n = features_current_positions.shape[0]
        Z = np.asarray(Z, dtype=float)
        if Z.shape == ():
            Z = np.full(n, float(Z))
        if Z.shape != (n,):
            raise ValueError("Z должна быть скаляром или массивом длины n")

        rows = []
        for i in range(n):
            x, y = self.normalize(features_current_positions[i])
            rows.append(self.calculate_interaction_matrix(x, y, float(Z[i])))
        return np.vstack(rows)

    def _damped_pinv(self, A: np.ndarray) -> np.ndarray:
        '''Демпфированное псевдообращение для связи v = −A⁺ e (регуляризация Тихонова).

        При m ≥ n: (AᵀA + λ²I)⁻¹ Aᵀ; при m < n: Aᵀ (AAᵀ + λ²I)⁻¹.
        '''
        m, n = A.shape
        lam2 = self._damping ** 2
        if m >= n:
            return np.linalg.inv(A.T @ A + lam2 * np.eye(n, dtype=float)) @ A.T
        return A.T @ np.linalg.inv(A @ A.T + lam2 * np.eye(m, dtype=float))

    def calculate_velocity_from_jacobian(
        self,
        total_interaction_matrix: np.ndarray,
        features_current_positions: Union[list, np.ndarray],
    ) -> np.ndarray:
        '''Закон управления: v = − diag(λ_a) ⊙ (L_a⁺ e); на неактивных осях нули.

        Здесь L_a = L[:, active_directions], λ_a — соответствующие элементы control_coefficient.
        '''
        L = np.asarray(total_interaction_matrix, dtype=float)
        e = self.calculate_error(features_current_positions)
        idx = self._active_directions
        L_a = L[:, idx]
        coeff_a = self._control_coefficient[idx]

        L_pinv_a = self._damped_pinv(L_a)
        v_a = -(L_pinv_a @ e) * coeff_a

        v = np.zeros(6, dtype=float)
        v[idx] = v_a
        return v

    def step(
        self,
        features_current_positions: Union[list, np.ndarray],
        Z,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''Один шаг IBVS: возвращает кортеж (винт камеры v_c, ошибка e, матрица L).'''
        L = self.get_jacobian(features_current_positions, Z)
        e = self.calculate_error(features_current_positions)
        v = self.calculate_velocity_from_jacobian(L, features_current_positions)
        return v, e, L

    def is_converged(self, error: np.ndarray) -> bool:
        '''True, если норма ошибки по признакам меньше порога exit_treshhold'''
        return float(np.linalg.norm(error)) < self._exit_treshhold

    def show(self, image, features_current_positions: Union[list, np.ndarray]):
        '''Отрисовка эталонных и текущих признаков на изображении'''
        import cv2

        delta = 10
        for pos in self._features_desired_positions:
            pos = [int(x) for x in pos]
            cv2.line(image, (pos[0] - delta, pos[1]), (pos[0] + delta, pos[1]), (255, 0, 0))
            cv2.line(image, (pos[0], pos[1] - delta), (pos[0], pos[1] + delta), (255, 0, 0))
        if features_current_positions is not None and len(features_current_positions) > 0:
            for pos in features_current_positions:
                pos = [int(x) for x in pos]
                cv2.line(image, (pos[0] - delta, pos[1]), (pos[0] + delta, pos[1]), (0, 0, 255))
                cv2.line(image, (pos[0], pos[1] - delta), (pos[0], pos[1] + delta), (0, 0, 255))
            for i, feature in enumerate(features_current_positions):
                pos1 = feature
                pos2 = self._features_desired_positions[i]
                pos1 = [int(x) for x in pos1]
                pos2 = [int(x) for x in pos2]
                cv2.line(image, tuple(pos1), tuple(pos2), (0, 255, 0))

        cv2.imshow("Features", image)
        cv2.waitKey(1)


if __name__ == '__main__':
    config_path = Path(__file__).resolve().parent / "config" / "camera.json"
    with open(config_path, "r", encoding="utf8") as file:
        camera_config = json.load(file)

    ibvs = IBVS(camera_config)
    # Контроль: нулевая ошибка → нулевая скорость
    v, e, _ = ibvs.step(camera_config["features"], Z=0.5)
    assert np.linalg.norm(v) < 1e-9
    assert np.linalg.norm(e) < 1e-9
    print("Проверка IBVS OK:", "||v||", np.linalg.norm(v), "||e||", np.linalg.norm(e))
