"""
dmp_trainer.py — обучение DMP на траектории руки из демонстрации

Pipeline:
    траектория (T, 3) из hand_tracker
        → нормализация по времени (resample до N точек)
        → обучение DMP через pydmps
        → сохранение в .pkl

Использование:
    trainer = DMPTrainer()
    dmp = trainer.train(trajectory)
    trainer.save(dmp, "policy/dmps/demo_cube_blue.pkl")

    # Проверка воспроизведения:
    trainer.verify(dmp, trajectory, plot=True)
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

try:
    import pydmps
    import pydmps.dmp_discrete
except ImportError:
    raise SystemExit("pydmps не установлен: pip install pydmps")


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------

def resample_trajectory(traj: np.ndarray, n_points: int = 100) -> np.ndarray:
    """
    Ресемплирует траекторию до фиксированного числа точек.

    DMP обучается на равномерной временной сетке — поэтому нужно
    привести все траектории к одинаковому числу точек независимо
    от длины оригинальной записи (у кого-то 200 кадров, у кого-то 400).

    Args:
        traj     : (T, 3) исходная траектория
        n_points : целевое число точек

    Returns:
        (n_points, 3)
    """
    T = traj.shape[0]
    if T == 0:
        raise ValueError("Пустая траектория")

    old_t = np.linspace(0, 1, T)
    new_t = np.linspace(0, 1, n_points)

    resampled = np.zeros((n_points, 3), dtype=np.float32)
    for axis in range(3):
        resampled[:, axis] = np.interp(new_t, old_t, traj[:, axis])

    return resampled


# ---------------------------------------------------------------------------
# DMPTrainer
# ---------------------------------------------------------------------------

class DMPTrainer:
    """
    Обучает DMP на 3D траектории и сохраняет результат.

    Args:
        n_bfs    : число радиальных базисных функций (больше = точнее форма,
                   но больше риск переобучения). 50 хватает для большинства движений.
        n_points : число точек для ресемплинга траектории
        dt       : временной шаг DMP (0.01 = 100 шагов в единицу времени)
    """

    def __init__(self,
                 n_bfs: int = 50,
                 n_points: int = 100,
                 dt: float = 0.01):
        self.n_bfs    = n_bfs
        self.n_points = n_points
        self.dt       = dt

    def train(self, trajectory: np.ndarray) -> pydmps.dmp_discrete.DMPs_discrete:
        """
        Обучает DMP на траектории.

        Args:
            trajectory: (T, 3) массив точек [x, y, z] в метрах

        Returns:
            обученный DMP объект
        """
        if trajectory.shape[0] == 0:
            raise ValueError("Траектория пустая — рука не была найдена в кадрах")

        # 1. Ресемплинг до фиксированного числа точек
        traj_resampled = resample_trajectory(trajectory, self.n_points)
        logger.info(f"  Траектория: {trajectory.shape[0]} → {self.n_points} точек")

        # 2. Создаём DMP для 3 степеней свободы (x, y, z)
        dmp = pydmps.dmp_discrete.DMPs_discrete(
            n_dmps=3,         # число измерений
            n_bfs=self.n_bfs, # число базисных функций
            dt=self.dt,
        )

        # 3. Обучаем на траектории
        # pydmps ожидает массив (n_dmps, n_points) — транспонируем
        dmp.imitate_path(y_des=traj_resampled.T)

        # Сохраняем оригинальную траекторию в объекте DMP для верификации
        dmp._demo_trajectory = traj_resampled
        dmp._demo_start      = traj_resampled[0].copy()
        dmp._demo_goal       = traj_resampled[-1].copy()

        logger.info(f"  DMP обучен: start={dmp._demo_start}, goal={dmp._demo_goal}")
        return dmp

    def verify(self,
               dmp: pydmps.dmp_discrete.DMPs_discrete,
               original_trajectory: np.ndarray,
               plot: bool = False) -> float:
        """
        Воспроизводит DMP и считает ошибку относительно оригинала.

        Returns:
            средняя евклидова ошибка в метрах
        """
        traj_orig = resample_trajectory(original_trajectory, self.n_points)

        # Воспроизводим с теми же start/goal что и в демо
        dmp.reset_state()
        y_track, _, _ = dmp.rollout()  # (n_points, n_dmps)

        error = float(np.mean(np.linalg.norm(y_track - traj_orig, axis=1)))
        logger.info(f"  Средняя ошибка воспроизведения: {error*100:.1f} см")

        if plot:
            self._plot(traj_orig, y_track)

        return error

    def _plot(self, original: np.ndarray, reproduced: np.ndarray):
        try:
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(14, 5))

            # 3D сравнение
            ax3d = fig.add_subplot(131, projection="3d")
            ax3d.plot(*original.T,    "b-",  linewidth=2,   label="оригинал")
            ax3d.plot(*reproduced.T,  "r--", linewidth=1.5, label="DMP")
            ax3d.scatter(*original[0],    color="green", s=60, zorder=5)
            ax3d.scatter(*original[-1],   color="red",   s=60, zorder=5)
            ax3d.set_title("3D траектория")
            ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
            ax3d.legend(fontsize=8)

            # По осям отдельно
            labels = ["X (м)", "Y (м)", "Z (м)"]
            t = np.linspace(0, 1, self.n_points)
            for i, (ax_idx, label) in enumerate(zip([132, 133], ["X", "Y"])):
                ax = fig.add_subplot(ax_idx)
                ax.plot(t, original[:, i],   "b-",  linewidth=2,   label="оригинал")
                ax.plot(t, reproduced[:, i], "r--", linewidth=1.5, label="DMP")
                ax.set_title(f"Ось {label}")
                ax.set_xlabel("время (норм.)")
                ax.set_ylabel(labels[i])
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

            plt.suptitle("Верификация DMP", fontsize=12)
            plt.tight_layout()
            plt.show()
        except ImportError:
            logger.warning("matplotlib не установлен: pip install matplotlib")

    def save(self, dmp: pydmps.dmp_discrete.DMPs_discrete, path: str | Path):
        """Сохраняет обученный DMP в .pkl файл."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(dmp, f)
        logger.info(f"  DMP сохранён: {path}")

    @staticmethod
    def load(path: str | Path) -> pydmps.dmp_discrete.DMPs_discrete:
        """Загружает DMP из .pkl файла."""
        with open(path, "rb") as f:
            return pickle.load(f)

    def train_and_save(self,
                       demo_path: str | Path,
                       output_path: str | Path,
                       plot: bool = False) -> float:
        """
        Удобный метод: читает демо → извлекает траекторию → обучает → сохраняет.

        Returns:
            ошибка воспроизведения в метрах
        """
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from assets.hand_tracker import HandTracker

        logger.info(f"Обработка демо: {demo_path}")

        with HandTracker() as tracker:
            trajectory = tracker.extract_trajectory(demo_path)

        if trajectory.shape[0] == 0:
            raise ValueError(f"Не удалось извлечь траекторию из {demo_path}")

        dmp   = self.train(trajectory)
        error = self.verify(dmp, trajectory, plot=plot)
        self.save(dmp, output_path)

        return error


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    ap = argparse.ArgumentParser(description="Обучение DMP на демонстрации")
    ap.add_argument("demo_path",   help="Путь к папке демо")
    ap.add_argument("output_path", help="Куда сохранить .pkl, напр. policy/dmps/cube_blue.pkl")
    ap.add_argument("--n-bfs",     type=int,   default=50,   help="Число базисных функций")
    ap.add_argument("--n-points",  type=int,   default=100,  help="Точек для ресемплинга")
    ap.add_argument("--plot",      action="store_true",       help="Показать график")
    args = ap.parse_args()

    trainer = DMPTrainer(n_bfs=args.n_bfs, n_points=args.n_points)
    error   = trainer.train_and_save(args.demo_path, args.output_path, plot=args.plot)

    print(f"\nГотово. Ошибка воспроизведения: {error*100:.1f} см")
    print(f"Сохранено: {args.output_path}")
