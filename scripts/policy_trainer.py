"""
policy_trainer.py — обучение и инференс policy для LfD pipeline

Классификатор отображает alpha (shape, color) → action_sequence.

Ядро классификатора: Decision Tree (sklearn).
Признаки: one-hot encoding shape + one-hot encoding color.
Метка: строковое представление action_sequence (например "MOVE(container_1)").

При неизвестном объекте — иерархия fallback:
    1. Точное совпадение (DT предсказывает уверенно)  → exact
    2. Обобщение по color (все того же цвета — одно)  → WARNING + color
    3. Обобщение по shape (все той же формы — одно)   → WARNING + shape
    4. Конфликт / нет данных                          → WARNING + FORWARD

Decision Tree выбран как интерпретируемый классификатор с минимальной
сложностью, соответствующей принципу Оккама при малом числе обучающих
примеров (7 демо, 2 признака).
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Структуры данных
# ---------------------------------------------------------------------------

@dataclass
class PolicyEntry:
    """Одна запись обучающих данных: alpha → action_sequence."""
    shape: str
    color: str
    action_sequence: list[str]
    source_demo_ids: list[str]


@dataclass
class InferenceResult:
    """Результат инференса для одного объекта."""
    action_sequence: list[str]
    confidence: str          # "exact" | "color" | "shape" | "fallback"
    warning: str | None


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def actions_to_str(action_sequence) -> list[str]:
    return [str(a) for a in action_sequence]


def actions_equal(a: list[str], b: list[str]) -> bool:
    return a == b


def encode_label(action_sequence: list[str]) -> str:
    """Список действий → одна строка-метка для классификатора."""
    return " | ".join(action_sequence)


def decode_label(label: str) -> list[str]:
    """Строка-метка → список действий."""
    return label.split(" | ")


# ---------------------------------------------------------------------------
# PolicyTrainer
# ---------------------------------------------------------------------------

class PolicyTrainer:
    """
    Обучает Decision Tree на парах (alpha, action_sequence) из демонстраций.

    Внутреннее хранение:
        entries    — сырые обучающие записи
        _clf       — обученный DecisionTreeClassifier
        _shapes    — список известных shape (порядок для one-hot)
        _colors    — список известных color (порядок для one-hot)
        _labels    — список известных меток action_sequence
    """

    def __init__(self):
        self.entries: list[PolicyEntry] = []
        self._clf    = None
        self._shapes: list[str] = []
        self._colors: list[str] = []
        self._labels: list[str] = []

    # -----------------------------------------------------------------------
    # Добавление данных
    # -----------------------------------------------------------------------

    def add(self, shape: str, color: str,
            action_sequence: list[str], demo_id: str) -> None:
        """
        Добавляет одну демонстрацию.
        Если (shape, color) уже есть и action_sequence совпадает — подтверждаем.
        Если конфликт — логируем WARNING.
        """
        acts = actions_to_str(action_sequence)
        for entry in self.entries:
            if entry.shape == shape and entry.color == color:
                if actions_equal(entry.action_sequence, acts):
                    if demo_id not in entry.source_demo_ids:
                        entry.source_demo_ids.append(demo_id)
                    logger.debug(f"  policy: ({shape},{color}) подтверждено {demo_id}")
                else:
                    logger.warning(
                        f"  policy КОНФЛИКТ: ({shape},{color}) "
                        f"уже {entry.action_sequence}, демо {demo_id} даёт {acts}"
                    )
                return

        self.entries.append(PolicyEntry(
            shape=shape, color=color,
            action_sequence=acts,
            source_demo_ids=[demo_id],
        ))
        logger.debug(f"  policy: новая запись ({shape},{color}) → {acts}")

    def fit_from_records(self, records) -> None:
        """Заполняет policy из списка DemoRecord."""
        for rec in records:
            if rec is None or rec.alpha is None:
                continue
            self.add(
                shape=rec.alpha.shape,
                color=rec.alpha.color,
                action_sequence=actions_to_str(rec.action_sequence),
                demo_id=rec.demo_id,
            )
        self._fit_tree()
        logger.info(
            f"Policy обучена: {len(self.entries)} уникальных alpha, "
            f"{sum(len(e.source_demo_ids) for e in self.entries)} демо"
        )

    # -----------------------------------------------------------------------
    # Обучение дерева
    # -----------------------------------------------------------------------

    def _fit_tree(self) -> None:
        """Строит и обучает DecisionTreeClassifier на текущих entries."""
        from sklearn.tree import DecisionTreeClassifier

        # Словари признаков
        self._shapes = sorted({e.shape for e in self.entries})
        self._colors = sorted({e.color for e in self.entries})
        self._labels = sorted({encode_label(e.action_sequence) for e in self.entries})

        X, y = [], []
        for entry in self.entries:
            X.append(self._encode_features(entry.shape, entry.color))
            y.append(encode_label(entry.action_sequence))

        self._clf = DecisionTreeClassifier(
            criterion="entropy",    # информационный выигрыш
            max_depth=None,         # полное дерево — при 7 примерах нет смысла обрезать
            random_state=42,
        )
        self._clf.fit(np.array(X), y)
        logger.debug(f"  DT: обучен, depth={self._clf.get_depth()}, "
                     f"leaves={self._clf.get_n_leaves()}")

    def _encode_features(self, shape: str, color: str) -> list[float]:
        """One-hot encoding (shape, color) → вектор признаков."""
        shape_vec = [1.0 if s == shape else 0.0 for s in self._shapes]
        color_vec = [1.0 if c == color else 0.0 for c in self._colors]
        return shape_vec + color_vec

    def _is_known(self, shape: str, color: str) -> bool:
        """Точное совпадение есть в обучающих данных."""
        return any(e.shape == shape and e.color == color for e in self.entries)

    # -----------------------------------------------------------------------
    # Инференс
    # -----------------------------------------------------------------------

    def predict(self, shape: str, color: str) -> InferenceResult:
        """
        Предсказывает action_sequence для объекта.

        Если объект известен — используем DT напрямую.
        Если нет — иерархия обобщения с WARNING на каждом уровне.
        """
        if self._clf is None:
            raise RuntimeError("Policy не обучена. Вызовите fit_from_records().")

        # 1. Точное совпадение — DT предсказывает
        if self._is_known(shape, color):
            feat  = self._encode_features(shape, color)
            label = self._clf.predict([feat])[0]
            acts  = decode_label(label)
            logger.debug(f"  predict: ({shape},{color}) → exact: {acts}")
            return InferenceResult(
                action_sequence=acts,
                confidence="exact",
                warning=None,
            )

        # Объект неизвестен — пробуем обобщить
        logger.debug(f"  predict: ({shape},{color}) не в обучающих данных")

        # 2. Обобщение по color
        by_color = [e for e in self.entries if e.color == color]
        if by_color:
            unique_acts = {encode_label(e.action_sequence) for e in by_color}
            if len(unique_acts) == 1:
                acts = decode_label(next(iter(unique_acts)))
                msg  = (
                    f"Объект ({shape},{color}) не в обучающих данных. "
                    f"Обобщение по color='{color}': "
                    f"все {len(by_color)} записей → {acts}"
                )
                logger.warning(f"  predict WARNING: {msg}")
                return InferenceResult(
                    action_sequence=acts,
                    confidence="color",
                    warning=msg,
                )
            logger.debug(
                f"  predict: color='{color}' конфликт {unique_acts} → переходим к shape"
            )

        # 3. Обобщение по shape — используем DT с маскированием color
        by_shape = [e for e in self.entries if e.shape == shape]
        if by_shape:
            unique_acts = {encode_label(e.action_sequence) for e in by_shape}
            if len(unique_acts) == 1:
                acts = decode_label(next(iter(unique_acts)))
                msg  = (
                    f"Объект ({shape},{color}) не в обучающих данных. "
                    f"Обобщение по shape='{shape}': "
                    f"все {len(by_shape)} записей → {acts}"
                )
                logger.warning(f"  predict WARNING: {msg}")
                return InferenceResult(
                    action_sequence=acts,
                    confidence="shape",
                    warning=msg,
                )

            # Конфликт по shape — пробуем DT с неизвестным color (нули)
            # Это позволяет дереву использовать только shape-признаки
            feat_no_color = (
                [1.0 if s == shape else 0.0 for s in self._shapes]
                + [0.0] * len(self._colors)
            )
            label = self._clf.predict([feat_no_color])[0]
            acts  = decode_label(label)
            conflict = {e.color: e.action_sequence for e in by_shape}
            msg = (
                f"Объект ({shape},{color}) не в обучающих данных. "
                f"Конфликт по shape='{shape}': {conflict}. "
                f"DT с обнулённым color предсказывает: {acts}"
            )
            logger.warning(f"  predict WARNING: {msg}")
            return InferenceResult(
                action_sequence=acts,
                confidence="shape",
                warning=msg,
            )

        # 4. Fallback — совсем неизвестный объект
        msg = (
            f"Объект ({shape},{color}) полностью неизвестен — "
            f"нет данных ни по shape='{shape}', ни по color='{color}'. "
            f"Отправляем FORWARD."
        )
        logger.warning(f"  predict FALLBACK: {msg}")
        return InferenceResult(
            action_sequence=["FORWARD"],
            confidence="fallback",
            warning=msg,
        )

    # -----------------------------------------------------------------------
    # Валидация
    # -----------------------------------------------------------------------

    def leave_one_out_validate(self, records) -> dict[str, Any]:
        """Leave-one-out cross-validation на обучающих демо."""
        valid = [r for r in records if r is not None and r.alpha is not None]
        results = []
        correct = 0

        for i, test_rec in enumerate(valid):
            trainer_loo = PolicyTrainer()
            for j, rec in enumerate(valid):
                if j != i:
                    trainer_loo.add(
                        shape=rec.alpha.shape,
                        color=rec.alpha.color,
                        action_sequence=actions_to_str(rec.action_sequence),
                        demo_id=rec.demo_id,
                    )
            trainer_loo._fit_tree()

            result     = trainer_loo.predict(test_rec.alpha.shape, test_rec.alpha.color)
            true_acts  = actions_to_str(test_rec.action_sequence)
            pred_acts  = result.action_sequence
            is_correct = actions_equal(true_acts, pred_acts)
            if is_correct:
                correct += 1

            results.append({
                "demo_id":    test_rec.demo_id,
                "alpha":      f"({test_rec.alpha.shape},{test_rec.alpha.color})",
                "true":       true_acts,
                "predicted":  pred_acts,
                "confidence": result.confidence,
                "correct":    is_correct,
                "warning":    result.warning,
            })

        accuracy = correct / len(valid) if valid else 0.0
        return {
            "accuracy": accuracy,
            "correct":  correct,
            "total":    len(valid),
            "details":  results,
        }

    # -----------------------------------------------------------------------
    # Печать дерева решений
    # -----------------------------------------------------------------------

    def print_tree(self) -> None:
        """Печатает дерево решений в человекочитаемом виде."""
        if self._clf is None:
            print("Дерево не обучено.")
            return

        from sklearn.tree import export_text
        feature_names = (
            [f"shape={s}" for s in self._shapes]
            + [f"color={c}" for c in self._colors]
        )
        print("\nDecision Tree:")
        print(export_text(self._clf, feature_names=feature_names))

    def print_table(self) -> None:
        """Печатает таблицу обучающих данных."""
        print("\nPolicy table:")
        print(f"  {'shape':<10} {'color':<12} {'action_sequence':<30} {'sources'}")
        print("  " + "-" * 75)
        for e in self.entries:
            acts    = " → ".join(e.action_sequence)
            sources = ", ".join(e.source_demo_ids)
            print(f"  {e.shape:<10} {e.color:<12} {acts:<30} [{sources}]")

    # -----------------------------------------------------------------------
    # Сохранение / загрузка
    # -----------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "entries": self.entries,
            "clf":     self._clf,
            "shapes":  self._shapes,
            "colors":  self._colors,
            "labels":  self._labels,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        logger.info(f"Policy сохранена: {path} ({len(self.entries)} записей)")

    @classmethod
    def load(cls, path: str | Path) -> "PolicyTrainer":
        path = Path(path)
        trainer = cls()
        with open(path, "rb") as f:
            payload = pickle.load(f)
        trainer.entries  = payload["entries"]
        trainer._clf     = payload["clf"]
        trainer._shapes  = payload["shapes"]
        trainer._colors  = payload["colors"]
        trainer._labels  = payload["labels"]
        logger.info(f"Policy загружена: {path} ({len(trainer.entries)} записей)")
        return trainer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys
    import yaml

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    ap = argparse.ArgumentParser(description="Обучение и валидация policy (Decision Tree)")
    ap.add_argument("--config",   default="config/demos.yaml")
    ap.add_argument("--output",   default="policy/policy.pkl")
    ap.add_argument("--validate", action="store_true",
                    help="запустить leave-one-out валидацию")
    ap.add_argument("--tree",     action="store_true",
                    help="распечатать дерево решений")
    ap.add_argument("--predict",  nargs=2, metavar=("SHAPE", "COLOR"),
                    help="предсказать action: --predict cube yellow")
    ap.add_argument("--debug",    action="store_true")
    args = ap.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    sys.path.insert(0, ".")
    from demo_parser import DemoParser
    from perception.detector import ObjectDetector

    with open("config/perception.yaml") as f:
        pcfg = yaml.safe_load(f)
    m = pcfg["model"]
    detector = ObjectDetector(
        weights_path=m["weights"],
        conf_thresh=m["conf_thresh"],
        iou_thresh=m["iou_thresh"],
        device=m["device"],
    )

    parser  = DemoParser(args.config, detector=detector)
    records = parser.parse_all()

    trainer = PolicyTrainer()
    trainer.fit_from_records(records)
    trainer.print_table()

    if args.tree:
        trainer.print_tree()

    if args.validate:
        print("\nLeave-one-out validation:")
        val = trainer.leave_one_out_validate(records)
        print(f"  Accuracy: {val['correct']}/{val['total']} = {val['accuracy']:.1%}\n")
        print(f"  {'Demo':<25} {'Alpha':<20} {'True':<25} {'Predicted':<25} {'OK'}")
        print("  " + "-" * 105)
        for d in val["details"]:
            ok     = "✓" if d["correct"] else "✗"
            true_s = " → ".join(d["true"])
            pred_s = " → ".join(d["predicted"])
            print(f"  {d['demo_id']:<25} {d['alpha']:<20} "
                  f"{true_s:<25} {pred_s:<25} {ok}  [{d['confidence']}]")
            if d["warning"]:
                print(f"    ⚠ {d['warning']}")

    trainer.save(args.output)

    if args.predict:
        shape, color = args.predict
        print(f"\nПредсказание для ({shape}, {color}):")
        result = trainer.predict(shape, color)
        print(f"  action_sequence : {' → '.join(result.action_sequence)}")
        print(f"  confidence      : {result.confidence}")
        if result.warning:
            print(f"  ⚠ {result.warning}")