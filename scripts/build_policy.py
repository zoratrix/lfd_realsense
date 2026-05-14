"""
build_policy.py — оркестратор LfD pipeline

Запускает полный pipeline: parse → train → save.

Что делает:
    1. Инициализирует ObjectDetector (YOLO)
    2. DemoParser.parse_all() → list[DemoRecord]
    3. PolicyTrainer.fit_from_records() → Decision Tree
    4. Собирает DMP-пути: (shape, color) → [Path, ...]
       Сканирует policy/dmps/, сопоставляет с DemoRecord по demo_id
    5. LOO-валидация (если --validate)
    6. Сохраняет policy.pkl — расширенный payload:
           entries, clf, shapes, colors, labels,   ← стандартный PolicyTrainer
           dmp_map,                                 ← {(shape,color): [str, ...]}
           meta                                     ← timestamp, версии, mAP и т.д.

Итоговый policy.pkl читается как PolicyTrainer.load() (обратная совместимость),
а расширенные поля доступны через policy_runner напрямую через pickle.

Использование:
    python build_policy.py
    python build_policy.py --validate --tree
    python build_policy.py --config config/demos.yaml \\
                           --dmps   policy/dmps       \\
                           --output policy/policy.pkl
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DMP-маппинг
# ---------------------------------------------------------------------------

def build_dmp_map(
    records,
    dmps_dir: Path,
) -> dict[tuple[str, str], list[str]]:
    """
    Строит маппинг (shape, color) → [dmp_path_str, ...].

    Логика поиска для каждого DemoRecord:
        1. Ищем {dmps_dir}/{demo_id}.pkl            — приоритет
        2. Ищем {dmps_dir}/{object_class}.pkl       — fallback по классу
        3. Ищем {dmps_dir}/{shape}_{color}.pkl      — fallback по alpha

    Если на один alpha несколько DMP-файлов (например washer_green из двух демо)
    — все пути кладём в список. policy_runner выберет нужный (random / by index).

    Returns:
        {(shape, color): ["/path/to/a.pkl", "/path/to/b.pkl", ...]}
    """
    dmp_map: dict[tuple[str, str], list[str]] = {}
    missing: list[str] = []

    for rec in records:
        if rec is None or rec.alpha is None:
            continue

        key = (rec.alpha.shape, rec.alpha.color)

        # Кандидаты в порядке приоритета
        candidates = [
            dmps_dir / f"{rec.demo_id}.pkl",
            dmps_dir / f"{rec.object_class}.pkl",
            dmps_dir / f"{rec.alpha.shape}_{rec.alpha.color}.pkl",
        ]

        found = None
        for cand in candidates:
            if cand.exists():
                found = cand
                break

        if found is None:
            # Широкий поиск: любой файл содержащий demo_id как подстроку
            matches = list(dmps_dir.glob(f"*{rec.demo_id}*"))
            if matches:
                found = matches[0]
                logger.debug(f"  DMP: '{rec.demo_id}' нашли по glob → {found.name}")

        if found is None:
            missing.append(rec.demo_id)
            logger.warning(
                f"  DMP не найден для '{rec.demo_id}' "
                f"(alpha={key}, class={rec.object_class})"
            )
            continue

        path_str = str(found.resolve())
        if key not in dmp_map:
            dmp_map[key] = []
        if path_str not in dmp_map[key]:
            dmp_map[key].append(path_str)
            logger.info(f"  DMP: {key} ← {found.name}")
        else:
            logger.debug(f"  DMP: {key} ← {found.name} (уже есть, пропуск)")

    if missing:
        logger.warning(f"Нет DMP для {len(missing)} демо: {missing}")
    logger.info(
        f"DMP-маппинг: {len(dmp_map)} уникальных alpha, "
        f"{sum(len(v) for v in dmp_map.values())} файлов"
    )
    return dmp_map


# ---------------------------------------------------------------------------
# Сохранение расширенного policy.pkl
# ---------------------------------------------------------------------------

def save_extended_policy(
    trainer,
    dmp_map: dict[tuple[str, str], list[str]],
    output_path: Path,
    extra_meta: dict | None = None,
) -> None:
    """
    Сохраняет расширенный payload — надмножество PolicyTrainer.save().

    Формат payload:
        "entries"  : list[PolicyEntry]       — обучающие записи
        "clf"      : DecisionTreeClassifier  — обученное дерево
        "shapes"   : list[str]
        "colors"   : list[str]
        "labels"   : list[str]
        "dmp_map"  : dict[(shape,color) → [str,...]]  — пути к DMP
        "meta"     : dict                             — метаданные сборки

    Обратная совместимость: PolicyTrainer.load() читает только первые 5 ключей
    и игнорирует остальные — ничего не сломается.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "built_at":    datetime.now(timezone.utc).isoformat(),
        "n_demos":     sum(len(e.source_demo_ids) for e in trainer.entries),
        "n_alpha":     len(trainer.entries),
        "n_dmp_files": sum(len(v) for v in dmp_map.values()),
        "tree_depth":  trainer._clf.get_depth() if trainer._clf else None,
        "tree_leaves": trainer._clf.get_n_leaves() if trainer._clf else None,
    }
    if extra_meta:
        meta.update(extra_meta)

    payload = {
        # Стандартные поля PolicyTrainer
        "entries": trainer.entries,
        "clf":     trainer._clf,
        "shapes":  trainer._shapes,
        "colors":  trainer._colors,
        "labels":  trainer._labels,
        # Расширение
        "dmp_map": dmp_map,
        "meta":    meta,
    }

    with open(output_path, "wb") as f:
        pickle.dump(payload, f)

    size_kb = output_path.stat().st_size / 1024
    logger.info(f"policy.pkl сохранён: {output_path} ({size_kb:.1f} KB)")
    logger.info(f"  Метаданные: {meta}")


# ---------------------------------------------------------------------------
# Печать итогового отчёта
# ---------------------------------------------------------------------------

def print_summary(
    records,
    trainer,
    dmp_map: dict[tuple[str, str], list[str]],
    loo_result: dict | None,
) -> None:
    valid   = [r for r in records if r is not None and r.is_valid]
    invalid = [r for r in records if r is not None and not r.is_valid]

    print("\n" + "=" * 65)
    print("  BUILD POLICY — ИТОГИ")
    print("=" * 65)

    print(f"\nДемонстрации: {len(valid)} ОК / {len(invalid)} с ошибками")
    if invalid:
        for r in invalid:
            print(f"  ✗ {r.demo_id}: alpha={r.alpha}, actions={r.action_sequence}")

    print("\nPolicy table:")
    trainer.print_table()

    print("\nDMP-маппинг:")
    for (shape, color), paths in sorted(dmp_map.items()):
        names = [Path(p).name for p in paths]
        print(f"  ({shape:8}, {color:10}) → {names}")

    no_dmp = [
        (e.shape, e.color) for e in trainer.entries
        if (e.shape, e.color) not in dmp_map
    ]
    if no_dmp:
        print(f"\n  ⚠ Без DMP: {no_dmp}")

    if loo_result:
        acc = loo_result["accuracy"]
        print(
            f"\nLeave-one-out: "
            f"{loo_result['correct']}/{loo_result['total']} = {acc:.1%}"
        )
        for d in loo_result["details"]:
            ok     = "✓" if d["correct"] else "✗"
            true_s = " → ".join(d["true"])
            pred_s = " → ".join(d["predicted"])
            print(
                f"  {ok} {d['demo_id']:<25} "
                f"true=[{true_s}]  pred=[{pred_s}]  [{d['confidence']}]"
            )
            if d["warning"] and not d["correct"]:
                print(f"    ⚠ {d['warning']}")

    print("\n" + "=" * 65)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Сборка LfD policy: parse → train → save"
    )
    ap.add_argument("--config",   default="config/demos.yaml",
                    help="путь к demos.yaml")
    ap.add_argument("--dmps",     default="policy/dmps",
                    help="директория с DMP .pkl файлами")
    ap.add_argument("--output",   default="policy/policy.pkl",
                    help="куда сохранить policy.pkl")
    ap.add_argument("--validate", action="store_true",
                    help="запустить LOO-валидацию")
    ap.add_argument("--tree",     action="store_true",
                    help="напечатать структуру дерева решений")
    ap.add_argument("--no-yolo",  action="store_true",
                    help="не загружать детектор (только ArUco, без YOLO)")
    ap.add_argument("--debug",    action="store_true",
                    help="подробный лог (уровень DEBUG)")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    # ------------------------------------------------------------------
    # 1. ObjectDetector
    # ------------------------------------------------------------------
    detector = None
    if not args.no_yolo:
        try:
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
            logger.info("ObjectDetector загружен")
        except Exception as e:
            logger.error(f"Не удалось загрузить детектор: {e}")
            raise SystemExit(1)

    # ------------------------------------------------------------------
    # 2. Парсинг демонстраций
    # ------------------------------------------------------------------
    from scripts.demo_parser import DemoParser
    from scripts.policy_trainer import PolicyTrainer

    logger.info("── Парсинг демонстраций ──────────────────────────────")
    demo_parser = DemoParser(args.config, detector=detector)
    records     = demo_parser.parse_all()

    valid_count = sum(1 for r in records if r is not None and r.is_valid)
    logger.info(f"Распознано {valid_count}/{len(records)} демонстраций")

    if valid_count == 0:
        logger.error("Нет валидных демонстраций — прерываем сборку")
        raise SystemExit(1)

    # ------------------------------------------------------------------
    # 3. Обучение Decision Tree
    # ------------------------------------------------------------------
    logger.info("── Обучение policy ───────────────────────────────────")
    trainer = PolicyTrainer()
    trainer.fit_from_records(records)

    if args.tree:
        trainer.print_tree()

    # ------------------------------------------------------------------
    # 4. DMP-маппинг
    # ------------------------------------------------------------------
    logger.info("── Сборка DMP-маппинга ───────────────────────────────")
    dmps_dir = Path(args.dmps)
    if not dmps_dir.exists():
        logger.warning(f"Директория DMP не найдена: {dmps_dir}")
        dmp_map: dict = {}
    else:
        valid_records = [r for r in records if r is not None and r.is_valid]
        dmp_map = build_dmp_map(valid_records, dmps_dir)

    # ------------------------------------------------------------------
    # 5. LOO-валидация (опционально)
    # ------------------------------------------------------------------
    loo_result = None
    if args.validate:
        logger.info("── LOO-валидация ─────────────────────────────────────")
        loo_result = trainer.leave_one_out_validate(records)
        logger.info(
            f"LOO accuracy: "
            f"{loo_result['correct']}/{loo_result['total']} = "
            f"{loo_result['accuracy']:.1%}"
        )

    # ------------------------------------------------------------------
    # 6. Сохранение
    # ------------------------------------------------------------------
    logger.info("── Сохранение policy.pkl ─────────────────────────────")
    output_path = Path(args.output)
    save_extended_policy(
        trainer=trainer,
        dmp_map=dmp_map,
        output_path=output_path,
        extra_meta={
            "demos_config": str(Path(args.config).resolve()),
            "dmps_dir":     str(dmps_dir.resolve()),
            "loo_accuracy": loo_result["accuracy"] if loo_result else None,
        },
    )

    # ------------------------------------------------------------------
    # 7. Итоговый отчёт
    # ------------------------------------------------------------------
    print_summary(records, trainer, dmp_map, loo_result)
    print(f"\n✓ policy.pkl готов: {output_path.resolve()}")


if __name__ == "__main__":
    main()
