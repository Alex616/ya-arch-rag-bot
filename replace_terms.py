#!/usr/bin/env python3
"""
Скрипт для замены терминов в статьях согласно mapping файлу.
Читает terms_map.json и применяет замены ко всем статьям в указанной директории.
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict


def load_terms_map(map_file: Path) -> Dict[str, str]:
    """
    Загружает маппинг терминов из JSON файла.

    Args:
        map_file: Путь к файлу с маппингом

    Returns:
        Словарь {старый_термин: новый_термин}
    """
    if not map_file.exists():
        print(f"Ошибка: файл {map_file} не найден")
        sys.exit(1)

    with open(map_file, 'r', encoding='utf-8') as f:
        terms_map = json.load(f)

    return terms_map


def replace_terms(text: str, terms_map: Dict[str, str], case_sensitive: bool = True) -> tuple[str, int]:
    """
    Заменяет термины в тексте согласно маппингу.

    Args:
        text: Исходный текст
        terms_map: Словарь замен {старый: новый}
        case_sensitive: Учитывать регистр при замене

    Returns:
        Кортеж (обновлённый_текст, количество_замен)
    """
    total_replacements = 0
    result_text = text

    terms_to_process = list(terms_map.items())

    # Сортируем термины по длине (от длинных к коротким)
    # чтобы избежать замены части слова
    sorted_terms = sorted(terms_to_process, key=lambda x: len(x[0]), reverse=True)

    for old_term, new_term in sorted_terms:
        # Используем word boundary для точной замены целых слов
        # Экранируем спецсимволы regex
        escaped_term = re.escape(old_term)

        if case_sensitive:
            pattern = rf'\b{escaped_term}\b'
            flags = 0
        else:
            pattern = rf'\b{escaped_term}\b'
            flags = re.IGNORECASE

        # Подсчитываем количество замен
        count = len(re.findall(pattern, result_text, flags=flags))
        total_replacements += count

        # Выполняем замену
        result_text = re.sub(pattern, new_term, result_text, flags=flags)

    return result_text, total_replacements


def process_file(file_path: Path, terms_map: Dict[str, str],
                 output_dir: Path | None = None, inplace: bool = False,
                 case_sensitive: bool = True) -> tuple[int, str]:
    """
    Обрабатывает файл, заменяя термины.

    Args:
        file_path: Путь к файлу
        terms_map: Словарь замен
        output_dir: Директория для сохранения (если None и not inplace, добавляется суффикс)
        inplace: Заменить содержимое исходного файла
        case_sensitive: Учитывать регистр

    Returns:
        Кортеж (количество_замен, путь_выходного_файла)
    """
    # Читаем файл
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Заменяем термины
    updated_text, replacements = replace_terms(text, terms_map, case_sensitive)

    # Определяем путь выходного файла
    if inplace:
        output_path = file_path
    elif output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / file_path.name
    else:
        output_path = file_path.parent / f"{file_path.stem}_replaced{file_path.suffix}"

    # Сохраняем результат
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(updated_text)

    return replacements, str(output_path)


def process_directory(directory: Path, terms_map: Dict[str, str],
                      output_dir: Path | None = None, inplace: bool = False,
                      case_sensitive: bool = True, pattern: str = "*.txt") -> None:
    """
    Обрабатывает все файлы в директории.

    Args:
        directory: Директория с файлами
        terms_map: Словарь замен
        output_dir: Директория для результатов
        inplace: Заменять исходные файлы
        case_sensitive: Учитывать регистр
        pattern: Паттерн для поиска файлов (по умолчанию *.txt)
    """
    if not directory.exists():
        print(f"Ошибка: директория {directory} не найдена")
        sys.exit(1)

    # Находим все файлы по паттерну
    files = list(directory.glob(pattern))

    if not files:
        print(f"В директории {directory} не найдено файлов по паттерну {pattern}")
        return

    print(f"Найдено файлов для обработки: {len(files)}")
    print(f"Терминов для замены: {len(terms_map)}")
    print("Режим: ЗАМЕНА\n")

    # Обрабатываем каждый файл
    total_replacements = 0
    processed = 0

    for file_path in files:
        try:
            replacements, output_path = process_file(
                file_path, terms_map, output_dir, inplace, case_sensitive
            )
            total_replacements += replacements
            processed += 1

            if replacements > 0:
                print(f"✓ {file_path.name}: {replacements} замен → {output_path}")
            else:
                print(f"- {file_path.name}: замен не найдено")

        except Exception as e:
            print(f"✗ {file_path.name}: ошибка - {e}")

    print(f"\nВсего обработано файлов: {processed}/{len(files)}")
    print(f"Всего выполнено замен: {total_replacements}")


def main():
    """Основная функция скрипта."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Замена терминов в текстовых файлах согласно маппингу',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  # Обработать все .txt файлы в директории
  python replace_terms.py --dir knowledge_base/

  # Обработать с заменой исходных файлов
  python replace_terms.py --dir knowledge_base/ --inplace

  # Обработать один файл
  python replace_terms.py --file input.txt

  # Использовать другой маппинг файл
  python replace_terms.py --dir knowledge_base/ --map custom_map.json

  # Без учёта регистра
  python replace_terms.py --dir knowledge_base/ --ignore-case
        """
    )

    parser.add_argument(
        '--map',
        default='terms_map.json',
        help='Путь к JSON файлу с маппингом терминов (по умолчанию: terms_map.json)'
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dir', help='Директория с файлами для обработки')
    group.add_argument('--file', help='Один файл для обработки')

    parser.add_argument(
        '--output',
        help='Директория для сохранения результатов (по умолчанию: добавляется суффикс _replaced)'
    )
    parser.add_argument(
        '--inplace',
        action='store_true',
        help='Заменить содержимое исходных файлов (ОСТОРОЖНО!)'
    )
    parser.add_argument(
        '--pattern',
        default='*.txt',
        help='Паттерн для поиска файлов (по умолчанию: *.txt)'
    )
    parser.add_argument(
        '--ignore-case',
        action='store_true',
        help='Не учитывать регистр при замене'
    )

    args = parser.parse_args()

    # Загружаем маппинг
    terms_map = load_terms_map(Path(args.map))

    if not terms_map:
        print("Предупреждение: маппинг терминов пуст")
        return

    output_dir = Path(args.output) if args.output else None
    case_sensitive = not args.ignore_case

    # Обрабатываем
    if args.dir:
        process_directory(
            Path(args.dir),
            terms_map,
            output_dir,
            args.inplace,
            case_sensitive,
            args.pattern
        )
    else:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Ошибка: файл {file_path} не найден")
            sys.exit(1)

        replacements, output_path = process_file(
            file_path,
            terms_map,
            output_dir,
            args.inplace,
            case_sensitive
        )
        print(f"Обработан файл: {file_path}")
        print(f"Результат сохранён: {output_path}")
        print(f"Выполнено замен: {replacements}")


if __name__ == '__main__':
    main()
