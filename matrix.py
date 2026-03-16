import numpy as np
import os


def read_matrix_from_file(filename):
    try:
        with open(filename, 'r') as f:
            n = int(f.readline().strip())
            matrix = []
            for i in range(n):
                row = list(map(float, f.readline().strip().split()))
                if len(row) != n:
                    print(f"Ошибка: в строке {i + 1} файла {filename} не {n} элементов.")
                    return None
                matrix.append(row)
            return np.array(matrix)
    except FileNotFoundError:
        print(f"Ошибка: файл {filename} не найден.")
        return None
    except Exception as e:
        print(f"Ошибка при чтении файла {filename}: {e}")
        return None


def verify_single_size(n):
    print(f"\n--- Проверка для размера {n}x{n} ---")

    file_a = f"matrix_a_{n}.txt"
    file_b = f"matrix_b_{n}.txt"
    file_c = f"matrix_c_{n}.txt"

    if not (os.path.exists(file_a) and os.path.exists(file_b) and os.path.exists(file_c)):
        print(f"Файлы для размера {n} не найдены. Пропускаем...")
        return None

    A = read_matrix_from_file(file_a)
    B = read_matrix_from_file(file_b)
    C_from_c = read_matrix_from_file(file_c)

    if A is None or B is None or C_from_c is None:
        print(f"Ошибка при чтении файлов для размера {n}")
        return None

    if A.shape[0] != n or B.shape[0] != n or C_from_c.shape[0] != n:
        print(f"Ошибка: несоответствие размеров матриц для n={n}")
        return None

    print(f"Вычисление эталонного произведения для {n}x{n}...")
    C_numpy = np.dot(A, B)

    # Сравнение результатов
    print("Сравнение результатов...")
    if np.allclose(C_numpy, C_from_c, atol=1e-6):
        print(f"РАЗМЕР {n}: ВЕРИФИКАЦИЯ УСПЕШНА")

        diff = np.abs(C_numpy - C_from_c)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        return {
            'size': n,
            'success': True,
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'matrix_a': file_a,
            'matrix_b': file_b,
            'matrix_c': file_c
        }
    else:
        print(f"РАЗМЕР {n}: ВЕРИФИКАЦИЯ НЕ УСПЕШНА")

        diff = np.abs(C_numpy - C_from_c)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        max_pos = np.unravel_index(np.argmax(diff), diff.shape)

        print(f"Максимальное расхождение: {max_diff:.6e}")
        print(f"Среднее расхождение: {mean_diff:.6e}")
        print(f"Позиция максимального расхождения: [{max_pos[0]}, {max_pos[1]}]")
        print(f"Значение в C программе: {C_from_c[max_pos]:.6f}")
        print(f"Эталонное значение (NumPy): {C_numpy[max_pos]:.6f}")

        return {
            'size': n,
            'success': False,
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'max_pos': max_pos,
            'c_value': C_from_c[max_pos],
            'numpy_value': C_numpy[max_pos],
            'matrix_a': file_a,
            'matrix_b': file_b,
            'matrix_c': file_c
        }


def main():
    print("=" * 60)
    print("АВТОМАТИЗИРОВАННАЯ ВЕРИФИКАЦИЯ РЕЗУЛЬТАТОВ УМНОЖЕНИЯ МАТРИЦ")
    print("=" * 60)

    sizes = [200, 400, 800, 1200, 1600, 2000]

    results = []
    success_count = 0
    fail_count = 0

    for n in sizes:
        result = verify_single_size(n)
        if result is not None:
            results.append(result)
            if result['success']:
                success_count += 1
            else:
                fail_count += 1

    print("\n" + "=" * 60)
    print("ИТОГОВЫЙ ОТЧЕТ ПО ВЕРИФИКАЦИИ")
    print("=" * 60)

    if results:
        print(f"\nВсего проверено размеров: {len(results)}")
        print(f"Успешно: {success_count}")
        print(f"Ошибки: {fail_count}")

        if success_count == len(results):
            print("\nВСЕ РЕЗУЛЬТАТЫ УСПЕШНО ВЕРИФИЦИРОВАНЫ!")
        else:
            print("\nОБНАРУЖЕНЫ ОШИБКИ В РЕЗУЛЬТАТАХ!")

        print("\n" + "-" * 60)
        print("| Размер | Статус    | Макс. расхождение | Средн. расхождение |")
        print("-" * 60)
        for r in results:
            status = "Успешно" if r['success'] else "Ошибка"
            print(f"| {r['size']:6d} | {status:9s} | {r['max_diff']:.6e} | {r['mean_diff']:.6e} |")
        print("-" * 60)

        # Сохраняем отчет в файл
        with open("verification_report.txt", "w") as f:
            f.write("=" * 60 + "\n")
            f.write("ОТЧЕТ ПО ВЕРИФИКАЦИИ РЕЗУЛЬТАТОВ УМНОЖЕНИЯ МАТРИЦ\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Дата проверки: {np.datetime64('now')}\n\n")

            f.write("Проверенные размеры:\n")
            for r in results:
                status = "УСПЕШНО" if r['success'] else "ОШИБКА"
                f.write(f"  {r['size']}x{r['size']}: {status}\n")
                f.write(f"    Файлы: {r['matrix_a']}, {r['matrix_b']}, {r['matrix_c']}\n")
                f.write(f"    Макс. расхождение: {r['max_diff']:.6e}\n")
                f.write(f"    Средн. расхождение: {r['mean_diff']:.6e}\n")
                if not r['success']:
                    f.write(f"    Позиция ошибки: {r['max_pos']}\n")
                    f.write(f"    Значение C: {r['c_value']:.6f}\n")
                    f.write(f"    Значение NumPy: {r['numpy_value']:.6f}\n")
                f.write("\n")

            if success_count == len(results):
                f.write("\nВСЕ РЕЗУЛЬТАТЫ УСПЕШНО ВЕРИФИЦИРОВАНЫ!\n")
            else:
                f.write("\nОБНАРУЖЕНЫ ОШИБКИ В РЕЗУЛЬТАТАХ!\n")

        print(f"\nПодробный отчет сохранен в файл 'verification_report.txt'")

    else:
        print("\nНЕ НАЙДЕНО ФАЙЛОВ ДЛЯ ВЕРИФИКАЦИИ")
        print("Убедитесь, что в текущей директории есть файлы:")
        for n in sizes:
            print(f"  - matrix_a_{n}.txt, matrix_b_{n}.txt, matrix_c_{n}.txt")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()