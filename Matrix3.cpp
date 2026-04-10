#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <locale.h>
#include <omp.h>
#include <windows.h>

double** allocate_matrix(int n) {
    double** matrix = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        matrix[i] = (double*)malloc(n * sizeof(double));
    }
    return matrix;
}

void free_matrix(double** matrix, int n) {
    for (int i = 0; i < n; i++) free(matrix[i]);
    free(matrix);
}

void generate_matrix(double** matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i][j] = (double)(rand() % 10);
        }
    }
}

double test_size(int n, int threads) {
    omp_set_num_threads(threads);

    double** A = allocate_matrix(n);
    double** B = allocate_matrix(n);
    double** C = allocate_matrix(n);

    generate_matrix(A, n);
    generate_matrix(B, n);

    double start = omp_get_wtime();

#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0;
            for (int k = 0; k < n; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }

    double end = omp_get_wtime();

    free_matrix(A, n);
    free_matrix(B, n);
    free_matrix(C, n);

    return end - start;
}

void save_results(const char* filename, int sizes[], double results[][4], int num_sizes, int threads[], int num_threads, int max_threads) {
    FILE* f = fopen(filename, "w");
    if (!f) return;

    fprintf(f, "ЭКСПЕРИМЕНТ ПО ПАРАЛЛЕЛЬНОМУ УМНОЖЕНИЮ МАТРИЦ (OpenMP)\n");
    fprintf(f, "========================================================\n\n");

    fprintf(f, "Таблица времени выполнения (в секундах):\n\n");
    fprintf(f, "| Размер матрицы |");
    for (int t = 0; t < num_threads; t++) {
        if (threads[t] <= max_threads) {
            fprintf(f, " %2d пот. |", threads[t]);
        }
    }
    fprintf(f, "\n|----------------|");
    for (int t = 0; t < num_threads; t++) {
        if (threads[t] <= max_threads) {
            fprintf(f, "---------|");
        }
    }
    fprintf(f, "\n");

    for (int i = 0; i < num_sizes; i++) {
        fprintf(f, "| %-14d |", sizes[i]);
        for (int t = 0; t < num_threads; t++) {
            if (threads[t] <= max_threads) {
                fprintf(f, " %7.4f |", results[i][t]);
            }
        }
        fprintf(f, "\n");
    }

    fprintf(f, "\n\nУСКОРЕНИЕ (Speedup):\n");
    fprintf(f, "====================\n\n");
    fprintf(f, "| Размер матрицы |");
    for (int t = 1; t < num_threads; t++) {
        if (threads[t] <= max_threads) {
            fprintf(f, " %2d пот. |", threads[t]);
        }
    }
    fprintf(f, "\n|----------------|");
    for (int t = 1; t < num_threads; t++) {
        if (threads[t] <= max_threads) {
            fprintf(f, "---------|");
        }
    }
    fprintf(f, "\n");

    for (int i = 0; i < num_sizes; i++) {
        fprintf(f, "| %-14d |", sizes[i]);
        for (int t = 1; t < num_threads; t++) {
            if (threads[t] <= max_threads) {
                double speedup = results[i][0] / results[i][t];
                fprintf(f, " %6.2fx |", speedup);
            }
        }
        fprintf(f, "\n");
    }

    fprintf(f, "\n\nЭФФЕКТИВНОСТЬ (%%):\n");
    fprintf(f, "===================\n\n");
    fprintf(f, "| Размер матрицы |");
    for (int t = 1; t < num_threads; t++) {
        if (threads[t] <= max_threads) {
            fprintf(f, " %2d пот. |", threads[t]);
        }
    }
    fprintf(f, "\n|----------------|");
    for (int t = 1; t < num_threads; t++) {
        if (threads[t] <= max_threads) {
            fprintf(f, "---------|");
        }
    }
    fprintf(f, "\n");

    for (int i = 0; i < num_sizes; i++) {
        fprintf(f, "| %-14d |", sizes[i]);
        for (int t = 1; t < num_threads; t++) {
            if (threads[t] <= max_threads) {
                double efficiency = (results[i][0] / results[i][t] / threads[t]) * 100;
                fprintf(f, " %6.1f%% |", efficiency);
            }
        }
        fprintf(f, "\n");
    }

    fprintf(f, "\n\nCSV ФОРМАТ:\n");
    fprintf(f, "===========\n");
    fprintf(f, "Size");
    for (int t = 0; t < num_threads; t++) {
        if (threads[t] <= max_threads) {
            fprintf(f, ",%d_threads", threads[t]);
        }
    }
    fprintf(f, "\n");

    for (int i = 0; i < num_sizes; i++) {
        fprintf(f, "%d", sizes[i]);
        for (int t = 0; t < num_threads; t++) {
            if (threads[t] <= max_threads) {
                fprintf(f, ",%.4f", results[i][t]);
            }
        }
        fprintf(f, "\n");
    }

    fprintf(f, "\n\nУСКОРЕНИЕ CSV:\n");
    fprintf(f, "Size");
    for (int t = 1; t < num_threads; t++) {
        if (threads[t] <= max_threads) {
            fprintf(f, ",%d_threads_speedup", threads[t]);
        }
    }
    fprintf(f, "\n");

    for (int i = 0; i < num_sizes; i++) {
        fprintf(f, "%d", sizes[i]);
        for (int t = 1; t < num_threads; t++) {
            if (threads[t] <= max_threads) {
                fprintf(f, ",%.4f", results[i][0] / results[i][t]);
            }
        }
        fprintf(f, "\n");
    }

    fclose(f);
    printf("\nРезультаты сохранены в %s\n", filename);
}

int main() {
    setlocale(LC_ALL, "Russian");

    int sizes[] = { 200, 400, 800, 1200, 1600, 2000 };
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int threads[] = { 1, 2, 4, 8 };
    int num_threads = sizeof(threads) / sizeof(threads[0]);

    int max_threads = 1;
#ifdef _OPENMP
    max_threads = omp_get_max_threads();
    printf("OpenMP поддерживается. Максимум потоков: %d\n", max_threads);
#else
    printf("ВНИМАНИЕ: OpenMP не поддерживается!\n");
    printf("Включите /openmp в настройках проекта.\n");
    printf("Продолжаем с 1 потоком...\n");
#endif

    printf("\n========================================\n");
    printf("ПАРАЛЛЕЛЬНОЕ УМНОЖЕНИЕ МАТРИЦ (OpenMP)\n");
    printf("========================================\n\n");

    srand((unsigned int)time(NULL));

    double results[6][4] = { 0 };

    for (int t = 0; t < num_threads; t++) {
        if (threads[t] > max_threads) {
            printf("\nПропуск %d потоков (доступно %d)\n", threads[t], max_threads);
            continue;
        }

        printf("\n=== %d поток(ов) ===\n", threads[t]);

        for (int s = 0; s < num_sizes; s++) {
            printf("  %dx%d: ", sizes[s], sizes[s]);
            fflush(stdout);

            results[s][t] = test_size(sizes[s], threads[t]);
            printf("%.4f сек\n", results[s][t]);
        }
    }

    printf("\n\n========================================\n");
    printf("ИТОГОВАЯ ТАБЛИЦА (секунды)\n");
    printf("========================================\n\n");

    printf("Размер   |");
    for (int t = 0; t < num_threads; t++) {
        if (threads[t] <= max_threads) {
            printf(" %2d пот. |", threads[t]);
        }
    }
    printf("\n---------|");
    for (int t = 0; t < num_threads; t++) {
        if (threads[t] <= max_threads) {
            printf("--------|");
        }
    }
    printf("\n");

    for (int s = 0; s < num_sizes; s++) {
        printf("%-8d |", sizes[s]);
        for (int t = 0; t < num_threads; t++) {
            if (threads[t] <= max_threads) {
                printf(" %6.3f |", results[s][t]);
            }
        }
        printf("\n");
    }

    printf("\n\nУСКОРЕНИЕ (относительно 1 потока):\n");
    printf("====================================\n\n");

    printf("Размер   |");
    for (int t = 1; t < num_threads; t++) {
        if (threads[t] <= max_threads) {
            printf(" %2d пот. |", threads[t]);
        }
    }
    printf("\n---------|");
    for (int t = 1; t < num_threads; t++) {
        if (threads[t] <= max_threads) {
            printf("--------|");
        }
    }
    printf("\n");

    for (int s = 0; s < num_sizes; s++) {
        printf("%-8d |", sizes[s]);
        for (int t = 1; t < num_threads; t++) {
            if (threads[t] <= max_threads) {
                printf(" %5.2fx |", results[s][0] / results[s][t]);
            }
        }
        printf("\n");
    }

    printf("\n\nЭФФЕКТИВНОСТЬ (%%):\n");
    printf("==================\n\n");

    printf("Размер   |");
    for (int t = 1; t < num_threads; t++) {
        if (threads[t] <= max_threads) {
            printf(" %2d пот. |", threads[t]);
        }
    }
    printf("\n---------|");
    for (int t = 1; t < num_threads; t++) {
        if (threads[t] <= max_threads) {
            printf("--------|");
        }
    }
    printf("\n");

    for (int s = 0; s < num_sizes; s++) {
        printf("%-8d |", sizes[s]);
        for (int t = 1; t < num_threads; t++) {
            if (threads[t] <= max_threads) {
                double efficiency = (results[s][0] / results[s][t] / threads[t]) * 100;
                printf(" %5.1f%% |", efficiency);
            }
        }
        printf("\n");
    }

    save_results("openmp_results.txt", sizes, results, num_sizes, threads, num_threads, max_threads);

    printf("\nНажмите Enter для выхода...");
    getchar();

    return 0;
}