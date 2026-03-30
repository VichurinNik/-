#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <locale.h>
#include <omp.h>

double** allocate_matrix(int n) {
    double** matrix = (double**)malloc(n * sizeof(double*));
    if (matrix == NULL) {
        printf("Ошибка: не удалось выделить память для строк матрицы\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < n; i++) {
        matrix[i] = (double*)malloc(n * sizeof(double));
        if (matrix[i] == NULL) {
            printf("Ошибка: не удалось выделить память для строки %d матрицы\n", i);
            exit(EXIT_FAILURE);
        }
    }
    return matrix;
}

void free_matrix(double** matrix, int n) {
    if (matrix == NULL) return;
    for (int i = 0; i < n; i++) {
        if (matrix[i] != NULL) {
            free(matrix[i]);
        }
    }
    free(matrix);
}

void generate_random_matrix(double** matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i][j] = rand() % 10;
        }
    }
}

void write_matrix_to_file(const char* filename, double** matrix, int n) {
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        printf("Ошибка: не удалось открыть файл %s для записи\n", filename);
        return;
    }

    fprintf(file, "%d\n", n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fprintf(file, "%.0f ", matrix[i][j]);
        }
        fprintf(file, "\n");
    }
    fclose(file);
}

void multiply_matrices_parallel(double** A, double** B, double** C, int n, int num_threads) {
    omp_set_num_threads(num_threads);

#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void save_results(const char* filename, int sizes[], double** times, int num_sizes, int num_threads_config[], int num_configs) {
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        printf("Ошибка: не удалось открыть файл %s для записи результатов\n", filename);
        return;
    }

    fprintf(file, "Эксперимент по параллельному умножению квадратных матриц\n");

    fprintf(file, "Размер матрицы |");
    for (int t = 0; t < num_configs; t++) {
        fprintf(file, " %d поток(ов) |", num_threads_config[t]);
    }
    fprintf(file, "\n");
    fprintf(file, "---------------");
    for (int t = 0; t < num_configs; t++) {
        fprintf(file, "---------------");
    }
    fprintf(file, "\n");

    for (int i = 0; i < num_sizes; i++) {
        fprintf(file, "%-15d|", sizes[i]);
        for (int t = 0; t < num_configs; t++) {
            fprintf(file, " %12.4f |", times[t][i]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
    printf("Результаты сохранены в файл %s\n", filename);
}

void save_speedup_results(const char* filename, int sizes[], double** times, int num_sizes, int num_threads_config[], int num_configs) {
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        printf("Ошибка: не удалось открыть файл %s для записи результатов ускорения\n", filename);
        return;
    }

    fprintf(file, "Ускорение (Speedup) относительно 1 потока\n");

    fprintf(file, "Размер матрицы |");
    for (int t = 1; t < num_configs; t++) {
        fprintf(file, " %d поток(ов) |", num_threads_config[t]);
    }
    fprintf(file, "\n");
    fprintf(file, "---------------");
    for (int t = 1; t < num_configs; t++) {
        fprintf(file, "---------------");
    }
    fprintf(file, "\n");

    for (int i = 0; i < num_sizes; i++) {
        fprintf(file, "%-15d|", sizes[i]);
        double base_time = times[0][i];
        for (int t = 1; t < num_configs; t++) {
            double speedup = base_time / times[t][i];
            fprintf(file, " %12.4f |", speedup);
        }
        fprintf(file, "\n");
    }

    fclose(file);
    printf("Результаты ускорения сохранены в файл %s\n", filename);
}

int main() {
    setlocale(LC_ALL, "Russian");

    int sizes[] = { 200, 400, 800, 1200, 1600, 2000 };
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    int num_threads_config[] = { 1, 2, 4, 8 };
    int num_configs = sizeof(num_threads_config) / sizeof(num_threads_config[0]);

    int available_cores = omp_get_num_procs();
    printf("Доступное количество ядер: %d\n", available_cores);

    int max_threads = (available_cores < num_threads_config[num_configs - 1]) ? available_cores : num_threads_config[num_configs - 1];

    printf("Параллельное умножение квадратных матриц\n");

    double** execution_times = (double**)malloc(num_configs * sizeof(double*));
    for (int t = 0; t < num_configs; t++) {
        execution_times[t] = (double*)malloc(num_sizes * sizeof(double));
    }

    srand(time(NULL));

    for (int t = 0; t < num_configs; t++) {
        int num_threads = num_threads_config[t];
        if (num_threads > available_cores) {
            printf("Пропуск конфигурации с %d потоками (доступно только %d ядер)\n", num_threads, available_cores);
            continue;
        }

        printf("Эксперимент с %d потоками\n", num_threads);

        for (int exp = 0; exp < num_sizes; exp++) {
            int n = sizes[exp];
            printf("\n--- Размер матрицы %d x %d ---\n", n, n);

            double** A = allocate_matrix(n);
            double** B = allocate_matrix(n);
            double** C = allocate_matrix(n);

            printf("Генерация матриц...\n");
            generate_random_matrix(A, n);
            generate_random_matrix(B, n);

            if (t == 0 && exp == 0) {
                char filename_a[50], filename_b[50];
                sprintf(filename_a, "matrix_a_%d.txt", n);
                sprintf(filename_b, "matrix_b_%d.txt", n);
                write_matrix_to_file(filename_a, A, n);
                write_matrix_to_file(filename_b, B, n);
                printf("Матрицы сохранены в файлы %s и %s\n", filename_a, filename_b);
            }

            printf("Параллельное умножение с %d потоками...\n", num_threads);

            if (n <= 400) {
                multiply_matrices_parallel(A, B, C, n, num_threads);
            }

            double start = omp_get_wtime();
            multiply_matrices_parallel(A, B, C, n, num_threads);
            double end = omp_get_wtime();

            double time_spent = end - start;
            execution_times[t][exp] = time_spent;

            printf("Время выполнения: %.4f секунд\n", time_spent);

            if (exp == num_sizes - 1) {
                char filename_c[50];
                sprintf(filename_c, "matrix_c_%d_%d_threads.txt", n, num_threads);
                write_matrix_to_file(filename_c, C, n);
                printf("Результат сохранен в файл %s\n", filename_c);
            }

            free_matrix(A, n);
            free_matrix(B, n);
            free_matrix(C, n);
        }
    }

    printf("Сохранение итоговых результатов...\n");
    save_results("experiment_results.txt", sizes, execution_times, num_sizes, num_threads_config, num_configs);
    save_speedup_results("speedup_results.txt", sizes, execution_times, num_sizes, num_threads_config, num_configs);

    printf("\n");
    printf("ИТОГОВАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ (время в секундах):\n");
    printf("+-----------------+");
    for (int t = 0; t < num_configs; t++) {
        printf("--------------+");
    }
    printf("\n");
    printf("| Размер матрицы |");
    for (int t = 0; t < num_configs; t++) {
        printf(" %d поток(ов) |", num_threads_config[t]);
    }
    printf("\n");
    printf("+-----------------+");
    for (int t = 0; t < num_configs; t++) {
        printf("--------------+");
    }
    printf("\n");

    for (int i = 0; i < num_sizes; i++) {
        printf("| %-15d |", sizes[i]);
        for (int t = 0; t < num_configs; t++) {
            if (execution_times[t][i] > 0) {
                printf(" %12.4f |", execution_times[t][i]);
            }
            else {
                printf(" %12s |", "N/A");
            }
        }
        printf("\n");
    }
    printf("+-----------------+");
    for (int t = 0; t < num_configs; t++) {
        printf("--------------+");
    }
    printf("\n");

    printf("\n");
    printf("ИТОГОВАЯ ТАБЛИЦА УСКОРЕНИЯ (Speedup):\n");
    printf("+-----------------+");
    for (int t = 1; t < num_configs; t++) {
        printf("--------------+");
    }
    printf("\n");
    printf("| Размер матрицы |");
    for (int t = 1; t < num_configs; t++) {
        printf(" %d поток(ов) |", num_threads_config[t]);
    }
    printf("\n");
    printf("+-----------------+");
    for (int t = 1; t < num_configs; t++) {
        printf("--------------+");
    }
    printf("\n");

    for (int i = 0; i < num_sizes; i++) {
        printf("| %-15d |", sizes[i]);
        double base_time = execution_times[0][i];
        for (int t = 1; t < num_configs; t++) {
            if (execution_times[t][i] > 0 && base_time > 0) {
                double speedup = base_time / execution_times[t][i];
                printf(" %12.4f |", speedup);
            }
            else {
                printf(" %12s |", "N/A");
            }
        }
        printf("\n");
    }
    printf("+-----------------+");
    for (int t = 1; t < num_configs; t++) {
        printf("--------------+");
    }
    printf("\n");

    for (int t = 0; t < num_configs; t++) {
        free(execution_times[t]);
    }
    free(execution_times);

    return 0;
}