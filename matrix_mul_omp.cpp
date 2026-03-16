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


void multiply_matrices_parallel(double** A, double** B, double** C, int n) {
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

void save_results(const char* filename, int sizes[], double times[], int count) {
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        printf("Ошибка: не удалось открыть файл %s для записи результатов\n", filename);
        return;
    }

    fprintf(file, "Эксперимент по параллельному умножению квадратных матриц\n");
    fprintf(file, "========================================================\n\n");
    fprintf(file, "| Размер матрицы | Время выполнения (сек) |\n");
    fprintf(file, "|----------------|------------------------|\n");

    for (int i = 0; i < count; i++) {
        fprintf(file, "| %d             | %f           |\n", sizes[i], times[i]);
    }

    fprintf(file, "\n\n");
    fprintf(file, "CSV формат для построения графиков:\n");
    fprintf(file, "Размер,Время\n");
    for (int i = 0; i < count; i++) {
        fprintf(file, "%d,%f\n", sizes[i], times[i]);
    }

    fclose(file);
    printf("Результаты сохранены в файл %s\n", filename);
}

int main() {
    setlocale(LC_ALL, "Russian");


    int sizes[] = { 200, 400, 800, 1200, 1600, 2000 };
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    double* execution_times = (double*)malloc(num_sizes * sizeof(double));

    printf("========================================\n");
    printf("Параллельное умножение квадратных матриц\n");
    printf("========================================\n\n");


    int num_threads = omp_get_max_threads();
    printf("Количество потоков: %d\n", num_threads);
    printf("----------------------------------------\n");

  
    srand(time(NULL));


    for (int exp = 0; exp < num_sizes; exp++) {
        int n = sizes[exp];
        printf("\n--- Эксперимент %d: размер матрицы %d x %d ---\n", exp + 1, n, n);

     
        double** A = allocate_matrix(n);
        double** B = allocate_matrix(n);
        double** C = allocate_matrix(n);

  
        printf("Генерация матриц...\n");
        generate_random_matrix(A, n);
        generate_random_matrix(B, n);

   
        char filename_a[50], filename_b[50];
        sprintf(filename_a, "matrix_a_%d.txt", n);
        sprintf(filename_b, "matrix_b_%d.txt", n);
        write_matrix_to_file(filename_a, A, n);
        write_matrix_to_file(filename_b, B, n);
        printf("Матрицы сохранены в файлы %s и %s\n", filename_a, filename_b);

    
        printf("Параллельное умножение...\n");
        double start = omp_get_wtime();
        multiply_matrices_parallel(A, B, C, n);
        double end = omp_get_wtime();

        double time_spent = end - start;
        execution_times[exp] = time_spent;

        printf("Время выполнения: %f секунд\n", time_spent);

    
        char filename_c[50];
        sprintf(filename_c, "matrix_c_%d.txt", n);
        write_matrix_to_file(filename_c, C, n);
        printf("Результат сохранен в файл %s\n", filename_c);

        // Очистка памяти
        free_matrix(A, n);
        free_matrix(B, n);
        free_matrix(C, n);

        printf("----------------------------------------\n");
    }


    printf("\n========================================\n");
    printf("Сохранение итоговых результатов...\n");
    save_results("experiment_results.txt", sizes, execution_times, num_sizes);

  
    printf("\n");
    printf("ИТОГОВАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ:\n");
    printf("+-----------------+------------------------+\n");
    printf("| Размер матрицы  | Время выполнения (сек) |\n");
    printf("+-----------------+------------------------+\n");
    for (int i = 0; i < num_sizes; i++) {
        printf("| %-15d | %-22f |\n", sizes[i], execution_times[i]);
    }
    printf("+-----------------+------------------------+\n");

    printf("\n");
    printf("Данные для построения графиков (CSV):\n");
    printf("Размер,Время\n");
    for (int i = 0; i < num_sizes; i++) {
        printf("%d,%f\n", sizes[i], execution_times[i]);
    }

    free(execution_times);

    printf("\n========================================\n");
    printf("Эксперимент завершен.\n");
    printf("Для построения графиков используйте данные из файла experiment_results.txt\n");

    return 0;
}
