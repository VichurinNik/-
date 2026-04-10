#include <iostream>
#include <ctime>
#include <fstream>
#include <locale>
#include <chrono>
#include <vector>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <mpi.h>

int MATRIX_DIMENSION = 200;

struct PerformanceMetrics
{
    long long mult_ops = 0;
    long long add_ops = 0;
    long long assign_ops = 0;

    long long total_ops() const
    {
        return mult_ops + add_ops + assign_ops;
    }
};

void init_matrix_random(std::vector<int>& matrix, int dim)
{
    for (int i = 0; i < dim; ++i)
        for (int j = 0; j < dim; ++j)
            matrix[i * dim + j] = rand() % 10;
}

void parallel_matrix_mult(const std::vector<int>& A,
    const std::vector<int>& B,
    std::vector<int>& local_C,
    MPI_Comm communicator,
    int dim,
    PerformanceMetrics& metrics)
{
    int proc_rank, proc_count;
    MPI_Comm_rank(communicator, &proc_rank);
    MPI_Comm_size(communicator, &proc_count);

    metrics = PerformanceMetrics();
    metrics.mult_ops = (long long)dim * dim * dim;
    metrics.add_ops = metrics.mult_ops;
    metrics.assign_ops = (long long)dim * dim * 2;

    int base_rows = dim / proc_count;
    int remainder = dim % proc_count;
    int start_row = proc_rank * base_rows + std::min(proc_rank, remainder);
    int local_rows = base_rows + (proc_rank < remainder ? 1 : 0);

    std::vector<int> local_data(local_rows * dim, 0);

    for (int i = 0; i < local_rows; ++i)
    {
        int global_i = start_row + i;
        for (int j = 0; j < dim; ++j)
        {
            int accumulator = 0;
            for (int k = 0; k < dim; ++k)
                accumulator += A[global_i * dim + k] * B[k * dim + j];
            local_data[i * dim + j] = accumulator;
        }
    }

    local_C = std::move(local_data);
}

void compute_stats(const std::vector<double>& time_data,
    double& avg, double& deviation,
    double& min_val, double& max_val)
{
    if (time_data.empty()) return;

    double total = 0;
    for (double t : time_data) total += t;
    avg = total / time_data.size();

    min_val = *std::min_element(time_data.begin(), time_data.end());
    max_val = *std::max_element(time_data.begin(), time_data.end());

    double sq_diff_sum = 0;
    for (double t : time_data)
        sq_diff_sum += (t - avg) * (t - avg);
    deviation = std::sqrt(sq_diff_sum / time_data.size());
}

void save_results_to_csv(int matrix_dim, int proc_num,
    const std::vector<double>& time_measures,
    const PerformanceMetrics& metrics,
    int exp_id)
{
    int global_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    if (global_rank != 0) return;

    static bool header_written = false;
    std::ofstream out_file("experiment_results.csv", std::ios::app);

    if (!header_written)
    {
        out_file << "ExpID,Dimension,Processes,AvgTime(sec),MinTime(sec),MaxTime(sec),"
            << "StdDev,MultOps,AddOps,AssignOps,TotalOps\n";
        header_written = false;
    }

    double avg_time, std_deviation, min_time, max_time;
    compute_stats(time_measures, avg_time, std_deviation, min_time, max_time);

    out_file << exp_id << ","
        << matrix_dim << ","
        << proc_num << ","
        << std::fixed << std::setprecision(6) << avg_time << ","
        << std::setprecision(6) << min_time << ","
        << std::setprecision(6) << max_time << ","
        << std::setprecision(6) << std_deviation << ","
        << metrics.mult_ops << ","
        << metrics.add_ops << ","
        << metrics.assign_ops << ","
        << metrics.total_ops() << "\n";
    out_file.close();
}

void generate_final_report(const std::vector<int>& dimensions,
    const std::vector<int>& proc_configs,
    const std::vector<std::vector<std::vector<double>>>& timing_data,
    int current_proc_count)
{
    int global_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    if (global_rank != 0) return;

    std::ofstream report("execution_report.txt", std::ios::app);
    if (!report.is_open()) return;

    report << "=========================================\n";
    report << "FINAL REPORT: MPI Matrix Multiplication\n";
    report << "=========================================\n\n";

    report << "SYSTEM SPECIFICATIONS:\n";
    report << "---------------------\n";
    report << "Processes used in this run: " << current_proc_count << "\n";
    report << "Measurements per configuration: 3\n\n";

    report << "PERFORMANCE RESULTS (for " << current_proc_count << " processes):\n";
    report << "----------------------------------------\n\n";

    report << "AVERAGE EXECUTION TIME (seconds):\n";
    report << "--------------------------------\n";
    report << "Size\tTime (mean ± std dev)\n";
    report << "----\t--------------------\n";

    for (size_t i = 0; i < dimensions.size(); ++i)
    {
        double sum = 0;
        for (double t : timing_data[i][0]) sum += t;
        double mean_val = sum / timing_data[i][0].size();

        double sq_sum = 0;
        for (double t : timing_data[i][0]) sq_sum += (t - mean_val) * (t - mean_val);
        double std_dev = std::sqrt(sq_sum / timing_data[i][0].size());

        report << dimensions[i] << "\t" << std::fixed << std::setprecision(4) << mean_val
            << " ± " << std::setprecision(4) << std_dev << "\n";
    }

    report << "\n\nSPEEDUP & EFFICIENCY:\n";
    report << "-------------------\n";
    report << "Speedup (S = T₁/Tₚ) and Efficiency (E = S/p × 100%) can be calculated\n";
    report << "by comparing single-process and multi-process results from separate runs.\n";

    report << "\n=========================================\n";
    report.close();
}

int main(int argc, char** argv)
{
    setlocale(LC_ALL, "ru_RU.UTF-8");

    MPI_Init(&argc, &argv);

    int global_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size != 4)
    {
        if (global_rank == 0)
        {
            std::cerr << "ERROR: This program requires exactly 4 MPI processes!\n";
            std::cerr << "Usage: mpirun -np 4 " << argv[0] << "\n";
        }
        MPI_Finalize();
        return 1;
    }

    srand(static_cast<unsigned>(time(nullptr)));

    std::vector<int> test_dimensions = { 200, 400, 800, 1200, 1600, 2000 };
    const int TRIALS_PER_CONFIG = 3;

    if (global_rank == 0)
    {
        std::cout << "\n=========================================\n";
        std::cout << "MPI MATRIX MULTIPLICATION BENCHMARK\n";
        std::cout << "=========================================\n";
        std::cout << "Testing with 1, 2, and 4 processes sequentially\n";
        std::cout << "NOTE: This may take considerable time!\n";
        std::cout << "=========================================\n\n";
    }

    MPI_Group global_group;
    MPI_Comm_group(MPI_COMM_WORLD, &global_group);

    std::vector<int> process_counts = { 1, 2, 4 };
    int experiment_id = 1;

    for (int proc_count : process_counts)
    {
        MPI_Group sub_group;
        std::vector<int> selected_ranks(proc_count);
        for (int i = 0; i < proc_count; ++i) selected_ranks[i] = i;
        MPI_Group_incl(global_group, proc_count, selected_ranks.data(), &sub_group);

        MPI_Comm sub_communicator;
        MPI_Comm_create(MPI_COMM_WORLD, sub_group, &sub_communicator);

        if (sub_communicator != MPI_COMM_NULL)
        {
            int local_rank, local_size;
            MPI_Comm_rank(sub_communicator, &local_rank);
            MPI_Comm_size(sub_communicator, &local_size);

            std::vector<std::vector<std::vector<double>>> all_timing_data;

            if (local_rank == 0 && global_rank == 0)
            {
                std::cout << "\n>>> Running benchmarks with " << proc_count << " process(es) <<<\n";
                std::cout << std::string(60, '=') << "\n";
            }

            for (int dim : test_dimensions)
            {
                MATRIX_DIMENSION = dim;

                if (local_rank == 0 && global_rank == 0)
                {
                    std::cout << "\n[ Testing matrix size: " << dim << "x" << dim << " ]\n";
                    std::cout << std::string(50, '-') << "\n";
                }

                std::vector<std::vector<double>> dimension_trials;

                for (int trial = 0; trial < TRIALS_PER_CONFIG; ++trial)
                {
                    std::vector<int> matrixA(dim * dim);
                    std::vector<int> matrixB(dim * dim);
                    init_matrix_random(matrixA, dim);
                    init_matrix_random(matrixB, dim);

                    MPI_Barrier(sub_communicator);

                    double elapsed_time = 0.0;

                    if (local_rank == 0 && global_rank == 0)
                    {
                        std::cout << "    Trial " << (trial + 1) << "/" << TRIALS_PER_CONFIG << "... ";
                        std::cout.flush();
                    }

                    auto start = std::chrono::high_resolution_clock::now();

                    std::vector<int> local_result;
                    PerformanceMetrics metrics;
                    parallel_matrix_mult(matrixA, matrixB, local_result,
                        sub_communicator, dim, metrics);

                    MPI_Barrier(sub_communicator);

                    auto end = std::chrono::high_resolution_clock::now();

                    if (local_rank == 0 && global_rank == 0)
                    {
                        elapsed_time = std::chrono::duration<double>(end - start).count();
                        std::cout << std::fixed << std::setprecision(3) << elapsed_time << " sec\n";
                        dimension_trials.push_back({ elapsed_time });
                    }
                }

                if (local_rank == 0 && global_rank == 0)
                {
                    std::vector<double> trial_times;
                    for (auto& record : dimension_trials)
                        trial_times.push_back(record[0]);

                    PerformanceMetrics dummy_metrics;
                    dummy_metrics.mult_ops = (long long)dim * dim * dim;
                    dummy_metrics.add_ops = dummy_metrics.mult_ops;
                    dummy_metrics.assign_ops = (long long)dim * dim * 2;

                    save_results_to_csv(dim, proc_count, trial_times,
                        dummy_metrics, experiment_id++);
                    all_timing_data.push_back({ trial_times });
                }
            }

            if (local_rank == 0 && global_rank == 0)
            {
                generate_final_report(test_dimensions, { proc_count },
                    all_timing_data, proc_count);
            }

            MPI_Comm_free(&sub_communicator);
        }

        MPI_Group_free(&sub_group);
    }

    MPI_Group_free(&global_group);
    MPI_Finalize();

    if (global_rank == 0)
    {
        std::cout << "\n=========================================\n";
        std::cout << "BENCHMARK COMPLETED SUCCESSFULLY!\n";
        std::cout << "=========================================\n";
        std::cout << "Output files generated:\n";
        std::cout << "  • experiment_results.csv - Detailed timing data\n";
        std::cout << "  • execution_report.txt   - Summary report\n";
        std::cout << "=========================================\n\n";
    }

    return 0;
}