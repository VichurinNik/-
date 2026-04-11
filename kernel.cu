#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

using namespace std;


void generateMatrix(const string& filename, int n) {
    ofstream file(filename);
    file << n << endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            file << rand() % 10;
            if (j < n - 1) file << " ";
        }
        file << endl;
    }
    file.close();
}

vector<vector<double>> readMatrix(const string& filename, int& n) {
    ifstream file(filename);
    file >> n;
    vector<vector<double>> matrix(n, vector<double>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            file >> matrix[i][j];
    return matrix;
}


__global__ void matrixMulKernel(const double* A, const double* B, double* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        double sum = 0.0;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}


void writeResult(const string& filename, const vector<double>& matrix, int n) {
    ofstream file(filename);
    file << n << endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            file << fixed << setprecision(0) << matrix[i * n + j];
            if (j < n - 1) file << " ";
        }
        file << endl;
    }
}


int main() {
    srand((unsigned)time(nullptr));


    vector<int> sizes = { 200, 400, 800, 1200, 1600, 2000 };
    vector<int> blockSizes = { 8, 16, 32 };
    const int repeats = 3;      


    cout << "=====================================================\n";
    cout << "  CUDA Matrix Multiplication – Automatic Benchmark\n";
    cout << "=====================================================\n\n";


    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        cerr << "No CUDA device found!\n";
        return 1;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    cout << "GPU: " << prop.name << "\n";
    cout << "Compute capability: " << prop.major << "." << prop.minor << "\n";
    cout << "Total global memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB\n\n";

    vector<vector<double>> results(sizes.size(), vector<double>(blockSizes.size(), 0.0));
    for (size_t i = 0; i < sizes.size(); i++) {
        int n = sizes[i];
        cout << "=== Size: " << n << "x" << n << " ===\n";

        string fileA = "matrix_a_" + to_string(n) + ".txt";
        string fileB = "matrix_b_" + to_string(n) + ".txt";
        generateMatrix(fileA, n);
        generateMatrix(fileB, n);
        cout << "Generated: " << fileA << ", " << fileB << endl;

        int nA, nB;
        auto A = readMatrix(fileA, nA);
        auto B = readMatrix(fileB, nB);
        if (nA != n || nB != n) {
            cerr << "Size mismatch!\n";
            return 1;
        }

        vector<double> h_A(n * n), h_B(n * n), h_C(n * n);
        for (int r = 0; r < n; r++) {
            for (int c = 0; c < n; c++) {
                h_A[r * n + c] = A[r][c];
                h_B[r * n + c] = B[r][c];
            }
        }

        double* d_A, * d_B, * d_C;
        size_t bytes = n * n * sizeof(double);
        cudaMalloc(&d_A, bytes);
        cudaMalloc(&d_B, bytes);
        cudaMalloc(&d_C, bytes);
        cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);


        for (size_t j = 0; j < blockSizes.size(); j++) {
            int bs = blockSizes[j];
            cout << "  Block " << bs << "x" << bs << " : ";

            dim3 threads(bs, bs);
            dim3 blocks((n + bs - 1) / bs, (n + bs - 1) / bs);


            matrixMulKernel << <blocks, threads >> > (d_A, d_B, d_C, n);
            cudaDeviceSynchronize();


            double totalTime = 0.0;
            for (int r = 0; r < repeats; r++) {
                cudaEvent_t start, stop;
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                cudaEventRecord(start);
                matrixMulKernel << <blocks, threads >> > (d_A, d_B, d_C, n);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                float ms = 0;
                cudaEventElapsedTime(&ms, start, stop);
                totalTime += ms / 1000.0;
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
            }
            double avgTime = totalTime / repeats;
            results[i][j] = avgTime;

            long long ops = 2LL * n * n * n;
            double gflops = ops / (avgTime * 1e9);
            cout << fixed << setprecision(4) << avgTime << " sec, " << setprecision(2) << gflops << " GFLOPS\n";

           
            if (j == blockSizes.size() - 1) {
                cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost);
                string outFile = "matrix_c_" + to_string(n) + ".txt";
                writeResult(outFile, h_C, n);
                cout << "    Result saved: " << outFile << "\n";
            }
        }

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cout << "----------------------------------------\n";
    }

   
    cout << "\n==================== FINAL TABLE ====================\n";
    cout << "Size\\Block |";
    for (int bs : blockSizes) cout << " " << setw(8) << bs << "x" << left << setw(2) << bs << " |";
    cout << "\n------------------------------------------------------\n";
    for (size_t i = 0; i < sizes.size(); i++) {
        cout << setw(8) << sizes[i] << " |";
        for (size_t j = 0; j < blockSizes.size(); j++) {
            cout << " " << setw(10) << fixed << setprecision(4) << results[i][j] << " |";
        }
        cout << "\n";
    }
    cout << "======================================================\n";


    ofstream csv("cuda_benchmark_auto.csv");
    csv << "Size,BlockSize,Time_sec,GFLOPS\n";
    for (size_t i = 0; i < sizes.size(); i++) {
        for (size_t j = 0; j < blockSizes.size(); j++) {
            double gflops = (2LL * sizes[i] * sizes[i] * sizes[i]) / (results[i][j] * 1e9);
            csv << sizes[i] << "," << blockSizes[j] << "," << results[i][j] << "," << gflops << "\n";
        }
    }
    csv.close();
    cout << "\nResults also saved to cuda_benchmark_auto.csv\n";

    return 0;
}