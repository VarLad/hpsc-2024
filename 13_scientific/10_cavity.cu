#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <fstream>
#include <vector>

using namespace std;

__global__ void bcalc(int nx, int ny, double dx, double dy, float **u,
                      float **v, float **b, double dt, double rho) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
    double tempu = 0.5 * (u[j][i + 1] - u[j][i - 1]);
    double tempv = 0.5 * (v[j + 1][i] - v[j - 1][i]);
    b[j][i] =
        rho *
        (1.0 / dt * (tempu / dx + tempv / dy) - ((tempu / dx) * (tempu / dx)) -
         2 * ((tempu / dy) * (tempv / dx)) - (tempv / dy) * (tempv / dy));
  }
}

__global__ void pcalc(int nx, int ny, double dx, double dy, float **p,
                      float **b, float **pn) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
    p[j][i] = (dy * dy * (pn[j][i + 1] + pn[j][i - 1]) +
               dx * dx * (pn[j + 1][i] + pn[j - 1][i]) -
               b[j][i] * dx * dx * dy * dy) /
              (2 * (dx * dx + dy * dy));
  }
}

__global__ void pbound(int nx, int ny, float **p) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nx) {
    p[0][i] = p[1][i];
    p[ny - 1][i] = 0.0;
  }
  if (j < ny) {
    p[j][0] = p[j][1];
    p[j][nx - 1] = p[j][nx - 2];
  }
}

__global__ void uvcalc(int nx, int ny, double dx, double dy, double dt,
                       float **u, float **v, float **p, float **un, float **vn,
                       double rho, double nu) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
    u[j][i] = un[j][i] - un[j][i] * dt / dx * (un[j][i] - un[j][i - 1]) -
              un[j][i] * dt / dy * (un[j][i] - un[j - 1][i]) -
              dt / (2 * rho * dx) * (p[j][i + 1] - p[j][i - 1]) +
              nu * dt / dx / dx * (un[j][i + 1] - 2 * un[j][i] + un[j][i - 1]) +
              nu * dt / dy / dy * (un[j + 1][i] - 2 * un[j][i] + un[j - 1][i]);
    v[j][i] = vn[j][i] - vn[j][i] * dt / dx * (vn[j][i] - vn[j][i - 1]) -
              vn[j][i] * dt / dy * (vn[j][i] - vn[j - 1][i]) -
              dt / (2 * rho * dx) * (p[j + 1][i] - p[j - 1][i]) +
              nu * dt / dx / dx * (vn[j][i + 1] - 2 * vn[j][i] + vn[j][i - 1]) +
              nu * dt / dy / dy * (vn[j + 1][i] - 2 * vn[j][i] + vn[j - 1][i]);
  }
}

__global__ void uvbound(int nx, int ny, float **u, float **v) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nx) {
    u[0][i] = 0.0;
    u[ny - 1][i] = 1.0;
    v[0][i] = 0.0;
    v[ny - 1][i] = 0.0;
  }
  if (j < ny) {
    u[j][0] = 0.0;
    u[j][nx - 1] = 0.0;
    v[j][0] = 0.0;
    v[j][nx - 1] = 0.0;
  }
}

int main() {
  int nx = 41;
  int ny = 41;
  int nt = 500;
  int nit = 50;
  double dx = 2.0 / (nx - 1);
  double dy = 2.0 / (ny - 1);
  double dt = 0.01;
  double rho = 1.0;
  double nu = 0.02;

  ofstream ufile("u.dat");
  ofstream vfile("v.dat");
  ofstream pfile("p.dat");

  // Matrix initialization
  float **u, **v, **p, **b, **un, **vn, **pn;
  cudaMallocManaged(&u, ny * sizeof(float *));
  cudaMallocManaged(&v, ny * sizeof(float *));
  cudaMallocManaged(&p, ny * sizeof(float *));
  cudaMallocManaged(&b, ny * sizeof(float *));
  cudaMallocManaged(&un, ny * sizeof(float *));
  cudaMallocManaged(&vn, ny * sizeof(float *));
  cudaMallocManaged(&pn, ny * sizeof(float *));
  for (int i = 0; i < ny; i++) {
    cudaMallocManaged(&u[i], nx * sizeof(float *));
    cudaMallocManaged(&v[i], nx * sizeof(float *));
    cudaMallocManaged(&p[i], nx * sizeof(float *));
    cudaMallocManaged(&b[i], nx * sizeof(float *));
    cudaMallocManaged(&un[i], nx * sizeof(float *));
    cudaMallocManaged(&vn[i], nx * sizeof(float *));
    cudaMallocManaged(&pn[i], nx * sizeof(float *));
    for (int j = 0; j < nx; j++) {
      u[i][j] = 0.0;
      v[i][j] = 0.0;
      p[i][j] = 0.0;
      b[i][j] = 0.0;
      un[i][j] = 0.0;
      vn[i][j] = 0.0;
      pn[i][j] = 0.0;
    }
  }

  // Creating block sizes
  dim3 blockSize(16, 16);
  dim3 gridSize((nx + blockSize.x - 1) / blockSize.x,
                (ny + blockSize.y - 1) / blockSize.y);
  for (int n = 0; n < nt; n++) {
    bcalc<<<gridSize, blockSize>>>(nx, ny, dx, dy, u, v, b, dt, rho);
    cudaDeviceSynchronize();
    for (int it = 0; it < nit; it++) {
      for (int j = 0; j < ny; j++)
        for (int i = 0; i < nx; i++)
          pn[j][i] = p[j][i];
      pcalc<<<gridSize, blockSize>>>(nx, ny, dx, dy, p, b, pn);
      cudaDeviceSynchronize();
    }
    pbound<<<gridSize, blockSize>>>(nx, ny, p);
    cudaDeviceSynchronize();
    for (int i = 0; i < ny; i++) {
      for (int j = 0; j < nx; j++) {
        un[i][j] = u[i][j];
        vn[i][j] = v[i][j];
      }
    }
    uvcalc<<<gridSize, blockSize>>>(nx, ny, dx, dy, dt, u, v, p, un, vn, rho,
                                    nu);
    cudaDeviceSynchronize();
    uvbound<<<gridSize, blockSize>>>(nx, ny, u, v);
    cudaDeviceSynchronize();
    if (n % 10 == 0) {
      for (int j = 0; j < ny; j++)
        for (int i = 0; i < nx; i++)
          ufile << u[j][i] << " ";
      ufile << "\n";
      for (int j = 0; j < ny; j++)
        for (int i = 0; i < nx; i++)
          vfile << v[j][i] << " ";
      vfile << "\n";
      for (int j = 0; j < ny; j++)
        for (int i = 0; i < nx; i++)
          pfile << p[j][i] << " ";
      pfile << "\n";
    }
  }

  cudaFree(u);
  cudaFree(v);
  cudaFree(p);
  cudaFree(b);
  cudaFree(un);
  cudaFree(vn);
  cudaFree(pn);

  ufile.close();
  vfile.close();
  pfile.close();
}
