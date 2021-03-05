//
// Created by mark on 2021/3/1.
//

#include <iostream>
using namespace std;

int main() {
    cudaDeviceProp deviceProp{};
    int count = 0;
    cudaError error;
    if(cudaGetDeviceCount(&count)!=cudaSuccess || count <= 0) {
        if(count >0)
            cout << cudaGetErrorString(error) << endl;
        else
            cout << "no error but count = 0" << endl;
        exit(EXIT_FAILURE);
    }
    cudaGetDeviceProperties(&deviceProp, 0);
    cout << deviceProp.computePreemptionSupported << endl;
    return 0;
}