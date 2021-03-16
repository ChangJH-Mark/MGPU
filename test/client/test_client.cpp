//
// Created by root on 2021/3/16.
//
#include <iostream>
#include "client/api.h"
using namespace std;

int main() {
    auto ret = mgpu::cudaMalloc(1 << 20);
    cout << "malloc success!" << ret << endl;
    return 0;
}
