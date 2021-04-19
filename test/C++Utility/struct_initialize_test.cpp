//
// Created by root on 2021/4/19.
//

#include <iostream>
using namespace std;

typedef struct Node {
    int a = 1;
    int b = 1;
    int c = 1;
    void print() {
        cout << "a : " << a << " b: " << b << " c: " << c << endl;
    }
} Node;

int main() {
    Node default_;
    default_.print();

    Node init_with_value {4, 5};
    init_with_value.print();
}