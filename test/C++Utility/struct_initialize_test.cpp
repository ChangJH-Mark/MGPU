#include <random>
#include <iostream>
#include <ctime>

using namespace std;

int main() {
    int seed = 1;
    srand(seed);
    for(int i =0;i<10;i++)
        cout << rand() % 100 << " ";
    cout << endl;
}