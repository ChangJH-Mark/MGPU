#include <iostream>
#include <chrono>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
using namespace std;

int main() {
    chrono::steady_clock::time_point start = chrono::steady_clock::now();
    pid_t first, sec;
    first = fork();
    if(first == 0){
        system("particlefilter_float -x 128 -y 128 -z 10 -np 1000");
        chrono::steady_clock::time_point end = chrono::steady_clock::now();
        cout << " in process a " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms " << endl;
        exit(0);
    } else {
        sec = fork();
        if(sec == 0)
        {
            system("hotspot 512 2 2 data/hotspot/temp_512 data/hotspot/power_512 hotspot.txt");
            chrono::steady_clock::time_point end = chrono::steady_clock::now();
            cout << " in process b " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms "<<endl;
            exit(0);
        }
    }
    waitpid(first, NULL, 0);
    waitpid(sec,NULL,0);
    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    cout << " total time cost is " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms "<< endl;
    exit(0);
}