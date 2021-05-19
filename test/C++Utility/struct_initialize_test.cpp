#include <functional>
#include <iostream>
#include <future>
using namespace std;

int print(int b)
{
    cout << "b is "<<b << endl;
    return 898;
}

int main() {
    std::packaged_task<int(int)> task(print);
    future<int> res = task.get_future();
    std::thread th(std::move(task), 89);
    th.detach();
    cout << res.get()<<endl;
}