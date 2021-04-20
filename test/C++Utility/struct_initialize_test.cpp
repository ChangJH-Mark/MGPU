#include <iostream>
#include <map>
#include <unordered_map>
using namespace std;

int main() {
    const char * res[4] = {"xiaoming", "xiaowang", "xiaoli", "draw"};
    int winner = 3, win_value = -1;
    int n;
    map<int,string> str;
    cin >> n;
    cin >> str[0] >> str[1] >> str[2];
    for(int i = 0;i < 3; i++)
    {
        int max_num = 0;
        unordered_map<char, int> appeared;
        // get every string's most often char
        for(auto iter = str[i].begin(); iter != str[i].end(); iter++)
        {
            if(appeared.count(*iter))
            {
                appeared[*iter] ++;
                max_num = std::max(max_num, appeared[*iter]);
            } else {
                appeared[*iter] = 1;
                max_num = max_num?max_num: 1;
            }
        }
        int final = std::min(static_cast<unsigned long >(max_num + n), str[i].size());
        int extra = (final ==str[i].size()) ? (max_num + n - str[i].size()) % 2 : 0;
        final -= extra;
        if(final > win_value)
        {
            winner = i;
            win_value = final;
        } else if(final == win_value)
        {
            winner = 3;
            break;
        }
    }
    cout << res[winner] << endl;
}