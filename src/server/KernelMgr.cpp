//
// Created by root on 2021/10/31.
//

#include "server/kernel.h"
#include "common/Log.h"
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include <string>
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>

using namespace mgpu;
using namespace rapidjson;
using namespace std;

void KernelMgr::init() {
    string path = "/opt/custom/kernels.json";

    struct stat file{};
    stat(path.c_str(), &file);
    char *json = new char[file.st_size + 1];
    int fd = open(path.c_str(), O_RDONLY);
    read(fd, json, file.st_size);
    close(fd);
    json[file.st_size] = '\0';

    Document d;
    d.Parse(json);
    delete[] json;

    //init kernels
    for (auto it = d.MemberBegin(); it != d.MemberEnd(); ++it) {
        string name = it->name.GetString();
        Value &v = it->value;
        kns[name] = Kernel{.property = v.GetObj()["property"].GetFloat()};
    }
}