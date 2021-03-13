//
// Created by root on 2021/3/13.
//
#include <sys/socket.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <cstdlib>
#include <unistd.h>
#include <cstddef>

char *socket_path = "server.socket";
struct a {
    int a;
    char * b;
};
struct b{
    int a;
    char b[20];
};

int main(){
    int fd, size, listenfd;
    struct sockaddr_un serun, cliun;
    memset(&serun, 0, sizeof(serun));
    serun.sun_family = AF_UNIX;
    strcpy(serun.sun_path, socket_path);
    if((fd = socket(AF_UNIX, SOCK_STREAM, 0)) < 0){
        exit(1);
    }
    size = offsetof(struct sockaddr_un, sun_path) + strlen(serun.sun_path);
    if(bind(fd, (struct sockaddr *) &serun, size) < 0) {
        exit(1);
    }
    if((listenfd = listen(fd, 10) )< 0) {
        exit(1);
    }
    printf("UNIX domain socket bound\n");
    while(1){
        int clifd, len;
        struct sockaddr_un un;
        struct stat statbuf;
        len = sizeof(un);
        if((clifd = accept(listenfd, (struct sockaddr *)&un, reinterpret_cast<socklen_t *>(&len))) < 0){
            exit(1);
        }

        len -= offsetof(struct sockaddr_un, sun_path);
        un.sun_path[len]=0;
        if(stat(un.sun_path, &statbuf) < 0){
            exit(1);
        }
        if(S_ISSOCK(statbuf.st_mode) == 0) {
            exit(1);
        }

        unlink(un.sun_path);
    }
    exit(0);
}