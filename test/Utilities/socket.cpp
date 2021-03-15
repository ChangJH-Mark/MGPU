//
// Created by root on 2021/3/13.
//
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
const char* cli_sock = "/tmp/mgpu/client.sock";
const char* server_sock = "/tmp/mgpu/server.sock";
int main(int argc, char ** argv){
    auto cli = socket(PF_LOCAL, SOCK_STREAM, 0);
    if(cli < 0){
        perror("fail to create socket");
        exit(1);
    }
    struct sockaddr_un cli_address;
    cli_address.sun_family = AF_LOCAL;
    strcpy(cli_address.sun_path, cli_sock);
    if(0 > bind(cli, (struct sockaddr *)&cli_address, SUN_LEN(&cli_address))) {
        perror("fail to bind address");
        exit(1);
    }

    struct sockaddr_un server_address {AF_LOCAL};
    strcpy(server_address.sun_path, server_sock);
    if( 0 > connect(cli, (struct sockaddr*) &server_address, SUN_LEN(&server_address)))
    {
        perror("fail to connect");
        exit(1);
    }
    const char * message = (argc > 1) ? argv[1] : "default message";
    auto sended = send(cli, message, strlen(message), 0);
    std::cout << "send:" << strlen(message) << " bytes" << std::endl;
    close(cli);
    unlink(cli_sock);
    exit(0);
}