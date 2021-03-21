//
// Created by root on 2021/3/16.
//

#ifndef FASTGPU_MESSAGE_H
#define FASTGPU_MESSAGE_H
#define MSG_CUDA_MALLOC 1
namespace mgpu {
    typedef uint msg_t;

    typedef struct {
    public:
        msg_t type; // message type
        uint key; // pid << 16 + stream_t
    } AbMsg; // abstract message

    typedef struct : public AbMsg {
        size_t size; // content
    } cudaMallocMSG;
}

#endif //FASTGPU_MESSAGE_H
