//
// Created by root on 2021/3/16.
//

#ifndef FASTGPU_MESSAGE_H
#define FASTGPU_MESSAGE_H
#define MSG_CUDA_MALLOC 1
namespace mgpu {
    typedef uint message_t;

    typedef struct {
    public:
        message_t type; // message type
    } AbMSG; // abstract message

    typedef struct : public AbMSG {
        size_t size; // content
    } cudaMallocMSG;
}

#endif //FASTGPU_MESSAGE_H
