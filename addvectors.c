#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "CL/cl.h"

#define CHECK_ERR(ret) cl_process_error(ret, __FILE__, __LINE__);
enum {STRING_BUFFSIZE = 4096};
enum {BUFSZ = 8 };
const char *source = "\
__kernel void add_vectors(__global const float *A, __global const float *B, __global float *C, const int n) {\
    int id = get_global_id(0);\
    if(id < n) {\
        C[id] = A[id] + B[id];\
    }\
}\
";
void cl_process_error(cl_int ret, const char *file, int line )
{
    const char *cause = "unknow";// *?
    switch (ret)
    {
    case CL_SUCCESS: return;
    case CL_DEVICE_NOT_AVAILABLE: cause = "device for this platform not av";break;
    case CL_DEVICE_NOT_FOUND: cause = "device for this platform not found ";break;
            
    }
    fprintf(stderr, "Error %s at %s:%d code %d\n", cause, file, line, ret);
    abort();
}

struct ocl_ctx_t{
    cl_context ctx;
    cl_command_queue queue;
};

struct platforms_t
{
    cl_uint n;
    cl_platform_id *ids;
};


cl_platform_id select_platform()
{
    cl_uint i;
    cl_int ret;
    struct platforms_t p;
    cl_platform_id selected =0;

    ret = clGetPlatformIDs(0, NULL, &p.n);
    CHECK_ERR(ret);
    assert(p.n>0);
    p.ids = malloc(p.n * sizeof(cl_platform_id));
    assert(p.ids);
    ret = clGetPlatformIDs(p.n, p.ids, NULL);
    CHECK_ERR(ret);
    for (i = 0; i < p.n; i++)
    {
        char buf[STRING_BUFFSIZE];
        cl_platform_id pid = p.ids[i];
        ret = clGetPlatformInfo(pid, CL_PLATFORM_PROFILE, sizeof(buf), buf, NULL); //
        CHECK_ERR(ret);
        int res_of_comp = strcmp(buf, "FULL_PROFILE");
        if (res_of_comp  == 0)
        {
            cl_uint num_of_divice = 0;
            ret = clGetPlatformInfo(pid, CL_PLATFORM_NAME, sizeof(buf), buf, NULL);
            CHECK_ERR(ret);
            ret = clGetDeviceIDs(pid, CL_DEVICE_TYPE_GPU, 0, NULL, &num_of_divice);//
            if (num_of_divice == 0) continue;
            CHECK_ERR(ret);
            printf("selected %s, of gpu div = %d\n", buf, num_of_divice);
            selected = pid;
            break;
        }
    }
    free(p.ids);
    if (selected == 0)
    {
        fprintf(stderr, "OOOOOOOUU, we dont find some device, ((");
        abort();
    }

    return selected;
}

void process_buffer(struct ocl_ctx_t *pct)
{
    float A[BUFSZ], B[BUFSZ];
    int i;
    float C[BUFSZ] = {0};
    size_t global_size =BUFSZ;
    cl_mem bufA, bufB, bufC;
    cl_int ret;
    for (i = 0; i < BUFSZ; i++) 
    {
        A[i] = i*i;
        B[i] = 2+i;
    }

    bufA = clCreateBuffer(pct->ctx, CL_MEM_READ_ONLY, BUFSZ*sizeof(float), NULL, &ret);
    CHECK_ERR(ret);
    bufB = clCreateBuffer(pct->ctx, CL_MEM_READ_ONLY, BUFSZ*sizeof(float), NULL, &ret);
    CHECK_ERR(ret);
    bufC = clCreateBuffer(pct->ctx, CL_MEM_WRITE_ONLY, BUFSZ*sizeof(float), NULL, &ret);
    CHECK_ERR(ret);

    ret = clEnqueueWriteBuffer(pct->queue, bufA, CL_TRUE, 0, BUFSZ*sizeof(float), A, 0, NULL, NULL);
    CHECK_ERR(ret);
    ret = clEnqueueWriteBuffer(pct->queue, bufB, CL_TRUE, 0, BUFSZ*sizeof(float), B, 0, NULL, NULL);
    CHECK_ERR(ret);

    cl_program program = clCreateProgramWithSource(pct->ctx, 1, (const char**)&source, NULL, &ret);
    CHECK_ERR(ret);
    ret = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    CHECK_ERR(ret);
    
    cl_kernel kernel = clCreateKernel(program, "add_vectors", &ret);
    CHECK_ERR(ret);

    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    CHECK_ERR(ret);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    CHECK_ERR(ret);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
    CHECK_ERR(ret);
    ret = clSetKernelArg(kernel, 3, sizeof(int), &global_size);
    CHECK_ERR(ret);

    ret = clEnqueueNDRangeKernel(pct->queue, kernel, 1, NULL, &global_size, NULL, 0, NULL,NULL);
    CHECK_ERR(ret);
    
    ret = clEnqueueReadBuffer(pct->queue, bufC, CL_TRUE, 0, BUFSZ*sizeof(float), C, 0, NULL, NULL);
    CHECK_ERR(ret);

    printf("Res\n");
    for (int i = 0; i < BUFSZ; i++)
    {
        printf("%f + %f = %f \n", A[i], B[i], C[i]);
    }
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);

    
    
}
static void  pfn_notify(const char *errinfo, const void *private_info, size_t cb, void *user_data)
{
    fprintf(stderr, "Context err: %s\n", errinfo);
    abort();
}
int main()
{
    size_t ndevs;
    cl_int ret;
    cl_device_id *devs;
    struct ocl_ctx_t context;
    cl_platform_id select_platform_id;
    select_platform_id = select_platform();
    cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)select_platform_id,0};
    context.ctx = clCreateContextFromType(properties, CL_DEVICE_TYPE_GPU, &pfn_notify, NULL, &ret);
    CHECK_ERR(ret);
    ret = clGetContextInfo(context.ctx, CL_CONTEXT_DEVICES, 0, NULL, &ndevs);
    CHECK_ERR(ret);
    assert(ndevs >0);
    devs = malloc(ndevs*sizeof(cl_device_id));
    ret = clGetContextInfo(context.ctx, CL_CONTEXT_DEVICES, ndevs, devs, NULL);
    CHECK_ERR(ret);
    context.queue = clCreateCommandQueueWithProperties(context.ctx, devs[0], 0, &ret);
    CHECK_ERR(ret);
    process_buffer(&context);
    ret = clFlush(context.queue);
    CHECK_ERR(ret);
    ret = clFinish(context.queue);
    CHECK_ERR(ret);
    ret = clReleaseCommandQueue(context.queue);
    CHECK_ERR(ret);
    ret = clReleaseContext(context.ctx);
    CHECK_ERR(ret);
    free(devs);

}
