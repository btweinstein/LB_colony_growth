### Various mako utility code

<%def name='enable_double_support()' filter='trim'>
% if num_type=='double':
#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
    #error "Double precision floating point not supported by OpenCL implementation."
#endif
% endif
</%def>

<%def name='define_node_types()' filter='trim'>
#define FLUID_NODE 0
#define WALL_NODE 1
#define NOT_IN_DOMAIN 2
</%def>