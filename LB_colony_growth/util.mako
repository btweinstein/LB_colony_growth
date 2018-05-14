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

<%def name='get_spatial_index(*args)', filter='trim'>

<%
num_pairs = len(args)/2
output = ''
for i in range(num_pairs):
    output += args[i]
    for j in range(0, i):
        output += '*' + args[num_pairs + j]
    if i < num_pairs - 1:
        output +='+'
context.write(output)
%>
</%def>

### Functions responsible for reading to local memory ###

<%def name='define_local_variables()' buffered='True' filter='trim'>
// Local position relative to (0, 0) in workgroup
const int lx = get_local_id(0);
const int ly = get_local_id(1);
% if dimension == 3:
const int lz = get_local_id(2);
% endif

// coordinates of the upper left corner of the buffer in image
// space, including halo
const int buf_corner_x = x - lx - halo;
const int buf_corner_y = y - ly - halo;
% if dimension == 3:
const int buf_corner_z = z - lz - halo;
% endif

// coordinates of our pixel in the local buffer
const int buf_x = lx + halo;
const int buf_y = ly + halo;
% if dimension == 3:
const int buf_z = lz + halo;
% endif

// Index of thread within our work-group
% if dimension == 2:
const int idx_1d = ${get_spatial_index('lx', 'ly', 'get_local_size(0)', 'get_local_size(1)')};
% elif dimension == 3:
const int idx_2d = ${get_spatial_index('lx', 'ly', 'lz', 'get_local_size(0)', 'get_local_size(1)', 'get_local_size(2)')};
% endif

// Spatial location of the thread within the buffer
% if dimension == 2:
const int local_index = ${get_spatial_index('buf_x', 'buf_y', 'buf_nx', 'buf_ny')};
% elif dimension == 3:
const int local_index = ${get_spatial_index('buf_x', 'buf_y', 'buf_z', 'buf_nx', 'buf_ny', 'buf_nz')};
% endif

</%def>

<%def name='read_to_local(var_name, local_mem, default_value)' buffered='True' filter='trim'>
% if dimension==2:
if (idx_1d < buf_nx) {
    for (int row = 0; row < buf_ny; row++) {
        // Read in 1-d slices
        int temp_x = buf_corner_x + idx_1d;
        int temp_y = buf_corner_y + row;

        ${num_type} value = ${default_value};
        % if var_name is not None:
        // If in the domain...
        if((temp_x < nx) && (temp_x > 0) && (temp_y < ny) && (temp_y > 0)){
            int temp_index = ${get_spatial_index('temp_x', 'temp_y', 'nx', 'ny')};
            value = ${var_name}[temp_index];
        }
        % endif

        ${local_mem}[row*buf_nx + idx_1d] = value;
    }
}
% elif dimension == 3:
if (idx_2d < buf_ny * buf_nx) {
    for (int row = 0; row < buf_nz; row++) {
        // Read in 2d-slices
        int temp_x = buf_corner_x + idx_2d % buf_nx;
        int temp_y = buf_corner_y + idx_2d/buf_ny;
        int temp_z = buf_corner_z + row;

        ${num_type} value = ${default_value};
        % if var_name is not None:
        if((temp_x < nx) && (temp_x > 0) && (temp_y < ny) && (temp_y > 0) && (temp_z < nz) && (temp_z > 0)){
            int temp_index = ${get_spatial_index('temp_x', 'temp_y', 'temp_z', 'nx', 'ny', 'nz')};
            value = ${var_name}[temp_index];
        }
        % endif
        ${local_mem}[row*buf_ny*buf_nx + idx_2d] = value;
    }
}
% endif
</%def>

<%def name='read_bc_to_local(var_name, local_mem, default_value)' buffered='True' filter='trim'>
% if dimension==2:
if (idx_1d < buf_nx) {
    for (int row = 0; row < buf_ny; row++) {
        // Read in 1-d slices
        int temp_x = buf_corner_x + idx_1d;
        int temp_y = buf_corner_y + row;

        // If in the bc_map...
        int value = ${default_value};
        if((temp_x < nx + halo) && (temp_x >= -halo) && (temp_y < ny + halo) && temp_y >= -halo){
            int temp_index = ${get_spatial_index('(temp_x + halo)', '(temp_y + halo)', 'nx_bc', 'ny_bc')};
            value = ${var_name}[temp_index];
        }

        ${local_mem}[row*buf_nx + idx_1d] = value;
    }
}
% elif dimension == 3:
if (idx_2d < buf_ny * buf_nx) {
    for (int row = 0; row < buf_nz; row++) {
        // Read in 2d-slices
        //TODO: NEED TO FIX THIS SO THAT IT WORKS THE SAME WAY AS 2D!
        int temp_x = buf_corner_x + idx_2d % buf_nx + halo;
        int temp_y = buf_corner_y + idx_2d/buf_ny + halo;
        int temp_z = buf_corner_z + row + halo;

        // If in the bc_map...
        int value = ${default_value};
        if((temp_x < nx_bc) && (temp_x > 0) && (temp_y < ny_bc) && (temp_y > 0) && (temp_z < nz_bc) && (temp_z > 0)){
            int temp_index = ${get_spatial_index('temp_x', 'temp_y', 'temp_z', 'nx_bc', 'ny_bc', 'nz_bc')};
            value = ${var_name}[temp_index];
        }

        ${local_mem}[row*buf_ny*buf_nx + idx_2d] = value;
    }
}
% endif
</%def>

##### Read in current thread info #####

<%def name='define_thread_location()', buffered='True', filter='trim'>
// Get the spatial index
const int x = get_global_id(0);
const int y = get_global_id(1);
% if dimension == 2:
const int spatial_index = ${get_spatial_index('x', 'y', 'nx', 'ny')};
% else:
const int z = get_global_id(2);
const int spatial_index = ${get_spatial_index('x', 'y', 'z', 'nx', 'ny', 'nz')};
% endif
</%def>

### Determine if thread is in domain
<%def name='if_thread_in_domain()', buffered='True', filter='trim'>
    % if dimension == 2:
    if ((x < nx) && (y < ny))
    % elif dimension == 3:
    if ((x < nx) && (y < ny) && (z < nz))
    % endif
</%def>

<%def name='if_thread_in_bc_domain()', buffered='True', filter='trim'>
    % if dimension == 2:
    if ((x < nx_bc) && (y < ny_bc))
    % elif dimension == 3:
    if ((x < nx_bc) && (y < ny_bc) && (z < nz_bc))
    % endif
</%def>

### Read node type from global memory
<%def name='define_global_bc_index()' buffered='True', filter='trim'>
% if dimension == 2:
const int global_bc_index = ${get_spatial_index('(x + halo)', '(y + halo)', 'nx_bc', 'ny_bc')};
% elif dimension == 3:
const int global_bc_index = ${get_spatial_index('(x + halo)', '(y + halo)', '(z + halo)', 'nx_bc', 'ny_bc', 'nz_bc')};
% endif

</%def>

<%def name='read_node_type_from_global()' buffered='True' filter='trim'>
// Remember, bc-map is larger than nx, ny, nz by a given halo!
${define_global_bc_index()}
const int node_type = bc_map_global[global_bc_index];
</%def>


<%def name='define_jump_index(jump_id="jump_id")' buffered='True' filter='trim'>

% if dimension == 2:
const int jump_index = spatial_index + ${jump_id}*nx*ny;
% elif dimension == 3:
const int jump_index = spatial_index + ${jump_id}*nx*ny*nz;
% endif

</%def>


<%def name='define_all_c(jump_id="jump_id", identifier="const int ")' buffered='True' filter='trim'>

${identifier}cur_cx = c_vec[${get_spatial_index('0', str(jump_id), str(dimension), 'num_jumpers')}];
${identifier}cur_cy = c_vec[${get_spatial_index('1', str(jump_id), str(dimension), 'num_jumpers')}];
%if dimension == 3:
${identifier}cur_cz = c_vec[${get_spatial_index('2',str(jump_id), str(dimension), 'num_jumpers')}];
%endif

</%def>