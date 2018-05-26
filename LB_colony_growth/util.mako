<%!
    from LB_colony_growth.filters import wrap1, wrap2, wrap3, wrap4
    from LB_colony_growth.node_types import node_types
%>

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
%for cur_key in node_types.keys():
#define ${cur_key} ${node_types[cur_key]}
%endfor
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

##### Functions dealing with reading from a halo ####

<%def name='define_local_slice_location()' buffered='True' filter='trim'>
### When looping over rows or 2d slices when reading in local memory, defines where you are
%if dimension == 2:
// Read in 1-d slices
int temp_buf_x = idx_1d;
int temp_buf_y = row;

const int temp_local_index = ${get_spatial_index('temp_buf_x', 'temp_buf_y',
                                                'buf_nx', 'buf_ny')};

int temp_x = buf_corner_x + temp_buf_x;
int temp_y = buf_corner_y + temp_buf_y;
%elif dimension == 3:
// Read in 2-d slices
int temp_buf_x = idx_2d % buf_nx;
int temp_buf_y = idx_2d/buf_nx;
int temp_buf_z = buf_corner_z + row;

const int temp_local_index = ${get_spatial_index('temp_buf_x', 'temp_buf_y', 'temp_buf_z',
                                                'buf_nx', 'buf_ny', 'buf_nz')};

int temp_x = buf_corner_x + cur_buf_x;
int temp_y = buf_corner_y + cur_buf_y;
int temp_z = buf_corner_z + cur_buf_z;
%endif
</%def>

<%def name='if_local_slice_location_in_domain()' buffered='True' filter='trim'>
### When looping over rows or 2d slices when reading in local memory, defines where you are
// If in the domain...
%if dimension == 2:
if((temp_x < nx) && (temp_x >= 0) && (temp_y < ny) && (temp_y >= 0))
%elif dimension == 3:
if((temp_x < nx) && (temp_x >= 0) && (temp_y < ny) && (temp_y >= 0) && (temp_z < nz) && (temp_z >= 0))
%endif
</%def>

<%def name='if_local_idx_in_slice()' buffered='True' filter='trim'>
### When looping over rows or 2d slices when reading in local memory, defines where you are
%if dimension == 2:
if (idx_1d < buf_nx)
%elif dimension == 3:
if (idx_2d < buf_ny * buf_nx)
%endif
</%def>

<%def name='slice_loop_length()' buffered='True' filter='trim'>
### When looping over rows or 2d slices when reading in local memory, defines where you are
%if dimension == 2:
buf_ny
%elif dimension == 3:
buf_nz
%endif
</%def>

<%def name='read_to_local(var_name, local_mem, default_value, unique_bcs, density_map_name=None)' buffered='True' filter='trim'>
### Must be run AFTER the bc_map is read in...
${if_local_idx_in_slice()}{
    for (int row = 0; row < ${slice_loop_length()}; row++) {
        ${define_local_slice_location() | wrap2}

        ${num_type} value = ${default_value};
        % if var_name is not None:
        ${if_local_slice_location_in_domain() | wrap2}{
            %if dimension == 2:
            int temp_index = ${get_spatial_index('temp_x', 'temp_y', 'nx', 'ny')};
            %elif dimension == 3:
            int temp_index = ${get_spatial_index('temp_x', 'temp_y', 'temp_z', 'nx', 'ny', 'nz')};
            %endif
            value = ${var_name}[temp_index];
        }
        % endif

        ${if_local_slice_location_in_bc_map() | wrap2}{
            //If it is, see what value should be on the boundary based on the bc_map.
            const int temp_bc_value = bc_map_local[temp_local_index];

            %if node_types['WALL_NODE'] in unique_bcs:
            if (temp_bc_value == WALL_NODE) value = 0;
            %endif

            %if node_types['PERIODIC'] in unique_bcs:
            else if (temp_bc_value == PERIODIC){
                if (temp_x < 0) temp_x += nx;
                if (temp_x >= nx) temp_x -= nx;

                if (temp_y < 0) temp_y += ny;
                if (temp_y >= ny) temp_y -= ny;

                %if dimension == 3:
                if (temp_z < 0) temp_z += nz;
                if (temp_z >= nz) temp_z -= nz;
                %endif

                % if dimension == 2:
                value = ${var_name}[${get_spatial_index('temp_x', 'temp_y', 'nx', 'ny')}];
                %elif dimension == 3:
                value = ${var_name}[${get_spatial_index(
                                    'temp_x', 'temp_y', 'temp_z',
                                    'nx', 'ny', 'nz')}];
                %endif
            }
            %endif

            %if node_types['FIXED_DENSITY'] in unique_bcs:
            else if (temp_bc_value == FIXED_DENSITY){
                // Read the fixed density value from the density_bc_map
                %if dimension == 2:
                int density_map_index = ${get_spatial_index('(temp_x + halo)', '(temp_y + halo)', 'nx_bc', 'ny_bc')};
                %elif dimension ==3:
                int density_map_index = ${get_spatial_index('(temp_x + halo)', '(temp_y + halo)', '(temp_z + halo)', 'nx_bc', 'ny_bc', 'nz_bc')};
                %endif
                <% assert density_map_name is not None, 'Need to provide input of density map name.' %>
                value = ${density_map_name}[density_map_index];
            }
            %endif
        }

        %if dimension == 2:
        ${local_mem}[row*buf_nx + idx_1d] = value;
        %elif dimension == 3:
        ${local_mem}[row*buf_ny*buf_nx + idx_2d] = value;
        %endif
    }
}
</%def>

<%def name='if_local_slice_location_in_bc_map()' buffered='True' filter='trim'>
### When looping over rows or 2d slices when reading in local memory, defines where you are
// If in the bc_map...
%if dimension == 2:
if(
    (temp_x < nx + halo) && (temp_x >= -halo) &&
    (temp_y < ny + halo) && (temp_y >= -halo))
%elif dimension == 3:
if(
    (temp_x < nx + halo) && (temp_x >= -halo) &&
    (temp_y < ny + halo) && (temp_y >= -halo) &&
    (temp_z < nz + halo) && (temp_z >= -halo))
%endif
</%def>

<%def name='read_bc_to_local(var_name, local_mem, default_value, wrap_periodic=False)' buffered='True' filter='trim'>
${if_local_idx_in_slice()}{
    for (int row = 0; row < ${slice_loop_length()}; row++) {
        ${define_local_slice_location() | wrap2}

        int value = ${default_value};
        ${if_local_slice_location_in_bc_map() | wrap2}
        {
            %if dimension == 2:
            int temp_index = ${get_spatial_index('(temp_x + halo)', '(temp_y + halo)', 'nx_bc', 'ny_bc')};
            %elif dimension ==3:
            int temp_index = ${get_spatial_index('(temp_x + halo)', '(temp_y + halo)', '(temp_z + halo)', 'nx_bc', 'ny_bc', 'nz_bc')};
            %endif
            value = ${var_name}[temp_index];

            %if wrap_periodic:
            ## In practice, this is almost never used. Only used when BC's change in time, i.e. reproducing cells.
            //Read the wrapped periodic value...
            if (value == PERIODIC){
                if (temp_x < 0) temp_x += nx;
                if (temp_x >= nx) temp_x -= nx;

                if (temp_y < 0) temp_y += ny;
                if (temp_y >= ny) temp_y -= ny;

                %if dimension == 3:
                if (temp_z < 0) temp_z += nz;
                if (temp_z >= nz) temp_z -= nz;
                %endif

                %if dimension == 2:
                temp_index = ${get_spatial_index('(temp_x + halo)', '(temp_y + halo)', 'nx_bc', 'ny_bc')};
                %elif dimension ==3:
                temp_index = ${get_spatial_index('(temp_x + halo)', '(temp_y + halo)', '(temp_z + halo)', 'nx_bc', 'ny_bc', 'nz_bc')};
                %endif
                value = ${var_name}[temp_index];
            }
            %endif
        }
        %if dimension == 2:
        ${local_mem}[row*buf_nx + idx_1d] = value;
        %elif dimension == 3:
        ${local_mem}[row*buf_ny*buf_nx + idx_2d] = value;
        %endif
    }
}
</%def>

### Code to fix the halo values depending on the BC's.

##         %if periodic_replacement:
##         // Replace values with periodic analogs...painful
##         if (value == PERIODIC){
##             if (temp_x < 0) temp_x += nx;
##             if (temp_x >= nx) temp_x -= nx;
##
##             if (temp_y < 0) temp_y += ny;
##             if (temp_y >= ny) temp_y -= ny;
##
##             temp_index = ${get_spatial_index('(temp_x + halo)', '(temp_y + halo)', 'nx_bc', 'ny_bc')}
##         }
##         %endif

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

<%def name='define_streamed_index_local()' buffered='True' filter='trim'>

% if dimension == 2:
int streamed_index_local = ${get_spatial_index('(buf_x + cur_cx)', '(buf_y + cur_cy)', 'buf_nx', 'buf_ny')};
% elif dimension == 3:
int streamed_index_local = ${get_spatial_index(
    '(buf_x + cur_cx)', '(buf_y + cur_cy)', '(buf_z + cur_cz)',
    'buf_nx', 'buf_ny', 'buf_nz'
)};
% endif

</%def>

<%def name='define_streamed_index_global(identifier="int")' buffered='True' filter='trim'>

% if dimension == 2:
${identifier} streamed_index_global = ${get_spatial_index('(x + cur_cx)', '(y + cur_cy)', 'nx', 'ny')};
% elif dimension == 3:
${identifier} streamed_index_global = ${get_spatial_index(
    '(x + cur_cx)', '(y + cur_cy)', '(z + cur_cz)',
    'nx', 'ny', 'nz'
)};
% endif

</%def>

##### Useful copy kernels ######

<%namespace file='kernel.mako' import='*' name='kernel' />

<%def name='needs_copy_streamed_onto_f_kernel()' filter='trim'>

${set_current_kernel('copy_streamed_onto_f')}

## Needed global variables
${needs_f()}
${needs_f_streamed()}

## Velocity set info
## Assume that num_jumpers is defined already in the "define" part of the code
## ${needs_num_jumpers()}

__kernel void
copy_streamed_onto_f(
    ${print_kernel_args()}
)
{
    ${define_thread_location() | wrap1}

    ${if_thread_in_domain() | wrap1}{
        for(int jump_id = 0; jump_id < num_jumpers; jump_id++){
            ${define_jump_index()}

            f_global[jump_index] = f_streamed_global[jump_index];
        }
    }
}

</%def>

<%def name='needs_copy_streamed_onto_bc_kernel()' filter='trim'>

${set_current_kernel('copy_streamed_onto_bc')}
${needs_bc_map('__write_only')}
${needs_bc_map_streamed('__read_only')}

__kernel void
copy_streamed_onto_bc(
    ${print_kernel_args()}
)
{
    ${define_thread_location() | wrap1}

    ${if_thread_in_bc_domain() | wrap1}{
        bc_map_global[spatial_index] = bc_map_streamed_global[spatial_index];
    }
}

</%def>