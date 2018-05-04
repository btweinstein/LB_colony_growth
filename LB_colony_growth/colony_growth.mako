% if num_type=='double':
#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
    #error "Double precision floating point not supported by OpenCL implementation."
#endif
% endif

// Define domain size
#define nx ${nx}
#define ny ${ny}
% if dimension == 3:
#define nz ${nz}
% endif
// Define boundary map domain size
#define nx_bc ${nx_bc}
#define ny_bc ${ny_bc}
% if dimension == 3:
#define nz_bc ${nz_bc}
% endif

#define SMALL 1e-6

%if dimension==2:
#define NUM_NEAREST_NEIGHBORS 4
__constant int cx_nearest[4] = {1, -1, 0, 0};
__constant int cy_nearest[4] = {0,  0, 1,-1};
%elif dimension == 3:
#define NUM_NEAREST_NEIGHBORS 6
__constant int cx_nearest[6] = {1, -1, 0, 0, 0, 0};
__constant int cy_nearest[6] = {0,  0, 1,-1, 0, 0};
__constant int cz_nearest[6] = {0,  0, 0, 0, 1,-1};
%endif

//The code is always ok, AS LONG as the halo is one! Regardless of the stencil.
// If any more, everything breaks.
#define halo 1

#define FLUID_NODE 0
#define WALL_NODE 1
#define NOT_IN_DOMAIN 2
//Alleles get negative numbers as identifiers
%for i in range(1, num_alleles + 1):
#define ALLELE_${i} ${-1*i}
%endfor

inline int get_spatial_index(
    int x, int y,
    int x_size, y_size)
{
    return y*x_size + x;
}

inline int get_spatial_index(
    int x, int y, int z,
    int x_size, int y_size, int z_size)
{
    return z*y_size*x_size + y*x_size + x;
}

inline int get_spatial_index(
    int x, int y, int z, int jump_id,
    int x_size, int y_size, int z_size, int num_jumpers)
{
    return jump_id * z_size*y_size *x_size + z*y_size*x_size + y*x_size + x;
}

### Helpful filters ###
<%!

space4 = '    '

def wrap1(t):
    return t.replace('\n', '\n' + space4)

def wrap2(t):
    return t.replace('\n', '\n' + space4 + space4)

def wrap3(t):
    return t.replace('\n', '\n' + space4 + space4 + space4)

def wrap4(t):
    return t.replace('\n', '\n' + space4 + space4 + space4 + space4)
%>

<%
def print_kernel_args(cur_kernel_list):
    num_args = len(cur_kernel_list)
    for i in range(num_args):
        context.write('     ')
        context.write(cur_kernel_list[i][1])
        if i < num_args - 1:
            context.write(',\n')
%>

######### Utility functions #################

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

const int nx_local = get_local_size(0);
const int ny_local = get_local_size(1);
% if dimension == 3:
const int nz_local = get_local_size(2);
% endif

// Index of thread within our work-group
% if dimension == 2:
const int idx_1d = get_spatial_index(lx, ly, nx_local, ny_local);
% elif dimension == 3:
const int idx_2d = get_spatial_index(lx, ly, lz, nx_local, ny_local, nz_local);
% endif
</%def>

<%def name='read_to_local(var_name, local_mem, default_value)' buffered='True' filter='trim'>
% if dimension==2:
if (idx_1D < buf_nx) {
    for (int row = 0; row < buf_ny; row++) {
        // Read in 1-d slices
        int temp_x = buf_corner_x + idx_1D;
        int temp_y = buf_corner_y + row;

        ${num_type} value = ${default_value};
        % if var_name is not None:
        // If in the domain...
        if((temp_x < nx) && (temp_x > 0) && (temp_y < ny) && (temp_y > 0)){
            int temp_index = get_spatial_index(temp_x, temp_y, nx, ny);
            value = ${var_name}[temp_index];
        }
        % endif

        ${local_mem}[row*buf_nx + idx_1D] = value;
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
            int temp_index = get_spatial_index(temp_x, temp_y, temp_z, nx, ny, nz);
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
if (idx_1D < buf_nx) {
    for (int row = 0; row < buf_ny; row++) {
        // Read in 1-d slices
        int temp_x = buf_corner_x + idx_1D + halo;
        int temp_y = buf_corner_y + row + halo;

        // If in the bc_map...
        int value = ${default_value};
        if((temp_x < nx_bc) && (temp_x > 0) && (temp_y < ny_bc) && (temp_y > 0)){
            int temp_index = get_spatial_index(temp_x, temp_y, nx_bc, ny_bc);
            value = ${var_name}[temp_index];
        }

        ${local_mem}[row*buf_nx + idx_1D] = value;
    }
}
% elif dimension == 3:
if (idx_2d < buf_ny * buf_nx) {
    for (int row = 0; row < buf_nz; row++) {
        // Read in 2d-slices
        int temp_x = buf_corner_x + idx_2d % buf_nx + halo;
        int temp_y = buf_corner_y + idx_2d/buf_ny + halo;
        int temp_z = buf_corner_z + row + halo;

        // If in the bc_map...
        int value = ${default_value};
        if((temp_x < nx_bc) && (temp_x > 0) && (temp_y < ny_bc) && (temp_y > 0) && (temp_z < nz_bc) && (temp_z > 0)){
            int temp_index = get_spatial_index(temp_x, temp_y, temp_z, nx_bc, ny_bc, nz_bc);
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
const int spatial_index = get_spatial_index(x, y, nx, ny);
% else:
const int z = get_global_id(2);
const int spatial_index = get_spatial_index(x, y, z, nx, ny, nz);
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

### Read node type from global memory
<%def name='read_node_type_from_global()' buffered='True' filter='trim'>
// Remember, bc-map is larger than nx, ny, nz by a given halo!
const int x_bc = halo + x;
const int y_bc = halo + y;
%if dimension == 3:
const int z_bc = halo + z;
%endif

% if dimension == 2:
int bc_index = get_spatial_index(x_bc, y_bc, nx_bc, ny_bc);
% elif dimension == 3:
int bc_index = get_spatial_index(x_bc, y_bc, z_bc, nx_bc, ny_bc, nz_bc);
% endif

const int node_type = bc_map[bc_index];
</%def>


<%def name='define_jump_index(jump_id="jump_id")' buffered='True' filter='trim'>
% if dimension == 2:
int jump_index = get_spatial_index(x, y, ${jump_id}, nx, ny, num_jumpers);
% elif dimension == 3:
int jump_index = get_spatial_index(x, y, z, ${jump_id}, nx, ny, nz, num_jumpers);
% endif
</%def>


<%def name='define_streamed_index_local()' buffered='True' filter='trim'>

int cur_cx = c_vec[get_spatial_index(0, jump_id, ${dimension}, num_jumpers)];
int cur_cy = c_vec[get_spatial_index(1, jump_id, ${dimension}, num_jumpers)];
%if dimension == 3:
int cur_cz = c_vec[get_spatial_index(2, jump_id, ${dimension}, num_jumpers)];
%endif

// Figure out what type of node the steamed position is
const int stream_x_local = buf_x + cur_cx;
const int stream_y_local = buf_y + cur_cy;
%if dimension == 3:
const int stream_z_local = buf_z + cur_cz;
%endif

% if dimension == 2:
int streamed_index_local = get_spatial_index(stream_x_local, stream_y_local, buf_nx, buf_ny);
% elif dimension == 3:
int streamed_index_local = get_spatial_index(
    stream_x_local, stream_y_local, stream_z_local,
    buf_nx, buf_ny, buf_nz
);
% endif

</%def>

######### Collide & Propagate kernel ########

<%
    cur_kernel = 'collide_and_propagate'
    kernel_arguments[cur_kernel] = []
    cur_kernel_list = kernel_arguments[cur_kernel]

    cur_kernel_list.append(['bc_map', '__global __read_only int *bc_map_global'])
    cur_kernel_list.append(['num_jumpers', 'const int num_jumpers'])
    cur_kernel_list.append(['f', '__global '+num_type+' *f_global'])
    cur_kernel_list.append(['feq', '__global __read_only '+num_type+' *feq_global'])
    cur_kernel_list.append(['omega', 'const '+num_type+' omega'])
    cur_kernel_list.append(['c_vec', '__constant int *c_vec'])
    cur_kernel_list.append(['c_mag', '__constant '+num_type+' *c_mag'])
    cur_kernel_list.append(['w', '__constant '+num_type+' *w'])
    cur_kernel_list.append(['num_jumpers', 'const int num_jumpers'])
    cur_kernel_list.append(['rho', '__global '+num_type+' *rho_global'])
    cur_kernel_list.append(['buf_nx', 'const int buf_nx'])
    cur_kernel_list.append(['buf_ny', 'const int buf_ny'])
    cur_kernel_list.append(['buf_nz', 'const int buf_nz'])
    cur_kernel_list.append(['local_mem', '__local '+num_type+' *rho_local'])
    cur_kernel_list.append(['local_mem', '__local '+num_type+' *bc_map_local'])
    cur_kernel_list.append(['k', 'const '+num_type+' k'])
    cur_kernel_list.append(['D', 'const '+num_type+' D'])
%>

__kernel void
collide_and_propagate(
<%
    print_kernel_args(cur_kernel_list)
%>
)
{
    // Get info about where thread is located in global memory
    ${define_thread_location() | wrap1}

    // We need local memory...define necessary variables.
    ${define_local_variables() | wrap1}
    // Read concentration and absorbed mass at nodes into memory

    barrier(CLK_LOCAL_MEM_FENCE);
    ${read_to_local('rho_global', 'rho_local', 0) | wrap1}
    ${read_bc_to_local('bc_map_global', 'bc_map_local', 'NOT_IN_DOMAIN') | wrap1}
    barrier(CLK_LOCAL_MEM_FENCE);

    // Main loop...
    ${if_thread_in_domain() | wrap1}{
        % if dimension == 2:
        const int local_bc_index = get_spatial_index(buf_x, buf_y, buf_nx, buf_ny);
        % elif dimension == 3:
        const int local_bc_index = get_spatial_index(buf_x, buf_y, buf_z, buf_nx, buf_ny, buf_nz);
        % endif

        const int node_type = bc_map_local[local_bc_index];
        if(node_type == FLUID_NODE){
            for(int jump_id=0; jump_id < num_jumpers; jump_id++){
                ${define_jump_index() | wrap4}

                ${collide_bgk() | wrap4}

                ${move() | wrap4}
            }
        }
        else if (node_type < 0){ // Population node!
            ${absorb_mass() | wrap3}
        }
    }
}

<%def name='absorb_mass()' buffered='True' filter='trim'>
// Loop over nearest neighbors

${num_type} mass_to_add = 0;
for(int i=0; i < NUM_NEAREST_NEIGHBORS; i++){
    const int cur_cx = cx_nearest[i];
    const int cur_cy = cy_nearest[i];
    %if dimension == 3:
    const int cur_cz = cz_nearest[i];
    %endif

    // Figure out what type of node the steamed position is
    const int stream_x_local = buf_x + cur_cx;
    const int stream_y_local = buf_y + cur_cy;
    %if dimension == 3:
    const int stream_z_local = buf_z + cur_cz;
    %endif

    % if dimension == 2:
    int streamed_index_local = get_spatial_index(stream_x_local, stream_y_local, buf_nx, buf_ny);
    % elif dimension == 3:
    int streamed_index_local = get_spatial_index(
        stream_x_local, stream_y_local, stream_z_local,
        buf_nx, buf_ny, buf_nz
    );
    % endif

    const int streamed_bc = bc_map_local[streamed_index_local];

    if (streamed_bc == FLUID_NODE){ // Scalar can diffuse in
        // Determine Cwall via finite difference
        % if dimension ==2:
        const ${num_type} cur_rho = rho_local[get_spatial_index(buf_x, buf_y, buf_nx, buf_ny)];
        %elif dimension == 3:
        const ${num_type} cur_rho = rho_local[get_spatial_index(
            buf_x, buf_y, buf_z,
            buf_nx, buf_ny, buf_nz)
        ];
        % endif

        const ${num_type} cur_c_mag = 1.0; // Magnitude to nearest neighbors is always one
        const ${num_type} rho_wall = cur_rho/(1 + (k*cur_c_mag)/(2*D));

        //Update the mass at the site accordingly
        // Flux in is k*rho_wall...and in lattice units, all additional factors are one.
        mass_to_add += k*rho_wall;
    }
}
absorbed_mass_global[spatial_index] += mass_to_add;

</%def>

<%def name='collide_bgk()' buffered='True' filter='trim'>
${num_type} f_after_collision = f_global[jump_index]*(1-omega) + omega*feq_global[jump_index];
//TODO: If a source is needed, additional terms are needed here.
</%def>

<%def name='move()' buffered='True' filter='trim'>
// After colliding, stream to the appropriate location. Needed to write collision to f6

${define_streamed_index_local()}

const int streamed_bc = bc_map_local[streamed_index_local];

int streamed_index = -1; // Initialize to a nonsense value

if (streamed_bc == FLUID_NODE){
    // Propagate the collided particle distribution as appropriate

    int stream_x = x + cur_cx;
    int stream_y = y + cur_cy;
    % if dimension == 3:
    int stream_z = z + cur_cz;
    % endif

    % if dimension == 2:
    streamed_index = get_spatial_index(stream_x, stream_y, jump_id, nx, ny, num_jumpers);
    % elif dimension == 3:
    streamed_index = get_spatial_index(stream_x, stream_y, stream_z, jump_id, nx, ny, nz, num_jumpers);
    % endif
}
else if (streamed_bc == WALL_NODE){ // Bounceback; impenetrable boundary
    int reflect_id = reflect_list[jump_id];
    % if dimension == 2:
    int reflect_index = get_spatial_index(x, y, reflect_id, nx, ny, num_jumpers);
    % elif dimension == 3:
    int reflect_index = get_spatial_index(x, y, z, reflect_id, nx, ny, nz, num_jumpers);
    % endif

    f_streamed_global[reflect_index] = f_after_collision;

    // The streamed part collides without moving.
    streamed_index = get_spatial_index(x, y, z, jump_id, nx, ny, nz, num_jumpers);
}

else if (streamed_bc < 0){ // You are at a population node
    // Determine Cwall via finite difference
    % if dimension ==2:
    ${num_type} cur_rho = rho_local[get_spatial_index(buf_x, buf_y, buf_nx, buf_ny)];
    %elif dimension == 3:
    ${num_type} cur_rho = rho_local[get_spatial_index(buf_x, buf_y, buf_z, buf_nx, buf_ny, buf_nz)];
    % endif

    ${num_type} cur_c_mag = c_mag[jump_id];
    ${num_type} rho_wall = cur_rho/(1 + (k*cur_c_mag)/(2*D));

    // Based on rho_wall, do the bounceback
    ${num_type} cur_w = w[jump_id];
    int reflect_id = reflect_list[jump_id];
    % if dimension == 2:
    int reflect_index = get_spatial_index(x, y, reflect_id, nx, ny, num_jumpers);
    % elif dimension == 3:
    int reflect_index = get_spatial_index(x, y, z, reflect_id, nx, ny, nz, num_jumpers);
    % endif

    f_streamed_global[reflect_index] = -f_after_collision + 2*cur_w*rho_wall;

    // The streamed part collides without moving.
    streamed_index = get_spatial_index(x, y, z, jump_id, nx, ny, nz, num_jumpers);
}

//Need to write to the streamed buffer...otherwise out of sync problems will occur
f_streamed_global[streamed_index] = f_after_collision;
</%def>

######### Update after streaming kernel #########
<%
    cur_kernel = 'update_after_streaming'
    kernel_arguments[cur_kernel] = []
    cur_kernel_list = kernel_arguments[cur_kernel]
%>

__kernel void
update_after_streaming(
<%
    print_kernel_args(cur_kernel_list)
%>
)
{
    // Get info about where thread is located in global memory
    ${define_thread_location() | wrap1}

    // Main loop...
    ${if_thread_in_domain() | wrap1}{
        // Figure out what type of node is present
        ${read_node_type_from_global() | wrap2}
        if (node_type == FLUID_NODE){
            ${update_hydro() | wrap3}
            ${update_feq() | wrap3}
        }
        else{
            // No concentration is present...act accordingly.
            rho_global[spatial_index] = 0;
            // We do not bother updating f & feq; we are interested in the macro variables.
        }
    }
}

<%def name='update_hydro()' buffered='True' filter='trim'>
// Update rho!
${num_type} new_rho = 0;

for(int jump_id=0; jump_id < num_jumpers; jump_id++){
    ${define_jump_index() | wrap1}

    ${num_type} cur_f = f_global[jump_index];

    new_rho += cur_f;
}

rho_global[spatial_index] = new_rho;
</%def>

<%def name='update_feq()' buffered='True' filter='trim'>
//Using the udpated hydrodynamic variables, update feq.
//Luckily, in this simple scenario, there is no velocity! And no cs!

for(int jump_id=0; jump_id < num_jumpers; jump_id++){
    ${define_jump_index() | wrap1}

    ${num_type} cur_w = w[jump_id];

    ${num_type} new_feq = cur_w*new_rho;

    feq_global[jump_index] = new_feq;
}

</%def>

######### Reproduce cells kernel #########
<%
    cur_kernel = 'reproduce'
    kernel_arguments[cur_kernel] = []
    cur_kernel_list = kernel_arguments[cur_kernel]
%>

__kernel void
reproduce(
<%
    print_kernel_args(cur_kernel_list)
%>
)
{
    // Get info about where thread is located in global memory
    ${define_thread_location() | wrap1}
    // We need local memory, notably the BC-map.

    // We need local memory...define necessary variables.
    ${define_local_variables() | wrap1}
    // Read concentration and absorbed mass at nodes into memory

    barrier(CLK_LOCAL_MEM_FENCE);
    ${read_bc_to_local('bc_map_global', 'bc_map_local', 'NOT_IN_DOMAIN') | wrap1}
    barrier(CLK_LOCAL_MEM_FENCE);

    // Main loop...
    ${if_thread_in_domain() | wrap1}{
        // Figure out what type of node is present
        % if dimension == 2:
        const int local_bc_index = get_spatial_index(buf_x, buf_y, buf_nx, buf_ny);
        % elif dimension == 3:
        const int local_bc_index = get_spatial_index(buf_x, buf_y, buf_z, buf_nx, buf_ny, buf_nz);
        % endif

        const int node_type = bc_map_local[local_bc_index];

        if (node_type < 0){ // Population node!
            //Check if you have accumulated enough mass
            ${num_type} current_mass = absorbed_mass_global[spatial_index];
            if (current_mass > m_reproduce){
                ${reproduce() | wrap3}
            }
        }
    }
}

<%def name='reproduce()' buffered='True' filter='trim'>

// Calculate the normalization constant
${num_type} norm_constant = 0;

for(int jump_id=0; jump_id < num_jumpers; jump_id++){
    ${define_streamed_index_local()}

    const int streamed_node_type = bc_map_local[streamed_index_local];
    bool can_reproduce = false;

    if (streamed_node_type == FLUID_NODE){ // Population can expand into this!
        norm_constant += w[jump_id];
        can_reproduce = true;
    }
}

//If you can't reproduce, go no further.
if (can_reproduce){

}

</%def>


###################### Old Stuff #############################


__kernel void
update_feq_fluid(
    __global __write_only ${num_type} *feq_global,
    __global __read_only ${num_type} *rho_global,
    __global __read_only ${num_type} *u_bary_global,
    __global __read_only ${num_type} *v_bary_global,
    __constant ${num_type} *w_arr,
    __constant int *cx_arr,
    __constant int *cy_arr,
    const ${num_type} cs,
    const int nx, const int ny,
    const int field_num,
    const int num_populations,
    const int num_jumpers)
{
    //Input should be a 2d workgroup. But, we loop over a 4d array...
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    const int two_d_index = y*nx + x;

    if ((x < nx) && (y < ny)){

        int three_d_index = field_num*nx*ny + two_d_index;

        const ${num_type} rho = rho_global[three_d_index];
        const ${num_type} u = u_bary_global[two_d_index];
        const ${num_type} v = v_bary_global[two_d_index];

        // Now loop over every jumper

        const ${num_type} cs_squared = cs*cs;
        const ${num_type} two_cs_squared = 2*cs_squared;
        const ${num_type} three_cs_squared = 3*cs_squared;
        const ${num_type} two_cs_fourth = 2*cs*cs*cs*cs;
        const ${num_type} six_cs_sixth = 6*cs*cs*cs*cs*cs*cs;

        for(int jump_id=0; jump_id < num_jumpers; jump_id++){
            const int four_d_index = jump_id*num_populations*nx*ny + three_d_index;

            const ${num_type} w = w_arr[jump_id];
            const int cx = cx_arr[jump_id];
            const int cy = cy_arr[jump_id];

            const ${num_type} c_dot_u = cx*u + cy*v;
            const ${num_type} u_squared = u*u + v*v;

            ${num_type} new_feq = 0;
            if (num_jumpers == 9){ //D2Q9
                new_feq =
                w*rho*(
                1
                + c_dot_u/cs_squared
                + (c_dot_u*c_dot_u)/(two_cs_fourth)
                - u_squared/(two_cs_squared)
                );
            }
            else if(num_jumpers == 25){ //D2Q25
                new_feq =
                w*rho*(
                1
                + c_dot_u/cs_squared
                + (c_dot_u*c_dot_u)/(two_cs_fourth)
                - u_squared/(two_cs_squared)
                + (c_dot_u * (c_dot_u*c_dot_u - three_cs_squared*u_squared))/(six_cs_sixth)
                );
            }

            feq_global[four_d_index] = new_feq;
        }
    }
}

__kernel void
collide_particles_fluid(
    __global ${num_type} *f_global,
    __global __read_only ${num_type} *feq_global,
    __global __read_only ${num_type} *rho_global,
    __global __read_only ${num_type} *u_bary_global,
    __global __read_only ${num_type} *v_bary_global,
    __global __read_only ${num_type} *Gx_global,
    __global __read_only ${num_type} *Gy_global,
    const ${num_type} omega,
    __constant ${num_type} *w_arr,
    __constant int *cx_arr,
    __constant int *cy_arr,
    const int nx, const int ny,
    const int cur_field,
    const int num_populations,
    const int num_jumpers,
    const ${num_type} cs)
{
    //Input should be a 2d workgroup! Loop over the third dimension.
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if ((x < nx) && (y < ny)){
        const int two_d_index = y*nx + x;
        int three_d_index = cur_field*ny*nx + two_d_index;

        const ${num_type} rho = rho_global[three_d_index];
        const ${num_type} u = u_bary_global[two_d_index];
        const ${num_type} v = v_bary_global[two_d_index];
        const ${num_type} Gx = Gx_global[three_d_index];
        const ${num_type} Gy = Gy_global[three_d_index];

        const ${num_type} cs_squared = cs*cs;
        const ${num_type} cs_fourth = cs*cs*cs*cs;
        const ${num_type} Fi_prefactor = (1 - .5*omega);

        for(int jump_id=0; jump_id < num_jumpers; jump_id++){
            int four_d_index = jump_id*num_populations*ny*nx + three_d_index;

            ${num_type} relax = f_global[four_d_index]*(1-omega) + omega*feq_global[four_d_index];
            //Calculate Fi
            ${num_type} c_dot_F = cx_arr[jump_id] * Gx + cy_arr[jump_id] * Gy;
            ${num_type} c_dot_u = cx_arr[jump_id] * u  + cy_arr[jump_id] * v;
            ${num_type} u_dot_F = Gx * u + Gy * v;

            ${num_type} w = w_arr[jump_id];

            ${num_type} Fi = Fi_prefactor*w*(
                c_dot_F/cs_squared
                + c_dot_F*c_dot_u/cs_fourth
                - u_dot_F/cs_squared
            );

            f_global[four_d_index] = relax + Fi;
        }
    }
}

__kernel void
add_eating_collision(
    const int eater_index,
    const int eatee_index,
    const ${num_type} eat_rate,
    const ${num_type} eater_cutoff,
    __global ${num_type} *f_global,
    __global __read_only ${num_type} *rho_global,
    __constant ${num_type} *w_arr,
    __constant int *cx_arr,
    __constant int *cy_arr,
    const int nx, const int ny,
    const int num_populations,
    const int num_jumpers,
    const ${num_type} cs)
{
    //Input should be a 2d workgroup! Loop over the third dimension.
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if ((x < nx) && (y < ny)){
        const int two_d_index = y*nx + x;
        int three_d_eater_index = eater_index*ny*nx + two_d_index;
        int three_d_eatee_index = eatee_index*ny*nx + two_d_index;

        const ${num_type} rho_eater = rho_global[three_d_eater_index];
        const ${num_type} rho_eatee = rho_global[three_d_eatee_index];

        ${num_type} all_growth = 0;
        if (rho_eater > eater_cutoff){
            all_growth = eat_rate*rho_eater*rho_eatee; // Eat a diffusing solute
        }

        for(int jump_id=0; jump_id < num_jumpers; jump_id++){
            int four_d_eater_index = jump_id*num_populations*ny*nx + three_d_eater_index;
            int four_d_eatee_index = jump_id*num_populations*ny*nx + three_d_eatee_index;

            float w = w_arr[jump_id];

            f_global[four_d_eater_index] += w * all_growth;
            f_global[four_d_eatee_index] -= w * all_growth;
        }
    }
}

__kernel void
add_growth(
    const int eater_index,
    const ${num_type} min_rho_cutoff,
    const ${num_type} max_rho_cutoff,
    const ${num_type} eat_rate,
    __global ${num_type} *f_global,
    __global __read_only ${num_type} *rho_global,
    __constant ${num_type} *w_arr,
    __constant int *cx_arr,
    __constant int *cy_arr,
    const int nx, const int ny,
    const int num_populations,
    const int num_jumpers,
    const ${num_type} cs)
{
    //Input should be a 2d workgroup! Loop over the third dimension.
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if ((x < nx) && (y < ny)){
        const int two_d_index = y*nx + x;
        int three_d_eater_index = eater_index*ny*nx + two_d_index;

        const ${num_type} rho_eater = rho_global[three_d_eater_index];

        ${num_type} all_growth = 0;
        // Only grow if you are in the correct phase...
        if ((rho_eater > min_rho_cutoff) && (rho_eater < max_rho_cutoff)){
            all_growth = eat_rate;
        }

        for(int jump_id=0; jump_id < num_jumpers; jump_id++){
            int four_d_eater_index = jump_id*num_populations*ny*nx + three_d_eater_index;
            float w = w_arr[jump_id];
            f_global[four_d_eater_index] += w * all_growth;
        }
    }
}

__kernel void
update_bary_velocity(
    __global ${num_type} *u_bary_global,
    __global ${num_type} *v_bary_global,
    __global __read_only ${num_type} *rho_global,
    __global __read_only ${num_type} *f_global,
    __global __read_only ${num_type} *Gx_global,
    __global __read_only ${num_type} *Gy_global,
    __constant ${num_type} *tau_arr,
    __constant ${num_type} *w_arr,
    __constant int *cx_arr,
    __constant int *cy_arr,
    const int nx, const int ny,
    const int num_populations,
    const int num_jumpers
    )
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if ((x < nx) && (y < ny)){
        const int two_d_index = y*nx + x;

        ${num_type} sum_x = 0;
        ${num_type} sum_y = 0;
        ${num_type} rho_sum = 0;

        for(int cur_field=0; cur_field < num_populations; cur_field++){
            int three_d_index = cur_field*ny*nx + two_d_index;

            ${num_type} cur_rho = rho_global[three_d_index];
            rho_sum += cur_rho;

            ${num_type} Gx = Gx_global[three_d_index];
            ${num_type} Gy = Gy_global[three_d_index];

            for(int jump_id=0; jump_id < num_jumpers; jump_id++){
                int four_d_index = jump_id*num_populations*ny*nx + three_d_index;
                ${num_type} f = f_global[four_d_index];
                int cx = cx_arr[jump_id];
                int cy = cy_arr[jump_id];

                sum_x += cx * f;
                sum_y += cy * f;
            }
            sum_x += Gx/2.;
            sum_y += Gy/2.;
        }
        u_bary_global[two_d_index] = sum_x/rho_sum;
        v_bary_global[two_d_index] = sum_y/rho_sum;
    }
}

__kernel void
update_hydro_fluid(
    __global __read_only ${num_type} *f_global,
    __global ${num_type} *rho_global,
    __global ${num_type} *u_global,
    __global ${num_type} *v_global,
    __global __read_only ${num_type} *Gx_global,
    __global __read_only ${num_type} *Gy_global,
    __constant ${num_type} *w_arr,
    __constant int *cx_arr,
    __constant int *cy_arr,
    const int nx, const int ny,
    const int cur_field,
    const int num_populations,
    const int num_jumpers
)
{
    //Input should be a 2d workgroup! Loop over the third dimension.
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if ((x < nx) && (y < ny)){
        const int two_d_index = y*nx + x;
        int three_d_index = cur_field*ny*nx + two_d_index;

        // Update rho!
        ${num_type} new_rho = 0;
        ${num_type} new_u = 0;
        ${num_type} new_v = 0;

        for(int jump_id=0; jump_id < num_jumpers; jump_id++){
            int four_d_index = jump_id*num_populations*ny*nx + three_d_index;
            ${num_type} f = f_global[four_d_index];

            new_rho += f;

            int cx = cx_arr[jump_id];
            int cy = cy_arr[jump_id];

            new_u += f*cx;
            new_v += f*cy;
        }
        rho_global[three_d_index] = new_rho;

        if(new_rho > ZERO_DENSITY){
            u_global[three_d_index] = new_u/new_rho;
            v_global[three_d_index] = new_v/new_rho;
        }
        else{
            u_global[three_d_index] = 0;
            v_global[three_d_index] = 0;
        }
    }
}


__kernel void
move_open_bcs(
    __global __read_only ${num_type} *f_global,
    const int nx, const int ny,
    const int cur_field,
    const int num_populations,
    const int num_jumpers)
{
    //Input should be a 2d workgroup!
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if ((x < nx) && (y < ny)){ // Make sure you are in the domain

        //LEFT WALL: ZERO GRADIENT, no corners
        if ((x==0) && (y >= 1)&&(y < ny-1)){
            for(int jump_id = 0; jump_id < num_jumpers; jump_id++){
                int four_d_index = jump_id*num_populations*nx*ny +  cur_field*nx*ny + y*nx + x;
                int new_x = 1;
                int new_four_d_index = jump_id*num_populations*nx*ny +  cur_field*nx*ny + y*nx + new_x;
                f_global[four_d_index] = f_global[new_four_d_index];
            }
        }

        //RIGHT WALL: ZERO GRADIENT, no corners
        else if ((x==nx - 1) && (y >= 1)&&(y < ny-1)){
            for(int jump_id = 0; jump_id < num_jumpers; jump_id++){
                int four_d_index = jump_id*num_populations*nx*ny +  cur_field*nx*ny + y*nx + x;
                int new_x = nx - 2;
                int new_four_d_index = jump_id*num_populations*nx*ny +  cur_field*nx*ny + y*nx + new_x;
                f_global[four_d_index] = f_global[new_four_d_index];
            }
        }

        //We need a barrier here! The top piece must run before the bottom one...

        //TOP WALL: ZERO GRADIENT, no corners
        else if ((y == ny - 1)&&((x >= 1)&&(x < nx-1))){
            for(int jump_id = 0; jump_id < num_jumpers; jump_id++){
                int four_d_index = jump_id*num_populations*nx*ny +  cur_field*nx*ny + y*nx + x;
                int new_y = ny - 2;
                int new_four_d_index = jump_id*num_populations*nx*ny +  cur_field*nx*ny + new_y*nx + x;
                f_global[four_d_index] = f_global[new_four_d_index];
            }
        }

        //BOTTOM WALL: ZERO GRADIENT, no corners
        else if ((y == 0)&&((x >= 1)&&(x < nx-1))){
            for(int jump_id = 0; jump_id < num_jumpers; jump_id++){
                int four_d_index = jump_id*num_populations*nx*ny +  cur_field*nx*ny + y*nx + x;
                int new_y = 1;
                int new_four_d_index = jump_id*num_populations*nx*ny +  cur_field*nx*ny + new_y*nx + x;
                f_global[four_d_index] = f_global[new_four_d_index];
            }
        }

        //BOTTOM LEFT CORNER
        else if ((x == 0)&&((y == 0))){
            for(int jump_id = 0; jump_id < num_jumpers; jump_id++){
                int four_d_index = jump_id*num_populations*nx*ny +  cur_field*nx*ny + y*nx + x;
                int new_x = 1;
                int new_y = 1;
                int new_four_d_index = jump_id*num_populations*nx*ny +  cur_field*nx*ny + new_y*nx + new_x;
                f_global[four_d_index] = f_global[new_four_d_index];
            }
        }

        //TOP LEFT CORNER
        else if ((x == 0)&&((y == ny-1))){
            for(int jump_id = 0; jump_id < num_jumpers; jump_id++){
                int four_d_index = jump_id*num_populations*nx*ny +  cur_field*nx*ny + y*nx + x;
                int new_x = 1;
                int new_y = ny - 2;
                int new_four_d_index = jump_id*num_populations*nx*ny +  cur_field*nx*ny + new_y*nx + new_x;
                f_global[four_d_index] = f_global[new_four_d_index];
            }
        }

        //BOTTOM RIGHT CORNER
        else if ((x == nx - 1)&&((y == 0))){
            for(int jump_id = 0; jump_id < num_jumpers; jump_id++){
                int four_d_index = jump_id*num_populations*nx*ny +  cur_field*nx*ny + y*nx + x;
                int new_x = nx - 2;
                int new_y = 1;
                int new_four_d_index = jump_id*num_populations*nx*ny +  cur_field*nx*ny + new_y*nx + new_x;
                f_global[four_d_index] = f_global[new_four_d_index];
            }
        }

        //TOP RIGHT CORNER
        else if ((x == nx - 1)&&((y == ny - 1))){
            for(int jump_id = 0; jump_id < num_jumpers; jump_id++){
                int four_d_index = jump_id*num_populations*nx*ny +  cur_field*nx*ny + y*nx + x;
                int new_x = nx - 2;
                int new_y = ny - 2;
                int new_four_d_index = jump_id*num_populations*nx*ny +  cur_field*nx*ny + new_y*nx + new_x;
                f_global[four_d_index] = f_global[new_four_d_index];
            }
        }
    }
}


__kernel void
copy_streamed_onto_f(
    __global __write_only ${num_type} *f_streamed_global,
    __global __read_only ${num_type} *f_global,
    __constant int *cx,
    __constant int *cy,
    const int nx, const int ny,
    const int cur_field,
    const int num_populations,
    const int num_jumpers)
{
    /* Moves you assuming periodic BC's. */
    //Input should be a 2d workgroup!
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if ((x < nx) && (y < ny)){
        for(int jump_id = 0; jump_id < num_jumpers; jump_id++){
            int cur_cy = cy[jump_id];

            int four_d_index = jump_id*num_populations*nx*ny + cur_field*nx*ny + y*nx + x;

            f_global[four_d_index] = f_streamed_global[four_d_index];
        }
    }
}

__kernel void
add_constant_g_force(
    const int field_num,
    const ${num_type} g_x,
    const ${num_type} g_y,
    __global ${num_type} *Gx_global,
    __global ${num_type} *Gy_global,
    __global ${num_type} *rho_global,
    const int nx, const int ny
)
{
    //Input should be a 2d workgroup! Loop over the third dimension.
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if ((x < nx) && (y < ny)){
        int three_d_index = field_num*nx*ny + y*nx + x;

        ${num_type} rho = rho_global[three_d_index];

        // Remember, force PER density! In *dimensionless* units.
        Gx_global[three_d_index] += g_x*rho;
        Gy_global[three_d_index] += g_y*rho;

    }
}

__kernel void
add_boussinesq_force(
    const int flow_field_num,
    const int solute_field_num,
    const ${num_type} rho_cutoff,
    const ${num_type} solute_ref_density,
    const ${num_type} g_x,
    const ${num_type} g_y,
    __global ${num_type} *Gx_global,
    __global ${num_type} *Gy_global,
    __global ${num_type} *rho_global,
    const int nx, const int ny
)
{
    //Input should be a 2d workgroup! Loop over the third dimension.
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if ((x < nx) && (y < ny)){
        int flow_three_d_index = flow_field_num*nx*ny + y*nx + x;
        int solute_three_d_index = solute_field_num*nx*ny + y*nx + x;

        ${num_type} rho_flow = rho_global[flow_three_d_index];
        ${num_type} rho_solute = rho_global[solute_three_d_index];
        ${num_type} delta_rho = rho_solute - solute_ref_density;

        ${num_type} force_x, force_y;

        if(rho_flow < rho_cutoff){ // Not in the main fluid anymore
            force_x = 0;
            force_y = 0;
        }
        else{
            force_x = g_x*delta_rho;
            force_y = g_y*delta_rho;
        }

        Gx_global[flow_three_d_index] += force_x;
        Gy_global[flow_three_d_index] += force_y;
    }
}

__kernel void
add_buoyancy_difference(
    const int flow_field_num,
    const ${num_type} rho_cutoff,
    const ${num_type} rho_ref,
    const ${num_type} g_x,
    const ${num_type} g_y,
    __global ${num_type} *Gx_global,
    __global ${num_type} *Gy_global,
    __global ${num_type} *rho_global,
    const int nx, const int ny
)
{
    //Input should be a 2d workgroup! Loop over the third dimension.
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if ((x < nx) && (y < ny)){
        int flow_three_d_index = flow_field_num*nx*ny + y*nx + x;
        ${num_type} rho_flow = rho_global[flow_three_d_index];
        ${num_type} rho_diff = rho_flow - rho_ref;

        ${num_type} force_x, force_y;

        if (rho_flow < rho_cutoff){ // Not in the fluid anymore
            force_x = 0;
            force_y = 0;
        }
        else{
            force_x = g_x*rho_diff;
            force_y = g_y*rho_diff;
        }

        //TODO: This is currently nonsense. lol. Be careful!
        Gx_global[flow_three_d_index] += force_x;
        Gy_global[flow_three_d_index] += force_y;
    }
}

__kernel void
add_radial_g_force(
    const int field_num,
    const int center_x,
    const int center_y,
    const ${num_type} prefactor,
    const ${num_type} radial_scaling,
    __global ${num_type} *Gx_global,
    __global ${num_type} *Gy_global,
    __global ${num_type} *rho_global,
    const int nx, const int ny
)
{
    //Input should be a 2d workgroup! Loop over the third dimension.
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if ((x < nx) && (y < ny)){
        int three_d_index = field_num*nx*ny + y*nx + x;

        ${num_type} rho = rho_global[three_d_index];
        // Get the current radius and angle

        const ${num_type} dx = x - center_x;
        const ${num_type} dy = y - center_y;

        const ${num_type} radius_dim = sqrt(dx*dx + dy*dy);
        const ${num_type} theta = atan2(dy, dx);

        // Get the unit vector
        const ${num_type} rhat_x = cos(theta);
        const ${num_type} rhat_y = sin(theta);

        // Get the gravitational acceleration
        ${num_type} magnitude = prefactor*((${num_type})pow(radius_dim, radial_scaling));
        Gx_global[three_d_index] += rho*magnitude*rhat_x;
        Gy_global[three_d_index] += rho*magnitude*rhat_y;
    }
}

void get_psi(
    const int PSI_SPECIFIER,
    ${num_type} rho_1, ${num_type} rho_2,
    ${num_type} *psi_1, ${num_type} *psi_2,
    __constant ${num_type} *parameters)
{
    //TODO: DO WE NEED ZERO CHECKING?
    if(rho_1 < 0) rho_1 = 0;
    if(rho_2 < 0) rho_2 = 0;

    if(PSI_SPECIFIER == 0){ // rho_1 * rho_2
        *psi_1 = rho_1;
        *psi_2 = rho_2;
    }
    if(PSI_SPECIFIER == 1){ // shan-chen
        ${num_type} rho_0 = parameters[0];
        *psi_1 = rho_0*(1 - exp(-rho_1/rho_0));
        *psi_2 = rho_0*(1 - exp(-rho_2/rho_0));
    }
    if(PSI_SPECIFIER == 2){ // pow(rho_1, alpha) * pow(rho_2, alpha)
        *psi_1 = (${num_type})pow(rho_1, parameters[0]);
        *psi_2 = (${num_type})pow(rho_2, parameters[0]);
    }
    if(PSI_SPECIFIER==3){ //van-der-waals; G MUST BE SET TO ONE TO USE THIS
        ${num_type} a = parameters[0];
        ${num_type} b = parameters[1];
        ${num_type} T = parameters[2];
        ${num_type} cs = parameters[3];

        ${num_type} P1 = (rho_1*T)/(1 - rho_1*b) - a*rho_1*rho_1;
        ${num_type} P2 = (rho_2*T)/(1 - rho_2*b) - a*rho_2*rho_2;

        *psi_1 = sqrt(2*(P1 - cs*cs*rho_1)/(cs*cs));
        *psi_2 = sqrt(2*(P2 - cs*cs*rho_2)/(cs*cs));
    }
}

void get_BC(
    int *streamed_x,
    int *streamed_y,
    const int BC_SPECIFIER,
    const int nx,
    const int ny)
{
    if (BC_SPECIFIER == 0){ //PERIODIC
        if (*streamed_x >= nx) *streamed_x -= nx;
        if (*streamed_x < 0) *streamed_x += nx;

        if (*streamed_y >= ny) *streamed_y -= ny;
        if (*streamed_y < 0) *streamed_y += ny;
    }
    if (BC_SPECIFIER == 1){ // ZERO GRADIENT
        if (*streamed_x >= nx) *streamed_x = nx - 1;
        if (*streamed_x < 0) *streamed_x = 0;

        if (*streamed_y >= ny) *streamed_y = ny - 1;
        if (*streamed_y < 0) *streamed_y = 0;
    }
    if (BC_SPECIFIER == 2){ // No density on walls...TODO: There is a better way to handle adhesion to walls...
        if (*streamed_x >= nx) *streamed_x = -1;
        if (*streamed_x < 0) *streamed_x = -1;

        if (*streamed_y >= ny) *streamed_y = -1;
        if (*streamed_y < 0) *streamed_y = -1;
    }
}
__kernel void
add_interaction_force(
    const int fluid_index_1,
    const int fluid_index_2,
    const ${num_type} G_int,
    __local ${num_type} *local_fluid_1,
    __local ${num_type} *local_fluid_2,
    __global __read_only ${num_type} *rho_global,
    __global ${num_type} *Gx_global,
    __global ${num_type} *Gy_global,
    const ${num_type} cs,
    __constant int *cx,
    __constant int *cy,
    __constant ${num_type} *w,
    const int nx, const int ny,
    const int buf_nx, const int buf_ny,
    const int halo,
    const int num_jumpers,
    const int BC_SPECIFIER,
    const int PSI_SPECIFIER,
    __constant ${num_type} *parameters,
    const ${num_type} rho_wall)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    // Have to use local memory where you read in everything around you in the workgroup.
    // Otherwise, you are actually doing 9x the work of what you have to...painful.

    // Local position relative to (0, 0) in workgroup
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    // coordinates of the upper left corner of the buffer in image
    // space, including halo
    const int buf_corner_x = x - lx - halo;
    const int buf_corner_y = y - ly - halo;

    // coordinates of our pixel in the local buffer
    const int buf_x = lx + halo;
    const int buf_y = ly + halo;

    // 1D index of thread within our work-group
    const int idx_1D = ly * get_local_size(0) + lx;

    barrier(CLK_LOCAL_MEM_FENCE);
    if (idx_1D < buf_nx) {
        for (int row = 0; row < buf_ny; row++) {
            int temp_x = buf_corner_x + idx_1D;
            int temp_y = buf_corner_y + row;

            //Painfully deal with BC's...i.e. use periodic BC's.
            get_BC(&temp_x, &temp_y, BC_SPECIFIER, nx, ny);

            ${num_type} rho_to_use_1 = rho_wall;
            ${num_type} rho_to_use_2 = rho_wall;

            if((temp_x != -1) && (temp_y != -1)){
                rho_to_use_1 = rho_global[fluid_index_1*ny*nx + temp_y*nx + temp_x];
                rho_to_use_2 = rho_global[fluid_index_2*ny*nx + temp_y*nx + temp_x];
            }

            local_fluid_1[row*buf_nx + idx_1D] = rho_to_use_1;
            local_fluid_2[row*buf_nx + idx_1D] = rho_to_use_2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //Now that all desired rhos are read in, do the multiplication
    if ((x < nx) && (y < ny)){

        ${num_type} force_x_fluid_1 = 0;
        ${num_type} force_y_fluid_1 = 0;

        ${num_type} force_x_fluid_2 = 0;
        ${num_type} force_y_fluid_2 = 0;

        // Get the psi at the current pixel
        const int old_2d_buf_index = buf_y*buf_nx + buf_x;

        ${num_type} rho_1_pixel = local_fluid_1[old_2d_buf_index];
        ${num_type} rho_2_pixel = local_fluid_2[old_2d_buf_index];

        ${num_type} psi_1_pixel = 0;
        ${num_type} psi_2_pixel = 0;

        get_psi(PSI_SPECIFIER, rho_1_pixel, rho_2_pixel, &psi_1_pixel, &psi_2_pixel, parameters);

        ${num_type} psi_1 = 0; // The psi that correspond to jumping around the lattice
        ${num_type} psi_2 = 0;

        for(int jump_id = 0; jump_id < num_jumpers; jump_id++){
            int cur_cx = cx[jump_id];
            int cur_cy = cy[jump_id];
            ${num_type} cur_w = w[jump_id];

            //Get the shifted positions
            int stream_buf_x = buf_x + cur_cx;
            int stream_buf_y = buf_y + cur_cy;

            int new_2d_buf_index = stream_buf_y*buf_nx + stream_buf_x;

            ${num_type} cur_rho_1 = local_fluid_1[new_2d_buf_index];
            ${num_type} cur_rho_2 = local_fluid_2[new_2d_buf_index];

            get_psi(PSI_SPECIFIER, cur_rho_1, cur_rho_2, &psi_1, &psi_2, parameters);

            force_x_fluid_1 += cur_w * cur_cx * psi_2;
            force_y_fluid_1 += cur_w * cur_cy * psi_2;

            force_x_fluid_2 += cur_w * cur_cx * psi_1;
            force_y_fluid_2 += cur_w * cur_cy * psi_1;
        }

        force_x_fluid_1 *= -(G_int*psi_1_pixel);
        force_y_fluid_1 *= -(G_int*psi_1_pixel);

        force_x_fluid_2 *= -(G_int*psi_2_pixel);
        force_y_fluid_2 *= -(G_int*psi_2_pixel);

        const int two_d_index = y*nx + x;
        int three_d_index_fluid_1 = fluid_index_1*ny*nx + two_d_index;
        int three_d_index_fluid_2 = fluid_index_2*ny*nx + two_d_index;

        // We need to move from *force* to force/density!
        // If rho is zero, force should be zero! That's what the books say.
        // So, just don't increment the force is rho is too small; equivalent to setting force = 0.
        Gx_global[three_d_index_fluid_1] += force_x_fluid_1;
        Gy_global[three_d_index_fluid_1] += force_y_fluid_1;

        Gx_global[three_d_index_fluid_2] += force_x_fluid_2;
        Gy_global[three_d_index_fluid_2] += force_y_fluid_2;
    }
}

__kernel void
add_interaction_force_second_belt(
    const int fluid_index_1,
    const int fluid_index_2,
    const ${num_type} G_int,
    __local ${num_type} *local_fluid_1,
    __local ${num_type} *local_fluid_2,
    __global __read_only ${num_type} *rho_global,
    __global ${num_type} *Gx_global,
    __global ${num_type} *Gy_global,
    const ${num_type} cs,
    __constant ${num_type} *pi1,
    __constant int *cx1,
    __constant int *cy1,
    const int num_jumpers_1,
    __constant ${num_type} *pi2,
    __constant int *cx2,
    __constant int *cy2,
    const int num_jumpers_2,
    const int nx, const int ny,
    const int buf_nx, const int buf_ny,
    const int halo,
    const int BC_SPECIFIER,
    const int PSI_SPECIFIER,
    __constant ${num_type} *parameters,
    const ${num_type} rho_wall)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    // Have to use local memory where you read in everything around you in the workgroup.
    // Otherwise, you are actually doing 9x the work of what you have to...painful.

    // Local position relative to (0, 0) in workgroup
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    // coordinates of the upper left corner of the buffer in image
    // space, including halo
    const int buf_corner_x = x - lx - halo;
    const int buf_corner_y = y - ly - halo;

    // coordinates of our pixel in the local buffer
    const int buf_x = lx + halo;
    const int buf_y = ly + halo;

    // 1D index of thread within our work-group
    const int idx_1D = ly * get_local_size(0) + lx;

    barrier(CLK_LOCAL_MEM_FENCE);
    if (idx_1D < buf_nx) {
        for (int row = 0; row < buf_ny; row++) {
            int temp_x = buf_corner_x + idx_1D;
            int temp_y = buf_corner_y + row;

            //Painfully deal with BC's...i.e. use periodic BC's.
            get_BC(&temp_x, &temp_y, BC_SPECIFIER, nx, ny);

            ${num_type} rho_to_use_1 = rho_wall;
            ${num_type} rho_to_use_2 = rho_wall;

            if((temp_x != -1) && (temp_y != -1)){
                rho_to_use_1 = rho_global[fluid_index_1*ny*nx + temp_y*nx + temp_x];
                rho_to_use_2 = rho_global[fluid_index_2*ny*nx + temp_y*nx + temp_x];
            }

            local_fluid_1[row*buf_nx + idx_1D] = rho_to_use_1;
            local_fluid_2[row*buf_nx + idx_1D] = rho_to_use_2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //Now that all desired rhos are read in, do the multiplication
    if ((x < nx) && (y < ny)){

        //Remember, this is force PER DENSITY to avoid problems
        ${num_type} force_x_fluid_1 = 0;
        ${num_type} force_y_fluid_1 = 0;

        ${num_type} force_x_fluid_2 = 0;
        ${num_type} force_y_fluid_2 = 0;

        // Get the psi at the current pixel
        const int old_2d_buf_index = buf_y*buf_nx + buf_x;

        ${num_type} rho_1_pixel = local_fluid_1[old_2d_buf_index];
        ${num_type} rho_2_pixel = local_fluid_2[old_2d_buf_index];

        ${num_type} psi_1_pixel = 0;
        ${num_type} psi_2_pixel = 0;

        get_psi(PSI_SPECIFIER, rho_1_pixel, rho_2_pixel, &psi_1_pixel, &psi_2_pixel, parameters);

        //Psi at other pixels

        ${num_type} psi_1 = 0;
        ${num_type} psi_2 = 0;

        for(int jump_id = 0; jump_id < num_jumpers_1; jump_id++){
            int cur_cx = cx1[jump_id];
            int cur_cy = cy1[jump_id];
            ${num_type} cur_w = pi1[jump_id];

            //Get the shifted positions
            int stream_buf_x = buf_x + cur_cx;
            int stream_buf_y = buf_y + cur_cy;

            int new_2d_buf_index = stream_buf_y*buf_nx + stream_buf_x;

            ${num_type} cur_rho_1 = local_fluid_1[new_2d_buf_index];
            ${num_type} cur_rho_2 = local_fluid_2[new_2d_buf_index];

            get_psi(PSI_SPECIFIER, cur_rho_1, cur_rho_2, &psi_1, &psi_2, parameters);

            force_x_fluid_1 += cur_w * cur_cx * psi_2;
            force_y_fluid_1 += cur_w * cur_cy * psi_2;

            force_x_fluid_2 += cur_w * cur_cx * psi_1;
            force_y_fluid_2 += cur_w * cur_cy * psi_1;
        }

        for(int jump_id = 0; jump_id < num_jumpers_2; jump_id++){
            int cur_cx = cx2[jump_id];
            int cur_cy = cy2[jump_id];
            ${num_type} cur_w = pi2[jump_id];

            //Get the shifted positions
            int stream_buf_x = buf_x + cur_cx;
            int stream_buf_y = buf_y + cur_cy;

            int new_2d_buf_index = stream_buf_y*buf_nx + stream_buf_x;

            ${num_type} cur_rho_1 = local_fluid_1[new_2d_buf_index];
            ${num_type} cur_rho_2 = local_fluid_2[new_2d_buf_index];

            get_psi(PSI_SPECIFIER, cur_rho_1, cur_rho_2, &psi_1, &psi_2, parameters);

            force_x_fluid_1 += cur_w * cur_cx * psi_2;
            force_y_fluid_1 += cur_w * cur_cy * psi_2;

            force_x_fluid_2 += cur_w * cur_cx * psi_1;
            force_y_fluid_2 += cur_w * cur_cy * psi_1;
        }

        force_x_fluid_1 *= -(G_int*psi_1_pixel);
        force_y_fluid_1 *= -(G_int*psi_1_pixel);

        force_x_fluid_2 *= -(G_int*psi_2_pixel);
        force_y_fluid_2 *= -(G_int*psi_2_pixel);

        const int two_d_index = y*nx + x;
        int three_d_index_fluid_1 = fluid_index_1*ny*nx + two_d_index;
        int three_d_index_fluid_2 = fluid_index_2*ny*nx + two_d_index;

        // We need to move from *force* to force/density!
        // If rho is zero, force should be zero! That's what the books say.
        // So, just don't increment the force is rho is too small; equivalent to setting force = 0.
        Gx_global[three_d_index_fluid_1] += force_x_fluid_1;
        Gy_global[three_d_index_fluid_1] += force_y_fluid_1;

        Gx_global[three_d_index_fluid_2] += force_x_fluid_2;
        Gy_global[three_d_index_fluid_2] += force_y_fluid_2;
    }
}
