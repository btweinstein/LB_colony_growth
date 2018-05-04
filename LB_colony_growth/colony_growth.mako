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

// Spatial location of the thread within the buffer
% if dimension == 2:
const int local_index = get_spatial_index(buf_x, buf_y, buf_nx, buf_ny);
% elif dimension == 3:
const int local_index = get_spatial_index(buf_x, buf_y, buf_z, buf_nx, buf_ny, buf_nz);
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
const int jump_index = get_spatial_index(x, y, ${jump_id}, nx, ny, num_jumpers);
% elif dimension == 3:
const int jump_index = get_spatial_index(x, y, z, ${jump_id}, nx, ny, nz, num_jumpers);
% endif

</%def>


<%def name='define_all_c(jump_id="jump_id")' buffered='True' filter='trim'>

const int cur_cx = c_vec[get_spatial_index(0, ${jump_id}, ${dimension}, num_jumpers)];
const int cur_cy = c_vec[get_spatial_index(1, ${jump_id}, ${dimension}, num_jumpers)];
%if dimension == 3:
const int cur_cz = c_vec[get_spatial_index(2, jump_id, ${dimension}, num_jumpers)];

%endif
</%def>


<%def name='define_streamed_index_local()' buffered='True' filter='trim'>

% if dimension == 2:
int streamed_index_local = get_spatial_index(buf_x + cur_cx, buf_y + cur_cy, buf_nx, buf_ny);
% elif dimension == 3:
int streamed_index_local = get_spatial_index(
    buf_x + cur_cx, buf_y + cur_cy, buf_z + cur_cz,
    buf_nx, buf_ny, buf_nz
);
% endif

</%def>

<%def name='define_streamed_index_global(identifier="int")' buffered='True' filter='trim'>

% if dimension == 2:
${identifier} streamed_index_global = get_spatial_index(x + cur_cx, y + cur_cy, nx, ny);
% elif dimension == 3:
${identifier} streamed_index_global = get_spatial_index(
    x + cur_cx, y + cur_cy, z + cur_cz,
    nx, ny, nz
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

        const int node_type = bc_map_local[local_index];
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
    ${define_streamed_index_local()}

    const int streamed_bc = bc_map_local[streamed_index_local];

    if (streamed_bc == FLUID_NODE){ // Scalar can diffuse in
        // Determine Cwall via finite difference
        const ${num_type} cur_rho = rho_local[local_index];

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

${define_all_c()}

${define_streamed_index_local()}

const int streamed_bc = bc_map_local[streamed_index_local];

int streamed_index_global = -1; // Initialize to a nonsense value

if (streamed_bc == FLUID_NODE){
    // Propagate the collided particle distribution as appropriate
    ## As streamed_index_global is already initialized, no identifer is needed
    ${define_streamed_index_global(identifier='') | wrap1}
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
    streamed_index_global = get_spatial_index(x, y, z, jump_id, nx, ny, nz, num_jumpers);
}

else if (streamed_bc < 0){ // You are at a population node
    // Determine Cwall via finite difference

    ${num_type} cur_rho = rho_local[local_index];
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
    streamed_index_global = get_spatial_index(x, y, z, jump_id, nx, ny, nz, num_jumpers);
}

//Need to write to the streamed buffer...otherwise out of sync problems will occur
f_streamed_global[streamed_index_global] = f_after_collision;
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

    const ${num_type} cur_w = w[jump_id];

    const ${num_type} new_feq = cur_w*new_rho;

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
        const int node_type = bc_map_local[local_index];

        if (node_type < 0){ // Population node!
            //Check if you have accumulated enough mass
            ${num_type} current_mass = absorbed_mass_global[spatial_index];
            if (current_mass > m_reproduce){
                ${reproduce() | wrap4}
            }
        }
    }
}

<%def name='reproduce()' buffered='True' filter='trim'>

// Calculate the normalization constant
${num_type} norm_constant = 0;

for(int jump_id=0; jump_id < num_jumpers; jump_id++){
    ${define_all_c() | wrap1}

    ${define_streamed_index_local() | wrap1}

    const int streamed_node_type = bc_map_local[streamed_index_local];
    bool can_reproduce = false;

    if (streamed_node_type == FLUID_NODE){ // Population can expand into this!
        norm_constant += w[jump_id];
        can_reproduce = true;
        can_reproduce_global[0] = 1; // Flag that indicates if someone, somewhere *can* reproduce
    }
}

if (can_reproduce){
    const ${num_type} rand_num = rand_global[spatial_index];

    ${num_type} prob_total = 0;

    for(int jump_id=0; jump_id < num_jumpers; jump_id++){
        ${define_all_c() | wrap2}
        ${define_streamed_index_local() | wrap2}

        const int streamed_node_type = bc_map_local[streamed_index_local];

        if (streamed_node_type == FLUID_NODE){ // Population can expand into this!
            prob_total += w[jump_id]/norm_constant;
            if (prob_total > rand_num){
                // Propagate to that spot, atomically!
                ${define_streamed_index_global() | wrap4}

                //TODO: This can probably be sped up by doing comparisons with local, *then* going to global...
                // Copy your node type into the new node atomically IF the fluid node is still there...
                const int prev_type = atomic_cmpxchg(
                    &reproduce_bc_map_global[streamed_index_global],
                    FLUID_NODE,
                    node_type
                );

                // If successful, subtract mass, because you reproduced!
                if (prev_type == FLUID_NODE){
                    mass_absorbed_global[spatial_index] -= m0;
                } // Otherwise, someone reproduced and blocked you! Try again next iteration...
            }
        }
    }
}

</%def>
