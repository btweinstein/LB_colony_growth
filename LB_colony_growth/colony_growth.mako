<%!
    from LB_colony_growth.filters import wrap1, wrap2, wrap3, wrap4
%>

<%namespace file='util.mako' import='*' name='util' />
<%namespace file='kernel.mako' import='*' name='kernel' />

${enable_double_support()}

// Define domain size
#define nx ${nx}
#define ny ${ny}
% if dimension == 3:
#define nz ${nz}
% endif

${define_node_types()}

// Everything works as long as halo is one...really should be velocity set dependent.
# define halo 1

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

//######### Collide & Propagate kernel ########

${set_current_kernel('collide_and_propagate')}

## Global variables
${needs_bc_map()}
${needs_f()}
${needs_f_streamed()}
${needs_feq()}
${needs_rho()}
${needs_absorbed_mass()}

## Local variables
${needs_local_mem_num('rho_local')}
${needs_local_mem_int('bc_map_local')}
${needs_local_buf_size()}

## Specific parameter choices
${needs_k_list()}
${needs_D()}

## Lattice velocity choices
${needs_num_jumpers()}
${needs_omega()}
${needs_c_vec()}
${needs_c_mag()}
${needs_w()}
${needs_reflect_list()}

__kernel void
collide_and_propagate(
${print_kernel_args()}
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
        // Alleles are negative...need to convert to an index
        //node_type: the allele present in this cell

        const int allele_index = -1*node_type - 1;

        const ${num_type} cur_k = k[allele_index];

        // Determine Cwall via finite difference
        const ${num_type} neighbor_rho = rho_local[streamed_index_local];

        const ${num_type} cur_c_mag = 1.0; // Magnitude to nearest neighbors is always one
        const ${num_type} rho_wall = neighbor_rho/(1 + (cur_k*cur_c_mag)/(2*D));

        //Update the mass at the site accordingly
        // Flux in is k*rho_wall...and in lattice units, all additional factors are one.
        mass_to_add += cur_k*rho_wall;
    }
}
absorbed_mass_global[spatial_index] += mass_to_add;

</%def>

<%def name='collide_bgk()' buffered='True' filter='trim'>
${num_type} f_after_collision = f_global[jump_index]*(1-omega) + omega*feq_global[jump_index];
//TODO: If a source is needed, additional terms are needed here.
</%def>

<%def name='move()' buffered='True' filter='trim'>
// After colliding, stream to the appropriate location.
${define_all_c()}

${define_streamed_index_local()}

const int streamed_bc = bc_map_local[streamed_index_local];

int streamed_index_global = -1; // Initialize to a nonsense value
${num_type} new_f = f_after_collision;

if (streamed_bc == FLUID_NODE){
    // Propagate the collided particle distribution as appropriate
    % if dimension == 2:
    streamed_index_global = ${get_spatial_index(
        '(x + cur_cx)', '(y + cur_cy)', 'jump_id',
        'nx', 'ny', 'num_jumpers'
    )};
    % elif dimension == 3:
    streamed_index_global = ${get_spatial_index(
        '(x + cur_cx)', '(y + cur_cy)', '(z+cur_cz)', 'reflect_id',
        'nx', 'ny', 'nz', 'num_jumpers')};
    % endif
}
else if (streamed_bc == WALL_NODE){ // Zero concentration on the wall; bounceback.
    const int reflect_id = reflect_list[jump_id];
    % if dimension == 2:
    const int reflect_index = spatial_index + nx*ny*reflect_id;
    % elif dimension == 3:
    const int reflect_index = spatial_index + nx*ny*nz*reflect_id;
    % endif

    streamed_index_global = reflect_index;
}
else if (streamed_bc < 0){ // You are at a population node
    // Determine Cwall via finite difference...

    // Alleles are negative...need to convert to an index
    const int allele_index = -1*streamed_bc - 1;

    const ${num_type} cur_k = k[allele_index];

    ${num_type} cur_rho = rho_local[local_index];
    ${num_type} cur_c_mag = c_mag[jump_id];
    ${num_type} rho_wall = cur_rho/(1 + (cur_k*cur_c_mag)/(2*D));

    // Based on rho_wall, do the bounceback
    ${num_type} cur_w = w[jump_id];
    int reflect_id = reflect_list[jump_id];
    % if dimension == 2:
    int reflect_index = spatial_index + nx*ny*reflect_id;
    % elif dimension == 3:
    int reflect_index = spatial_index + nx*ny*nz*reflect_id;
    % endif

    streamed_index_global = reflect_index;
    new_f = -f_after_collision + 2*cur_w*rho_wall;
}
else if (streamed_bc == NOT_IN_DOMAIN){
    printf("There is something wrong with the BC-map! I'm streaming to a not-defined region...");
}

//Need to write to the streamed buffer...otherwise out of sync problems will occur

f_streamed_global[streamed_index_global] = new_f;

</%def>

//######### Update after streaming kernel #########
<%
    cur_kernel = 'update_after_streaming'
    kernel_arguments[cur_kernel] = []
    cur_kernel_list = kernel_arguments[cur_kernel]
%>

${needs_bc_map(cur_kernel_list)}
    cur_kernel_list.append(['bc_map', '__global __read_only int *bc_map_global'])
    cur_kernel_list.append(['nx_bc', 'const int nx_bc'])
    cur_kernel_list.append(['ny_bc', 'const int ny_bc'])
    if dimension == 3:
        cur_kernel_list.append(['nz_bc', 'const int nz_bc'])

    cur_kernel_list.append(['f', '__global '+num_type+' *f_global'])
    cur_kernel_list.append(['feq', '__global '+num_type+' *feq_global'])
    cur_kernel_list.append(['rho', '__global '+num_type+' *rho_global'])

    # Velocity set info
    cur_kernel_list.append(['w', '__constant '+num_type+' *w'])
    cur_kernel_list.append(['num_jumpers', 'const int num_jumpers'])


__kernel void
update_after_streaming(
${print_kernel_args(cur_kernel_list)}
)
{
    // Get info about where thread is located in global memory
    ${define_thread_location() | wrap1}

    // Main loop...
    ${if_thread_in_domain() | wrap1}{
        // Figure out what type of node is present
        ${read_node_type_from_global() | wrap2}

        ${update_hydro() | wrap2}
        ${update_feq() | wrap2}
    }
}

<%def name='update_hydro()' buffered='True' filter='trim'>
// Update rho!
${num_type} new_rho = 0;
for(int jump_id=0; jump_id < num_jumpers; jump_id++){
    ${define_jump_index() | wrap1}

    new_rho += f_global[jump_index];
}

if (node_type != FLUID_NODE) new_rho = 0; // No density if not in the fluid
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

//######### Update feq for initialization #####

<%
    cur_kernel = 'init_feq'
    kernel_arguments[cur_kernel] = []
    cur_kernel_list = kernel_arguments[cur_kernel]

    cur_kernel_list.append(['feq', '__global __read_only '+num_type+' *feq_global'])
    cur_kernel_list.append(['rho', '__global '+num_type+' *rho_global'])

    # Velocity set info
    cur_kernel_list.append(['w', '__constant '+num_type+' *w'])
    cur_kernel_list.append(['num_jumpers', 'const int num_jumpers'])
%>

__kernel void
init_feq(
${print_kernel_args(cur_kernel_list)}
)
{
    // Get info about where thread is located in global memory
    ${define_thread_location() | wrap1}

    // Main loop...
    ${if_thread_in_domain() | wrap1}{
        // Figure out what type of node is present

        ${num_type} new_rho = rho_global[spatial_index];

        ${update_feq() | wrap2}
    }
}


//######### Reproduce cells kernel #########
<%
    cur_kernel = 'reproduce'
    kernel_arguments[cur_kernel] = []
    cur_kernel_list = kernel_arguments[cur_kernel]

    cur_kernel_list.append(['bc_map', '__global __read_only int *bc_map_global'])
    cur_kernel_list.append(['bc_map_streamed', '__global int *bc_map_streamed_global'])
    cur_kernel_list.append(['nx_bc', 'const int nx_bc'])
    cur_kernel_list.append(['ny_bc', 'const int ny_bc'])
    if dimension == 3:
        cur_kernel_list.append(['nz_bc', 'const int nz_bc'])

    cur_kernel_list.append(['absorbed_mass', '__global '+num_type+' *absorbed_mass_global'])
    cur_kernel_list.append(['rand', '__global '+num_type+' *rand_global'])

    # Pointer that determines whether everyone is done reproducing
    cur_kernel_list.append(['can_reproduce_pointer', '__global int *can_reproduce_global'])

    # Input parameters
    cur_kernel_list.append(['m_reproduce_list', '__constant '+num_type+' *m_reproduce'])

    # Velocity set info
    cur_kernel_list.append(['w', '__constant '+num_type+' *w'])
    cur_kernel_list.append(['num_jumpers', 'const int num_jumpers'])
    cur_kernel_list.append(['c_vec', '__constant int *c_vec'])

    # Local memory info
    cur_kernel_list.append(['local_mem_int', '__local int *bc_map_local'])

    cur_kernel_list.append(['buf_nx', 'const int buf_nx'])
    cur_kernel_list.append(['buf_ny', 'const int buf_ny'])
    if dimension == 3:
        cur_kernel_list.append(['buf_nz', 'const int buf_nz'])
%>

__kernel void
reproduce(
${print_kernel_args(cur_kernel_list)}
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

            // Alleles are negative...need to convert to an index
            const int allele_index = -1*node_type - 1;

            const ${num_type} cur_m_reproduce = m_reproduce[allele_index];

            ${num_type} current_mass = absorbed_mass_global[spatial_index];
            if (current_mass > cur_m_reproduce){
                ${reproduce() | wrap4}
            }
        }
    }
}

<%def name='reproduce()' buffered='True' filter='trim'>

// Calculate the normalization constant
${num_type} norm_constant = 0;

bool space_to_reproduce = false;

//Determine if you can reproduce...need enough space, mass is already checked.
for(int jump_id=0; jump_id < num_jumpers; jump_id++){
    ${define_all_c() | wrap1}

    ${define_streamed_index_local() | wrap1}

    const int streamed_node_type = bc_map_local[streamed_index_local];

    if (streamed_node_type == FLUID_NODE){ // Population can expand into this!
        norm_constant += w[jump_id];
        space_to_reproduce = true;
        can_reproduce_global[0] = 1; // Flag that indicates if someone, somewhere *can* reproduce
    }
}

if (space_to_reproduce){

    const ${num_type} rand_num = rand_global[spatial_index];

    bool has_chosen_direction = false;

    int cur_cx = 0;
    int cur_cy = 0;
    %if dimension == 3:
    int cur_cz = 0;
    %endif

    int jump_id = -1;

    ${num_type} prob_total = 0;
    while((jump_id < num_jumpers) && (!has_chosen_direction)){
        jump_id += 1;

        ## Use the c's defined outside the loop
        ${define_all_c(identifier='') | wrap2}
        ${define_streamed_index_local() | wrap2}

        const int streamed_node_type = bc_map_local[streamed_index_local];
        if (streamed_node_type == FLUID_NODE){ // Population can expand into this!
            prob_total += w[jump_id]/norm_constant;
            if (prob_total > rand_num){
                has_chosen_direction = true;
            }
        }
    }
    // The final jump_id corresponds to the direction to jump!
    // Same with the current jump directions

    % if dimension == 2:
    const int streamed_global_bc_index = ${get_spatial_index(
        '(x + cur_cx + halo)', '(y + cur_cy + halo)',
        'nx_bc', 'ny_bc'
    )};
    % elif dimension == 3:
    const int streamed_global_bc_index = ${get_spatial_index(
        '(x + cur_cx + halo)', '(y + cur_cy + halo)', '(z + cur_cz + halo)',
        'nx_bc', 'ny_bc', 'nz_bc'
    )};
    % endif

    //TODO: This can probably be sped up by doing comparisons with local, *then* going to global...
    // Copy your node type into the new node atomically IF the fluid node is still there...
    const int prev_type = atomic_cmpxchg(
        &bc_map_streamed_global[streamed_global_bc_index],
        FLUID_NODE,
        node_type
    );

    // If successful, subtract mass, because you reproduced!
    if (prev_type == FLUID_NODE){
        absorbed_mass_global[spatial_index] -= cur_m_reproduce;
    } // Otherwise, someone reproduced and blocked you! Try again next iteration...
}

</%def>

//######### Copy kernels #########
<%
    cur_kernel = 'copy_streamed_onto_f'
    kernel_arguments[cur_kernel] = []
    cur_kernel_list = kernel_arguments[cur_kernel]

    # Needed global variables
    cur_kernel_list.append(['f_streamed', '__global '+num_type+' *f_streamed_global'])
    cur_kernel_list.append(['f', '__global '+num_type+' *f_global'])

    # Velocity set info
    cur_kernel_list.append(['num_jumpers', 'const int num_jumpers'])
%>

__kernel void
copy_streamed_onto_f(
${print_kernel_args(cur_kernel_list)}
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

<%
    cur_kernel = 'copy_streamed_onto_bc'
    kernel_arguments[cur_kernel] = []
    cur_kernel_list = kernel_arguments[cur_kernel]

    # Needed global variables
    cur_kernel_list.append(['bc_map', '__global int *bc_map_global'])
    cur_kernel_list.append(['bc_map_streamed', '__global int *bc_map_streamed_global'])

    cur_kernel_list.append(['nx_bc', 'const int nx_bc'])
    cur_kernel_list.append(['ny_bc', 'const int ny_bc'])
    if dimension == 3:
        cur_kernel_list.append(['nz_bc', 'const int nz_bc'])
%>

__kernel void
copy_streamed_onto_bc(
${print_kernel_args(cur_kernel_list)}
)
{
    ${define_thread_location() | wrap1}

    ${if_thread_in_bc_domain() | wrap1}{
        bc_map_global[spatial_index] = bc_map_streamed_global[spatial_index];
    }
}