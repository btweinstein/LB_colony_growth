import numpy as np
import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
import pyopencl as cl
import pyopencl.tools
import pyopencl.clrandom
import pyopencl.array
import ctypes as ct
import matplotlib.pyplot as plt

import mako as m
import mako.template as mte
import mako.runtime as mrt
import StringIO as sio
import weakref

# Required to draw obstacles
import skimage as ski
import skimage.draw

# Get path to *this* file. Necessary when reading in opencl code.
full_path = os.path.realpath(__file__)
file_dir = os.path.dirname(full_path)
parent_dir = os.path.dirname(file_dir)

# Required for allocating local memory
num_size = ct.sizeof(ct.c_double)
num_type = np.double
int_type = np.int32

# Constants for defining the node map...
FLUID_NODE = int_type(0)
WALL_NODE = int_type(1)
NOT_IN_DOMAIN = int_type(2)

def get_divisible_global(global_size, local_size):
    """
    Given a desired global size and a specified local size, return the smallest global
    size that the local size fits into. Required when specifying arbitrary local
    workgroup sizes.

    :param global_size: A tuple of the global size, i.e. (x, y, z)
    :param local_size:  A tuple of the local size, i.e. (lx, ly, lz)
    :return: The smallest global size that the local size fits into.
    """
    new_size = []
    for cur_global, cur_local in zip(global_size, local_size):
        remainder = cur_global % cur_local
        if remainder == 0:
            new_size.append(cur_global)
        else:
            new_size.append(cur_global + cur_local - remainder)
    return tuple(new_size)

class Fluid(object):

    def __init__(self, sim, field_index, nu = 1.0, bc='periodic'):

        self.sim = sim # TODO: MAKE THIS A WEAKREF

        self.field_index = int_type(field_index)

        self.lb_nu_e = num_type(nu)
        self.bc = bc

        # Determine the viscosity
        self.tau = num_type(.5 + self.lb_nu_e / (sim.cs**2))
        print 'tau', self.tau
        self.omega = num_type(self.tau ** -1.)  # The relaxation time of the jumpers in the simulation
        print 'omega', self.omega
        assert self.omega < 2.


    def update_forces(self):
        """For internal forces...none in this case."""

        pass

    def update_feq(self):
        """
        Based on the hydrodynamic fields, create the local equilibrium feq that the jumpers f will relax to.
        Implemented in OpenCL.
        """

        sim = self.sim

        self.sim.kernels.update_feq_fluid(
            sim.queue, sim.two_d_global_size, sim.two_d_local_size,
            sim.feq.data,
            sim.rho.data,
            sim.u_bary.data, sim.v_bary.data,
            sim.w, sim.cx, sim.cy, sim.cs,
            sim.nx, sim.ny,
            self.field_index, sim.num_populations,
            sim.num_jumpers).wait()


    def move(self):
        """
        Move all other jumpers than those on the boundary. Implemented in OpenCL. Consists of two steps:
        streaming f into a new buffer, and then copying that new buffer onto f. We could not think of a way to stream
        in parallel without copying the temporary buffer back onto f.
        """

        sim = self.sim

        self.sim.kernels.move_with_bcs(
            sim.queue, sim.two_d_global_size, sim.two_d_local_size,
            sim.f.data, sim.f_streamed.data,
            sim.cx, sim.cy,
            sim.nx, sim.ny,
            self.field_index, sim.num_populations, sim.num_jumpers,
            sim.bc_map.data, sim.nx_bc, sim.ny_bc, sim.halo_bc,
            sim.reflect_index, sim.slip_x_index, sim.slip_y_index
        ).wait()

        # Copy the streamed buffer into f so that it is correctly updated.
        self.sim.kernels.copy_streamed_onto_f(
            sim.queue, sim.two_d_global_size, sim.two_d_local_size,
            sim.f_streamed.data, sim.f.data,
            sim.cx, sim.cy,
            sim.nx, sim.ny,
            self.field_index, sim.num_populations, sim.num_jumpers).wait()

    def update_hydro(self):

        sim = self.sim

        sim.kernels.update_hydro_fluid(
            sim.queue, sim.two_d_global_size, sim.two_d_local_size,
            sim.f.data,
            sim.rho.data,
            sim.u.data, sim.v.data,
            sim.Gx.data, sim.Gy.data,
            sim.w, sim.cx, sim.cy,
            sim.nx, sim.ny,
            self.field_index, sim.num_populations,
            sim.num_jumpers
        ).wait()

        if sim.check_max_ulb:
            max_ulb = cl.array.max((sim.u[:, :, self.field_index]**2 + sim.v[:, :, self.field_index]**2)**.5, queue=sim.queue)

            if max_ulb > sim.cs*sim.mach_tolerance:
                print 'max_ulb is greater than cs/10! Ma=', max_ulb/sim.cs

    def collide_particles(self):
        sim = self.sim

        self.sim.kernels.collide_particles_fluid(
            sim.queue, sim.two_d_global_size, sim.two_d_local_size,
            sim.f.data,
            sim.feq.data,
            sim.rho.data,
            sim.u_bary.data, sim.v_bary.data,
            sim.Gx.data, sim.Gy.data,
            self.omega,
            sim.w, sim.cx, sim.cy,
            sim.nx, sim.ny,
            self.field_index, sim.num_populations,
            sim.num_jumpers,
            sim.cs
        ).wait()

class Velocity_Set(object):
    def __init__(self, ctx_info, context, kernel_args):
        self.ctx_info = ctx_info
        self.context = context
        self.kernel_args = kernel_args

        self.name = None

        # Variables that will be passed to kernels...

        self.num_jumpers = None
        self.w = None
        self.c_vec = None
        self.c_mag = None
        self.cs = None

        self.reflect_index = None
        self.slip_index = None

        self.halo = None

        self.nx_bc = None
        self.ny_bc = None
        self.nz_bc = None

        self.bc_size = None # Individual to each fluid...b/c each may have a different BC.

        self.buf_nx = None
        self.buf_ny = None
        self.buf_nz = None

    def set_kernel_args(self):

        const_flags = cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR

        self.kernel_args['num_jumpers'] = self.num_jumpers
        self.kernel_args['w'] = cl.Buffer(self.context, const_flags, hostbuf=self.w)
        self.kernel_args['c_vec'] = cl.Buffer(self.context, const_flags, hostbuf=self.c_vec)
        self.kernel_args['c_mag'] = cl.Buffer(self.context, const_flags, hostbuf=self.c_mag)
        self.kernel_args['cs'] = self.cs

        self.kernel_args['reflect_index'] = cl.Buffer(self.context, const_flags, hostbuf=self.reflect_index)
        self.kernel_args['slip_index'] = cl.Buffer(self.context, const_flags, hostbuf=self.slip_index)

        self.kernel_args['halo'] = self.halo

        self.kernel_args['nx_bc'] = self.nx_bc
        self.kernel_args['ny_bc'] = self.ny_bc
        self.kernel_args['nz_bc'] = self.nz_bc

        self.kernel_args['buf_nx'] = self.buf_nx
        self.kernel_args['buf_ny'] = self.buf_ny
        self.kernel_args['buf_nz'] = self.buf_nz

        # Add the ability to create local memory
        self.kernel_args['local_mem_num'] = lambda: self.create_local_memory(self.ctx_info['num_type'])
        self.kernel_args['local_mem_int'] = lambda: self.create_local_memory('int')

    def create_local_memory(self, dtype):

        num_size = None

        if dtype == 'double':
            num_size = ct.sizeof(ct.c_double)
        elif dtype == 'float':
            num_size = ct.sizeof(ct.c_float)
        elif dtype == 'int':
            num_size = ct.sizeof(ct.c_int)

        num_elements = self.buf_nx * self.buf_ny
        if self.buf_nz is not None:
            num_elements *= self.buf_nz

        local = cl.LocalMemory(num_size * num_elements)

        return local

class D2Q9(Velocity_Set):

    def __init__(self, ctx_info, context, kernel_args):

        super(D2Q9, self).__init__(ctx_info, context, kernel_args)

        self.name = 'D2Q9'

        ##########################
        ##### D2Q9 parameters ####
        ##########################

        self.w = np.array([4. / 9., 1. / 9., 1. / 9., 1. / 9., 1. / 9., 1. / 36.,
                      1. / 36., 1. / 36., 1. / 36.], order='F', dtype=num_type)  # weights for directions
        self.cx = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], order='F', dtype=int_type)  # direction vector for the x direction
        self.cy = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], order='F', dtype=int_type)  # direction vector for the y direction

        self.c_vec = np.array([self.cx, self.cy])
        self.c_mag = np.sqrt(np.sum(self.c_vec**2, axis=0))

        self.cs = num_type(1. / np.sqrt(3))  # Speed of sound on the lattice
        self.num_jumpers = int_type(9)  # Number of jumpers for the D2Q9 lattice: 9

        # Create arrays for bounceback and zero-shear/symmetry conditions
        self.reflect_index = np.zeros(self.num_jumpers, order='F', dtype=int_type)
        for i in range(self.reflect_index.shape[0]):
            cur_cx = self.cx[i]
            cur_cy = self.cy[i]

            reflect_cx = -cur_cx
            reflect_cy = -cur_cy

            opposite = (reflect_cx == self.cx) & (reflect_cy == self.cy)
            self.reflect_index[i] = np.where(opposite)[0][0]

        # When you go out of bounds in the x direction...and need to reflect back keeping y momentum
        slip_x_index = np.zeros(self.num_jumpers, order='F', dtype=int_type)
        for i in range(slip_x_index.shape[0]):
            cur_cx = self.cx[i]
            cur_cy = self.cy[i]

            reflect_cx = -cur_cx
            reflect_cy = cur_cy

            opposite = (reflect_cx == self.cx) & (reflect_cy == self.cy)
            slip_x_index[i] = np.where(opposite)[0][0]

        # When you go out of bounds in the y direction...and need to reflect back keeping x momentum
        slip_y_index = np.zeros(self.num_jumpers, order='F', dtype=int_type)
        for i in range(slip_y_index.shape[0]):
            cur_cx = self.cx[i]
            cur_cy = self.cy[i]

            reflect_cx = cur_cx
            reflect_cy = -cur_cy

            opposite = (reflect_cx == self.cx) & (reflect_cy == self.cy)
            slip_y_index[i] = np.where(opposite)[0][0]

        self.slip_index = np.array([slip_x_index, slip_y_index])


        # Define other important info
        self.halo = int_type(1)
        self.buf_nx = int_type(self.ctx_info['local_size'][0] + 2*self.halo)
        self.buf_ny = int_type(self.ctx_info['local_size'][1] + 2*self.halo)
        self.buf_nz = None

        self.nx_bc = int_type(self.ctx_info['nx'] + 2*self.halo)
        self.ny_bc = int_type(self.ctx_info['ny'] + 2*self.halo)
        self.nz_bc = None

        self.bc_size = (self.nx_bc, self.ny_bc)

        # Now that everything is defined...set the corresponding kernel definitions
        self.set_kernel_args()

class Autogen_Kernel(object):
    def __init__(self, short_name, opencl_kernel, sim):

        print 'Connecting python to the opencl_kernel ' + short_name + '...'

        self.short_name = short_name # The name of the kernel in the kernel_arg dict

        self.opencl_kernel = opencl_kernel # The opencl kernel that is run

        self.sim = weakref.proxy(sim) # Need a weakref...kernel is a part of the simulation, and not vice versa

        self.arg_list = None

        self.create_arg_list()

    def create_arg_list(self):
        """
        Determine what Python variables correspond to those required by the auto-generated (Mako-generated) kernel.
        """

        sim = self.sim

        py_kernel_args = sim.kernel_args # Python variables that are passed into the kernel
        gen_kernel_args = sim.ctx_info['kernel_arguments'] # A list of needed kernel arguments from kernel autogen (Mako)

        list_for_kernel = gen_kernel_args[self.short_name]

        python_args_needed = [z[0] for z in list_for_kernel]

        self.arg_list = [py_kernel_args[z] for z in python_args_needed]

        additional_cl_args = [sim.queue, sim.global_size, sim.local_size]

        self.arg_list = additional_cl_args + self.arg_list

    def run(self):
        """Usually attaches a .wait() on the return value."""
        return self.opencl_kernel(*self.arg_list)

class DLA_Colony(object):

    def __init__(self, ctx_info=None, velocity_set=None,
                 bc_map=None, rho=None, absorbed_mass=None,
                 k_list=None, m_reproduce_list=None, D=None,
                 context=None, use_interop=False, f_rand_amp=1e-6):

        self.ctx_info = ctx_info
        self.kernel_args = {}

        # Create global & local sizes appropriately
        self.local_size = self.ctx_info['local_size']
        self.global_size = get_divisible_global(self.ctx_info['domain_size'], self.local_size)

        print 'global size:' , self.global_size
        print 'local size:' , self.local_size

        # Initialize the opencl environment
        self.context = context     # The pyOpenCL context
        self.queue = None       # The queue used to issue commands to the desired device
        self.kernels = None     # Compiled OpenCL kernels
        self.use_interop = use_interop
        self.init_opencl()      # Initializes all items required to run OpenCL code

        # Convert the list of k's and m_reproduce to buffer
        self.k_list = np.array(k_list, dtype=num_type, order='F')
        self.m_reproduce_list = np.array(m_reproduce_list, dtype=num_type, order='F')

        const_flags = cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR

        self.kernel_args['k_list'] = cl.Buffer(self.context, const_flags, hostbuf=self.k_list)
        self.kernel_args['m_reproduce_list'] = cl.Buffer(self.context, const_flags, hostbuf=self.m_reproduce_list)

        self.D = num_type(D)
        self.kernel_args['D'] = self.D

        # Initialize the velocity set...and other important context-wide
        # variables.
        self.velocity_set = None
        if velocity_set == 'D2Q9':
            self.velocity_set = D2Q9(self.ctx_info, self.context, self.kernel_args)

        # Determine the relxation time scale
        self.tau = num_type(.5 + self.D / (self.velocity_set.cs ** 2))
        print 'tau', self.tau
        self.omega = num_type(self.tau ** -1.)  # The relaxation time of the jumpers in the simulation
        print 'omega', self.omega

        self.kernel_args['tau'] = self.tau
        self.kernel_args['omega'] = self.omega

        ## Initialize the node map...user is responsible for passing this in correctly.
        bc_map = np.array(bc_map, dtype=int_type, order='F')
        self.bc_map = cl.array.to_device(self.queue, bc_map)
        self.bc_map_streamed = self.bc_map.copy()

        self.kernel_args['bc_map'] = self.bc_map.data
        self.kernel_args['bc_map_streamed'] = self.bc_map_streamed.data

        self.global_size_bc = get_divisible_global(self.velocity_set.bc_size, self.local_size)
        print 'global_size_bc:', self.global_size_bc

        ## Initialize hydrodynamic variables
        rho_host = np.array(rho, dtype=num_type, order='F')
        self.rho = cl.array.to_device(self.queue, rho_host)
        self.kernel_args['rho'] = self.rho.data

        absorbed_mass_host = np.array(absorbed_mass, dtype=num_type, order='F')
        self.absorbed_mass = cl.array.to_device(self.queue, rho_host)
        self.kernel_args['absorbed_mass'] = self.absorbed_mass.data

        # Intitialize the underlying feq equilibrium field
        feq_host = np.zeros(self.get_jumper_tuple(), dtype=num_type, order='F')
        self.feq = cl.array.to_device(self.queue, feq_host)
        self.kernel_args['feq'] = self.feq.data

        self.init_feq = Autogen_Kernel('init_feq', self.kernels.init_feq, self)

        self.init_feq.run().wait() # Based on the input hydrodynamic fields, create feq


        f_host = np.zeros(self.get_jumper_tuple(), dtype=num_type, order='F')
        self.f = cl.array.to_device(self.queue, f_host)
        self.f_streamed = self.f.copy()

        self.kernel_args['f'] = self.f.data
        self.kernel_args['f_streamed'] = self.f_streamed.data

        # Now initialize the nonequilibrium f
        self.init_pop(amplitude=f_rand_amp) # Based on feq, create the hopping non-equilibrium fields

        # Generate the rest of the needed kernels
        ker = self.kernels
        #self.collide_and_propagate = Autogen_Kernel('collide_and_propagate', ker.collide_and_propagate, self)
        #self.update_after_streaming = Autogen_Kernel('update_after_streaming', ker.update_after_streaming, self)
        #self.reproduce = Autogen_Kernel('reproduce', ker.reproduce, self)


    def get_dimension_tuple(self):

        dimension = self.ctx_info['dimension']
        nx = self.ctx_info['nx']
        ny = self.ctx_info['ny']
        nz = self.ctx_info['nz']

        if dimension == 2:
            return (nx, ny)
        elif dimension == 3:
            return (nx, ny, nz)

    def get_jumper_tuple(self):
        return self.get_dimension_tuple() + (self.velocity_set.num_jumpers)

    def init_opencl(self):
        """
        Initializes the base items needed to run OpenCL code.
        """

        # Startup script shamelessly taken from CS205 homework

        if self.context is None:
            platforms = cl.get_platforms()
            print 'The platforms detected are:'
            print '---------------------------'
            for platform in platforms:
                print platform.name, platform.vendor, 'version:', platform.version

            # List devices in each platform
            for platform in platforms:
                print 'The devices detected on platform', platform.name, 'are:'
                print '---------------------------'
                for device in platform.get_devices():
                    print device.name, '[Type:', cl.device_type.to_string(device.type), ']'
                    print 'Maximum clock Frequency:', device.max_clock_frequency, 'MHz'
                    print 'Maximum allocable memory size:', int(device.max_mem_alloc_size / 1e6), 'MB'
                    print 'Maximum work group size', device.max_work_group_size
                    print 'Maximum work item dimensions', device.max_work_item_dimensions
                    print 'Maximum work item size', device.max_work_item_sizes
                    print '---------------------------'

            # Create a context with all the devices
            devices = platforms[0].get_devices()
            if not self.use_interop:
                self.context = cl.Context(devices)
            else:
                self.context = cl.Context(properties=[(cl.context_properties.PLATFORM, platforms[0])]
                                                     + cl.tools.get_gl_sharing_context_properties(),
                                          devices= devices)
            print 'This context is associated with ', len(self.context.devices), 'devices'

        # Create a simple queue
        self.queue = cl.CommandQueue(self.context, self.context.devices[0],
                                     properties=cl.command_queue_properties.PROFILING_ENABLE)
        # Compile our OpenCL code...render MAKO first.
        template = mte.Template(
            filename= file_dir + '/colony_growth.mako',
            strict_undefined=True
        )
        buf = sio.StringIO()

        mako_context = mrt.Context(buf, **self.ctx_info)
        template.render_context(mako_context)

        with open('temp_kernels.cl', 'w') as fi:
            fi.write(buf.getvalue())

        self.kernels = cl.Program(self.context, buf.getvalue()).build(options='')

    def init_pop(self, amplitude=0.001):
        """Based on feq, create the initial population of jumpers."""

        # For simplicity, copy feq to the local host, where you can make a copy. There is probably a better way to do this.
        f_host = self.feq.get()

        # We now slightly perturb f.
        perturb = 1. + amplitude * np.random.randn(*f_host.shape)
        f_host *= perturb

        # Now send f to the GPU
        self.f = cl.array.to_device(self.queue, f_host)
        self.f_streamed = self.f.copy()


    def add_eating_rate(self, eater_index, eatee_index, rate, eater_cutoff):
        """
        Eater eats eatee at a given rate.
        :param eater:
        :param eatee:
        :param rate:
        :return:
        """

        kernel_to_run = self.kernels.add_eating_collision
        arguments = [
            self.queue, self.two_d_global_size, self.two_d_local_size,
            int_type(eater_index), int_type(eatee_index), num_type(rate),
            num_type(eater_cutoff),
            self.f.data, self.rho.data,
            self.w, self.cx, self.cy,
            self.nx, self.ny, self.num_populations, self.num_jumpers,
            self.cs
        ]

        self.additional_collisions.append([kernel_to_run, arguments])


    def add_growth(self, eater_index, min_rho_cutoff, max_rho_cutoff, eat_rate):
        """
        Grows uniformly everywhere.
        """

        kernel_to_run = self.kernels.add_growth
        arguments = [
            self.queue, self.two_d_global_size, self.two_d_local_size,
            int_type(eater_index),
            num_type(min_rho_cutoff), num_type(max_rho_cutoff),
            num_type(eat_rate),
            self.f.data, self.rho.data,
            self.w, self.cx, self.cy,
            self.nx, self.ny, self.num_populations, self.num_jumpers,
            self.cs
        ]

        self.additional_collisions.append([kernel_to_run, arguments])


    def add_constant_g_force(self, fluid_index, force_x, force_y):

        kernel_to_run = self.kernels.add_constant_g_force
        arguments = [
            self.queue, self.two_d_global_size, self.two_d_local_size,
            int_type(fluid_index), num_type(force_x), num_type(force_y),
            self.Gx.data, self.Gy.data,
            self.rho.data,
            self.nx, self.ny
        ]

        self.additional_forces.append([kernel_to_run, arguments])

    def add_boussinesq_force(self, flow_field_num, solute_field_num, rho_cutoff, solute_ref_density, g_x, g_y):

        kernel_to_run = self.kernels.add_boussinesq_force
        arguments = [
            self.queue, self.two_d_global_size, self.two_d_local_size,
            int_type(flow_field_num), int_type(solute_field_num),
            num_type(rho_cutoff), num_type(solute_ref_density),
            num_type(g_x), num_type(g_y),
            self.Gx.data, self.Gy.data,
            self.rho.data,
            self.nx, self.ny
        ]

        self.additional_forces.append([kernel_to_run, arguments])

    def add_buoyancy_difference(self, flow_field_num, rho_cutoff, rho_ref, g_x, g_y):

        kernel_to_run = self.kernels.add_buoyancy_difference
        arguments = [
            self.queue, self.two_d_global_size, self.two_d_local_size,
            int_type(flow_field_num), num_type(rho_cutoff),
            num_type(rho_ref),
            num_type(g_x), num_type(g_y),
            self.Gx.data, self.Gy.data,
            self.rho.data,
            self.nx, self.ny
        ]

        self.additional_forces.append([kernel_to_run, arguments])

    def add_radial_g_force(self, fluid_index, center_x, center_y, prefactor, radial_scaling):

        kernel_to_run = self.kernels.add_radial_g_force
        arguments = [
            self.queue, self.two_d_global_size, self.two_d_local_size,
            int_type(fluid_index), int_type(center_x), int_type(center_y),
            num_type(prefactor), num_type(radial_scaling),
            self.Gx.data, self.Gy.data,
            self.rho.data,
            self.nx, self.ny
        ]

        self.additional_forces.append([kernel_to_run, arguments])

    ##### Dealing with Poisson Repulsion. ###########
    def add_screened_poisson_force(self, source_index, force_index, interaction_length, amplitude):

        input_density = self.rho.get()[:, :, source_index]
        self.poisson_solver = sp.Screened_Poisson(input_density, cl_context=self.context, cl_queue = self.queue,
                                                  lam=interaction_length, dx=1.0)
        self.poisson_solver.create_grad_fields()

        self.poisson_force_active = True
        self.poisson_source_index = int_type(source_index)
        self.poisson_force_index = int_type(force_index)
        self.poisson_amp = amplitude

    def screened_poisson_kernel(self):
        # Update the charge field for the poisson solver
        density_view = self.rho[:, :, self.poisson_source_index]

        cl.enqueue_copy(self.queue, self.poisson_solver.charge.data, density_view.astype(np.complex64).data)

        self.poisson_solver.solve_and_update_grad_fields()
        self.poisson_xgrad = self.poisson_amp * self.poisson_solver.xgrad.real
        self.poisson_ygrad = self.poisson_amp * self.poisson_solver.ygrad.real

        self.Gx[:, :, self.poisson_force_index] += self.poisson_xgrad
        self.Gy[:, :, self.poisson_force_index] += self.poisson_ygrad
    ############################################

    def add_interaction_force(self, fluid_1_index, fluid_2_index, G_int, bc='periodic', potential='linear',
                              potential_parameters=None, rho_wall = None):

        # We use the D2Q9 stencil for this force
        w_arr = np.array([4. / 9., 1. / 9., 1. / 9., 1. / 9., 1. / 9., 1. / 36.,
                      1. / 36., 1. / 36., 1. / 36.], order='F', dtype=num_type)  # weights for directions
        cx_arr = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], order='F', dtype=int_type)  # direction vector for the x direction
        cy_arr = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], order='F', dtype=int_type)  # direction vector for the y direction

        w = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=w_arr)
        cx = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=cx_arr)
        cy = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=cy_arr)

        cs = num_type(1. / np.sqrt(3))  # Speed of sound on the lattice
        num_jumpers = int_type(9)  # Number of jumpers for the D2Q9 lattice: 9

        # Allocate local memory
        halo = int_type(1) # As we are doing D2Q9, we have a halo of one
        buf_nx = int_type(self.two_d_local_size[0] + 2 * halo)
        buf_ny = int_type(self.two_d_local_size[1] + 2 * halo)


        psi_local_1 = cl.LocalMemory(num_size * buf_nx * buf_ny)
        psi_local_2 = cl.LocalMemory(num_size * buf_nx * buf_ny)

        kernel_to_run = self.kernels.add_interaction_force
        arguments = [
            self.queue, self.two_d_global_size, self.two_d_local_size,
            int_type(fluid_1_index), int_type(fluid_2_index), num_type(G_int),
            psi_local_1, psi_local_2,
            self.rho.data, self.Gx.data, self.Gy.data,
            cs, cx, cy, w,
            self.nx, self.ny,
            buf_nx, buf_ny, halo, num_jumpers
        ]

        if bc is 'periodic':
            arguments += [int_type(0)]
        elif bc is 'zero_gradient':
            arguments += [int_type(1)]
        elif bc is 'zero_density':
            arguments += [int_type(2)]
        else:
            raise ValueError('Specified boundary condition does not exist')

        if potential is 'linear':
            arguments += [int_type(0)]
        elif potential is 'shan_chen':
            arguments += [int_type(1)]
        elif potential is 'pow':
            arguments += [int_type(2)]
        elif potential is 'vdw':
            arguments += [int_type(3)]
        else:
            raise ValueError('Specified pseudopotential does not exist.')

        if potential_parameters is None:
            potential_parameters = np.array([0.], dtype=num_type)
        else:
            potential_parameters = np.array(potential_parameters, dtype=num_type)

        parameters_const = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                     hostbuf=potential_parameters)

        arguments += [parameters_const]

        if rho_wall is None:
            rho_wall = 1.0
        arguments += [num_type(rho_wall)]

        self.additional_forces.append([kernel_to_run, arguments])

    def add_interaction_force_second_belt(self, fluid_1_index, fluid_2_index, G_int, bc='periodic', potential='linear',
                                          potential_parameters=None, rho_wall=None):

        #### pi1 ####
        pi1 = []
        cx1 = []
        cy1 = []

        c_temp = [
            [1, 0],
            [0, 1],
            [-1, 0],
            [0, -1]
        ]

        for c_vec in c_temp:
            pi1.append(4./63.)
            cx1.append(c_vec[0])
            cy1.append(c_vec[1])

        c_temp = [
            [1, 1],
            [-1, 1],
            [-1, -1],
            [1, -1]
        ]

        for c_vec in c_temp:
            pi1.append(4./135.)
            cx1.append(c_vec[0])
            cy1.append(c_vec[1])

        num_jumpers_1 = int_type(len(pi1))

        #### pi2 ####
        pi2 = []
        cx2 = []
        cy2 = []

        c_temp = [
            [2, 0],
            [0, 2],
            [-2, 0],
            [0, -2]
        ]

        for c_vec in c_temp:
            pi2.append(1./180.)
            cx2.append(c_vec[0])
            cy2.append(c_vec[1])

        c_temp = [
            [2, -1],
            [2, 1],
            [1, 2],
            [-1, 2],
            [-2, 1],
            [-2, -1],
            [-1, -2],
            [1, -2]
        ]

        for c_vec in c_temp:
            pi2.append(2./945.)
            cx2.append(c_vec[0])
            cy2.append(c_vec[1])

        c_temp = [
            [2, 2],
            [-2, 2],
            [-2, -2],
            [2, -2]
        ]
        for c_vec in c_temp:
            pi2.append(1./15120.)
            cx2.append(c_vec[0])
            cy2.append(c_vec[1])

        num_jumpers_2 = int_type(len(pi2))

        ### Finish setup ###

        pi1 = np.array(pi1, dtype=num_type)
        cx1 = np.array(cx1, dtype=int_type)
        cy1 = np.array(cy1, dtype=int_type)

        pi2 = np.array(pi2, dtype=num_type)
        cx2 = np.array(cx2, dtype=int_type)
        cy2 = np.array(cy2, dtype=int_type)

        pi1_const = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=pi1)
        cx1_const = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=cx1)
        cy1_const = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=cy1)

        pi2_const = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=pi2)
        cx2_const = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=cx2)
        cy2_const = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=cy2)

        # Allocate local memory for the clumpiness
        cur_halo = int_type(2)
        cur_buf_nx = int_type(self.two_d_local_size[0] + 2 * cur_halo)
        cur_buf_ny = int_type(self.two_d_local_size[1] + 2 * cur_halo)

        local_1 = cl.LocalMemory(num_size * cur_buf_nx * cur_buf_ny)
        local_2 = cl.LocalMemory(num_size * cur_buf_nx * cur_buf_ny)

        kernel_to_run = self.kernels.add_interaction_force_second_belt
        arguments = [
            self.queue, self.two_d_global_size, self.two_d_local_size,
            int_type(fluid_1_index), int_type(fluid_2_index), num_type(G_int),
            local_1, local_2,
            self.rho.data, self.Gx.data, self.Gy.data,
            self.cs,
            pi1_const, cx1_const, cy1_const, num_jumpers_1,
            pi2_const, cx2_const, cy2_const, num_jumpers_2,
            self.nx, self.ny,
            cur_buf_nx, cur_buf_ny, cur_halo
        ]

        if bc is 'periodic':
            arguments += [int_type(0)]
        elif bc is 'zero_gradient':
            arguments += [int_type(1)]
        elif bc is 'zero_density':
            arguments += [int_type(2)]
        else:
            raise ValueError('Specified boundary condition does not exist')

        if potential is 'linear':
            arguments += [int_type(0)]
        elif potential is 'shan_chen':
            arguments += [int_type(1)]
        elif potential is 'pow':
            arguments += [int_type(2)]
        elif potential is 'vdw':
            arguments += [int_type(3)]
        else:
            raise ValueError('Specified pseudopotential does not exist.')

        if potential_parameters is None:
            potential_parameters = np.array([0.], dtype=num_type)
        else:
            potential_parameters = np.array(potential_parameters, dtype=num_type)

        parameters_const = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                     hostbuf=potential_parameters)

        arguments += [parameters_const]

        if rho_wall is None:
            rho_wall = 1.0
        arguments += [num_type(rho_wall)]

        self.additional_forces.append([kernel_to_run, arguments])

    def run(self, num_iterations, debug=False):
        """
        Run the simulation for num_iterations. Be aware that the same number of iterations does not correspond
        to the same non-dimensional time passing, as delta_t, the time discretization, will change depending on
        your resolution.

        :param num_iterations: The number of iterations to run
        """
        for cur_iteration in range(num_iterations):
            if debug:
                print 'At beginning of iteration:'
                self.check_fields()

            for cur_fluid in self.fluid_list:
                cur_fluid.move() # Move all jumpers
            if debug:
                print 'After move'
                self.check_fields()

            # Update forces here as appropriate
            for cur_fluid in self.fluid_list:
                cur_fluid.update_hydro() # Update the hydrodynamic variables
            if debug:
                print 'After updating hydro'
                self.check_fields()

            # Reset the total body force and add to it as appropriate
            self.Gx[...] = 0
            self.Gy[...] = 0
            for d in self.additional_forces:
                kernel = d[0]
                arguments = d[1]
                kernel(*arguments).wait()
            if self.poisson_force_active:
                self.screened_poisson_kernel()
            if debug:
                print 'After updating supplementary forces'
                self.check_fields()

            # Update other forces...includes pourous effects & must be run last
            for cur_fluid in self.fluid_list:
                cur_fluid.update_forces()
            if debug:
                print 'After updating internal forces'
                self.check_fields()

            # After updating forces, update the bary_velocity
            self.update_bary_velocity()
            if debug:
                print 'After updating bary-velocity'
                self.check_fields()

            for cur_fluid in self.fluid_list:
                cur_fluid.update_feq() # Update the equilibrium fields
            if debug:
                print 'After updating feq'
                self.check_fields()

            for cur_fluid in self.fluid_list:
                cur_fluid.collide_particles() # Relax the nonequilibrium fields.
            if debug:
                print 'After colliding particles'
                self.check_fields()

            # Loop over any additional collisions that are required (i.e. mass gain/loss)
            for d in self.additional_collisions:
                kernel = d[0]
                arguments = d[1]
                kernel(*arguments).wait()

    def check_fields(self):
        # Start with rho
        for i in range(self.num_populations):
            print 'Field:', i
            print 'rho_sum', cl.array.sum(self.rho[:, :, i])
            print 'u, v bary sum', cl.array.sum(self.u_bary), cl.array.sum(self.u_bary)
            print 'f_sum', np.sum(self.f.get()[:, :, i, :])
            print 'f_eq_sum', np.sum(self.feq.get()[:, :, i, :])

        print 'Total rho_sum', cl.array.sum(self.rho)
        print 'Total f_sum', np.sum(self.f.get())
        print 'Total feq_sum', np.sum(self.feq.get())

        print

#
# class Simulation_RunnerD2Q25(Simulation_Runner):
#     def __init__(self, **kwargs):
#         super(Simulation_RunnerD2Q25, self).__init__(**kwargs)
#
#     def allocate_constants(self):
#         """
#         Allocates constants and local memory to be used by OpenCL.
#         """
#
#         ##########################
#         ##### D2Q25 parameters####
#         ##########################
#         t0 = (4./45.)*(4 + np.sqrt(10))
#         t1 = (3./80.)*(8 - np.sqrt(10))
#         t3 = (1./720.)*(16 - 5*np.sqrt(10))
#
#         w_list = []
#         cx_list = []
#         cy_list = []
#
#         # Mag 0
#         cx_list += [0]
#         cy_list += [0]
#         w_list += [t0*t0]
#
#         # Mag 1
#         cx_list += [0, 0, 1, -1]
#         cy_list += [1, -1, 0, 0]
#         w_list += 4*[t0*t1]
#
#         # Mag sqrt(2)
#         cx_list += [1, 1, -1, -1]
#         cy_list += [1, -1, 1, -1]
#         w_list += 4*[t1*t1]
#
#         # Mag 3
#         cx_list += [3, -3, 0, 0]
#         cy_list += [0, 0, 3, -3]
#         w_list += 4*[t0*t3]
#
#         # Mag sqrt(10)
#         cx_list += [1, 1, -1, -1, 3, 3, -3, -3]
#         cy_list += [3, -3, 3, -3, 1, -1, 1, -1]
#         w_list += 8*[t1*t3]
#
#         # Mag sqrt(18)
#         cx_list += [3, 3, -3, -3]
#         cy_list += [3, -3, 3, -3]
#         w_list += 4*[t3 * t3]
#
#         # Now send everything to disk
#         w = np.array(w_list, order='F', dtype=num_type)  # weights for directions
#         cx = np.array(cx_list, order='F', dtype=int_type)  # direction vector for the x direction
#         cy = np.array(cy_list, order='F', dtype=int_type)  # direction vector for the y direction
#
#         self.cs = num_type(np.sqrt(1. - np.sqrt(2./5.)))  # Speed of sound on the lattice
#         self.num_jumpers = int_type(w.shape[0])  # Number of jumpers: should be 25
#
#         reflect_index = np.zeros(self.num_jumpers, order='F', dtype=int_type)
#         for i in range(reflect_index.shape[0]):
#             cur_cx = cx[i]
#             cur_cy = cy[i]
#
#             reflect_cx = -cur_cx
#             reflect_cy = -cur_cy
#
#             opposite = (reflect_cx == cx) & (reflect_cy == cy)
#             reflect_index[i] = np.where(opposite)[0][0]
#
#         # When you go out of bounds in the x direction...and need to reflect back keeping y momentum
#         slip_x_index = np.zeros(self.num_jumpers, order='F', dtype=int_type)
#         for i in range(slip_x_index.shape[0]):
#             cur_cx = cx[i]
#             cur_cy = cy[i]
#
#             reflect_cx = -cur_cx
#             reflect_cy = cur_cy
#
#             opposite = (reflect_cx == cx) & (reflect_cy == cy)
#             slip_x_index[i] = np.where(opposite)[0][0]
#
#         # When you go out of bounds in the y direction...and need to reflect back keeping x momentum
#         slip_y_index = np.zeros(self.num_jumpers, order='F', dtype=int_type)
#         for i in range(slip_y_index.shape[0]):
#             cur_cx = cx[i]
#             cur_cy = cy[i]
#
#             reflect_cx = cur_cx
#             reflect_cy = -cur_cy
#
#             opposite = (reflect_cx == cx) & (reflect_cy == cy)
#             slip_y_index[i] = np.where(opposite)[0][0]
#
#         self.w = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=w)
#         self.cx = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=cx)
#         self.cy = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=cy)
#         self.reflect_index = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
#                                        hostbuf=reflect_index)
#         self.slip_x_index = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
#                                        hostbuf=slip_x_index)
#         self.slip_y_index = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
#                                       hostbuf=slip_y_index)