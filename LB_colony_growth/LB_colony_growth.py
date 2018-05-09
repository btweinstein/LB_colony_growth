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

import inspect

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

        self.reflect_list = None
        self.slip_list = None

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

        self.kernel_args['reflect_list'] = cl.Buffer(self.context, const_flags, hostbuf=self.reflect_list)
        self.kernel_args['slip_list'] = cl.Buffer(self.context, const_flags, hostbuf=self.slip_list)

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

        print 'Creating local memory of', dtype, 'type...'

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

        self.c_vec = np.array([self.cx, self.cy], order='F', dtype=int_type)
        self.c_mag = np.sqrt(np.sum(self.c_vec**2, axis=0), order='F', dtype=num_type)

        self.cs = num_type(1. / np.sqrt(3))  # Speed of sound on the lattice
        self.num_jumpers = int_type(9)  # Number of jumpers for the D2Q9 lattice: 9

        # Create arrays for bounceback and zero-shear/symmetry conditions
        self.reflect_list = np.zeros(self.num_jumpers, order='F', dtype=int_type)
        for i in range(self.reflect_list.shape[0]):
            cur_cx = self.cx[i]
            cur_cy = self.cy[i]

            reflect_cx = -cur_cx
            reflect_cy = -cur_cy

            opposite = (reflect_cx == self.cx) & (reflect_cy == self.cy)
            self.reflect_list[i] = np.where(opposite)[0][0]

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

        self.slip_list = np.array([slip_x_index, slip_y_index], order='F', dtype=int_type)


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
    def __init__(self, short_name, opencl_kernel, sim, kernel_global_size=None, kernel_local_size=None):

        print 'Connecting python to the opencl_kernel ' + short_name + '...'

        self.short_name = short_name # The name of the kernel in the kernel_arg dict

        self.opencl_kernel = opencl_kernel # The opencl kernel that is run

        self.sim = weakref.proxy(sim) # Need a weakref...kernel is a part of the simulation, and not vice versa

        self.kernel_global_size = kernel_global_size
        if self.kernel_global_size is None:
            self.kernel_global_size = self.sim.global_size

        self.kernel_local_size = kernel_local_size
        if self.kernel_local_size is None:
            self.kernel_local_size = self.sim.local_size

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

        # Loop over the arg_list...if the argument is a function, call it!
        for i in range(len(self.arg_list)):
            value = self.arg_list[i]
            if inspect.isfunction(value):
                self.arg_list[i] = value()

        additional_cl_args = [sim.queue, self.kernel_global_size, self.kernel_local_size]

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
        self.dimension = self.ctx_info['dimension']
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
        assert k_list is not None, 'Need k for each allele'
        assert m_reproduce_list is not None, 'Need mass to reproduce for each allele'
        self.k_list = np.array(k_list, dtype=num_type, order='F')
        self.m_reproduce_list = np.array(m_reproduce_list, dtype=num_type, order='F')

        const_flags = cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR

        self.kernel_args['k_list'] = cl.Buffer(self.context, const_flags, hostbuf=self.k_list)
        self.kernel_args['m_reproduce_list'] = cl.Buffer(self.context, const_flags, hostbuf=self.m_reproduce_list)

        assert D is not None, 'Need diffusion constant D'

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
        assert bc_map is not None, 'Need map of boundary conditions...'

        bc_map = np.array(bc_map, dtype=int_type, order='F')
        self.bc_map = cl.array.to_device(self.queue, bc_map)
        self.bc_map_streamed = self.bc_map.copy()

        self.kernel_args['bc_map'] = self.bc_map.data
        self.kernel_args['bc_map_streamed'] = self.bc_map_streamed.data

        self.global_size_bc = get_divisible_global(self.velocity_set.bc_size, self.local_size)
        print 'global_size_bc:', self.global_size_bc

        ## Initialize hydrodynamic variables
        assert rho is not None, 'Need input initial density'
        rho_host = np.array(rho, dtype=num_type, order='F')
        self.rho = cl.array.to_device(self.queue, rho_host)
        self.kernel_args['rho'] = self.rho.data

        assert absorbed_mass is not None, 'Need initial absorbed mass for each cell'
        absorbed_mass_host = np.array(absorbed_mass, dtype=num_type, order='F')
        self.absorbed_mass = cl.array.to_device(self.queue, absorbed_mass_host)
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

        # Now initialize the nonequilibrium f
        self.init_pop(amplitude=f_rand_amp) # Based on feq, create the hopping non-equilibrium fields

        # This needs to occur AFTER the population is initialized!
        self.kernel_args['f'] = self.f.data
        self.kernel_args['f_streamed'] = self.f_streamed.data

        # Initialize the random generator
        self.random_generator = cl.clrandom.PhiloxGenerator(self.context)
        # Draw random normals for each population
        random_host = np.ones(self.get_dimension_tuple(), dtype=num_type, order='F')
        self.rand_array = cl.array.to_device(self.queue, random_host)

        self.random_generator.fill_uniform(self.rand_array, queue=self.queue)
        self.rand_array.finish()

        self.kernel_args['rand'] = self.rand_array.data

        # Create global memory required for reproduction
        can_reproduce = np.array([0], dtype=int_type, order='F')
        self.can_reproduce = cl.array.to_device(self.queue, can_reproduce)

        self.kernel_args['can_reproduce_pointer'] = self.can_reproduce.data

        # Generate the rest of the needed kernels
        ker = self.kernels
        self.collide_and_propagate = Autogen_Kernel('collide_and_propagate', ker.collide_and_propagate, self)
        self.update_after_streaming = Autogen_Kernel('update_after_streaming', ker.update_after_streaming, self)
        self.reproduce = Autogen_Kernel('reproduce', ker.reproduce, self)

        self.copy_streamed_onto_f = Autogen_Kernel('copy_streamed_onto_f', ker.copy_streamed_onto_f, self)
        self.copy_streamed_onto_bc = Autogen_Kernel('copy_streamed_onto_bc', ker.copy_streamed_onto_bc, self,
                                                    kernel_global_size=self.global_size_bc)

    def run(self, num_iterations, reproduction_cutoff = 1000):
        for i in range(num_iterations):
            self.collide_and_propagate.run().wait()
            self.copy_streamed_onto_f.run().wait() # TODO: Use the ABA access patern for streaming
            self.update_after_streaming.run().wait()

            #plt.imshow(self.absorbed_mass.get() > 1.)

            # Reproduce!
            self.can_reproduce[0] = int_type(1) # Test if anyone can reproduce
            num_times = 0
            while (self.can_reproduce[0] == 1):
                # Generate new random numbers
                self.random_generator.fill_uniform(self.rand_array, queue=self.queue)
                self.rand_array.finish()

                # Attempt to reproduce
                self.can_reproduce[0] = int_type(0) # The kernel will reset the flag if anyone can reproduce
                self.reproduce.run().wait()
                self.copy_streamed_onto_bc.run().wait()

                num_times += 1
                if num_times >= reproduction_cutoff:
                    assert False, "I've run the reproduction step 1000 times. Something is probably wrong. Quitting..."

    def get_pop_field(self):
        # Chop off the edge of the bc_map
        populations = self.bc_map.get()
        halo = self.velocity_set.halo
        if self.dimension == 2:
            populations = populations[halo:-halo, halo:-halo]
        elif self.dimension == 3:
            populations = populations[halo:-halo, halo:-halo]

        # Let 0 be background. Other integers refer to populations
        populations*= -1

        return populations

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
        return self.get_dimension_tuple() + (int(self.velocity_set.num_jumpers),)

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

        with open('temp_kernels_DLA_colony.cl', 'w') as fi:
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
        self.f[...] = f_host
        self.f_streamed = self.f.copy()



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