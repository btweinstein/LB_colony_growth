import numpy as np
import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
import pyopencl as cl
import pyopencl.tools
import pyopencl.clrandom
import pyopencl.array
import ctypes as ct
import weakref

class Velocity_Set(object):
    def __init__(self, sim):

        self.sim = weakref.proxy(sim)  # Need a weakref...kernel is a part of the simulation, and not vice versa

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

        kernel_args = self.sim.kernel_args
        opencl_context = self.sim.context

        kernel_args['num_jumpers'] = self.num_jumpers
        kernel_args['w'] = cl.Buffer(opencl_context, const_flags, hostbuf=self.w)
        kernel_args['c_vec'] = cl.Buffer(opencl_context, const_flags, hostbuf=self.c_vec)
        kernel_args['c_mag'] = cl.Buffer(opencl_context, const_flags, hostbuf=self.c_mag)
        kernel_args['cs'] = self.cs

        kernel_args['reflect_list'] = cl.Buffer(opencl_context, const_flags, hostbuf=self.reflect_list)
        kernel_args['slip_list'] = cl.Buffer(opencl_context, const_flags, hostbuf=self.slip_list)

        kernel_args['halo'] = self.halo

        kernel_args['nx_bc'] = self.nx_bc
        kernel_args['ny_bc'] = self.ny_bc
        kernel_args['nz_bc'] = self.nz_bc

        kernel_args['buf_nx'] = self.buf_nx
        kernel_args['buf_ny'] = self.buf_ny
        kernel_args['buf_nz'] = self.buf_nz

        # Add the ability to create local memory
        kernel_args['local_mem_num'] = lambda: self.create_local_memory('num_type')
        kernel_args['local_mem_int'] = lambda: self.create_local_memory('int')

    def create_local_memory(self, dtype):

        print 'Creating local memory of', dtype, 'type...'

        num_size = None

        if dtype == 'num_type':
            num_size = self.sim.num_size
        elif dtype == 'int':
            num_size = ct.sizeof(ct.c_int)

        num_elements = self.buf_nx * self.buf_ny
        if self.buf_nz is not None:
            num_elements *= self.buf_nz

        local = cl.LocalMemory(num_size * num_elements)

        return local

class D2Q9(Velocity_Set):

    def __init__(self, sim):

        super(D2Q9, self).__init__(sim)

        self.name = 'D2Q9'

        ##########################
        ##### D2Q9 parameters ####
        ##########################

        num_type = self.sim.num_type
        int_type = np.int32

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
        self.buf_nx = int_type(self.sim.ctx_info['local_size'][0] + 2*self.halo)
        self.buf_ny = int_type(self.sim.ctx_info['local_size'][1] + 2*self.halo)
        self.buf_nz = None

        self.nx_bc = int_type(self.sim.ctx_info['nx'] + 2*self.halo)
        self.ny_bc = int_type(self.sim.ctx_info['ny'] + 2*self.halo)
        self.nz_bc = None

        self.bc_size = (self.nx_bc, self.ny_bc)

        # Now that everything is defined...set the corresponding kernel definitions
        self.set_kernel_args()

    @classmethod
    def get_num_jumpers(cls):
        return np.int32(9)

class D3Q27(Velocity_Set):

    def __init__(self, sim):

        super(D3Q27, self).__init__(sim)

        self.name = 'D3Q27'

        num_type = self.sim.num_type
        int_type = np.int32

        # Pulled from LB principles and practice
        self.w = np.array(
            [
                8./27.,
                2./27., 2./27., 2./27., 2./27., 2./27., 2./27.,
                1./54., 1./54., 1./54., 1./54., 1./54., 1./54.,
                1./54., 1./54., 1./54., 1./54., 1./54., 1./54.,
                1./216., 1./216., 1./216., 1./216., 1./216., 1./216.,
                1./216., 1./216.
            ], order='F', dtype=num_type)  # weights for directions

        self.cx = np.array(
            [
                0, 1, -1,
                0, 0, 0, 0,
                1, -1, 1, -1, 0, 0,
                1, -1, 1, -1, 0, 0,
                1, -1, 1, -1, 1, -1, -1, 1
            ], order='F', dtype=int_type)  # direction vector for the x direction

        self.cy = np.array(
            [
                0, 0, 0,
                1, -1, 0, 0,
                1, -1, 0, 0,
                1, -1, -1, 1, 0, 0,
                1, -1, 1, -1, 1, -1, -1, 1, 1, -1
             ], order='F', dtype=int_type)  # direction vector for the y direction

        self.cz = np.array(
            [
                0, 0, 0, 0, 0,
                1, -1, 0, 0,
                1, -1, 1, -1, 0, 0,
                -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1
            ], order='F', dtype=int_type)  # direction vector for the z direction

        self.c_vec = np.array([self.cx, self.cy, self.cz], order='F', dtype=int_type)
        self.c_mag = np.sqrt(np.sum(self.c_vec**2, axis=0), order='F', dtype=num_type)

        self.cs = num_type(1. / np.sqrt(3))  # Speed of sound on the lattice
        self.num_jumpers = int_type(27)  # Number of jumpers for the D2Q9 lattice: 9

        # Create arrays for bounceback and zero-shear/symmetry conditions
        self.reflect_list = np.zeros(self.num_jumpers, order='F', dtype=int_type)
        for i in range(self.reflect_list.shape[0]):
            cur_cx = self.cx[i]
            cur_cy = self.cy[i]
            cur_cz = self.cz[i]

            reflect_cx = -cur_cx
            reflect_cy = -cur_cy
            reflect_cz = -cur_cz

            opposite = (reflect_cx == self.cx) & (reflect_cy == self.cy) & (reflect_cz == self.cz)
            self.reflect_list[i] = np.where(opposite)[0][0]

        # When you go out of bounds in the x direction...
        slip_x_index = np.zeros(self.num_jumpers, order='F', dtype=int_type)
        for i in range(slip_x_index.shape[0]):
            cur_cx = self.cx[i]
            cur_cy = self.cy[i]
            cur_cz = self.cz[i]

            reflect_cx = -cur_cx
            reflect_cy = cur_cy
            reflect_cz = cur_cz

            opposite = (reflect_cx == self.cx) & (reflect_cy == self.cy) & (reflect_cz == self.cz)
            slip_x_index[i] = np.where(opposite)[0][0]

        # When you go out of bounds in the y direction...
        slip_y_index = np.zeros(self.num_jumpers, order='F', dtype=int_type)
        for i in range(slip_y_index.shape[0]):
            cur_cx = self.cx[i]
            cur_cy = self.cy[i]
            cur_cz = self.cz[i]

            reflect_cx = cur_cx
            reflect_cy = -cur_cy
            reflect_cz = cur_cz

            opposite = (reflect_cx == self.cx) & (reflect_cy == self.cy) & (reflect_cz == self.cz)
            slip_y_index[i] = np.where(opposite)[0][0]

        # When you go out of bounds in the z direction...
        slip_z_index = np.zeros(self.num_jumpers, order='F', dtype=int_type)
        for i in range(slip_y_index.shape[0]):
            cur_cx = self.cx[i]
            cur_cy = self.cy[i]
            cur_cz = self.cz[i]

            reflect_cx = cur_cx
            reflect_cy = cur_cy
            reflect_cz = -cur_cz

            opposite = (reflect_cx == self.cx) & (reflect_cy == self.cy) & (reflect_cz == self.cz)
            slip_z_index[i] = np.where(opposite)[0][0]

        self.slip_list = np.array([slip_x_index, slip_y_index, slip_z_index], order='F', dtype=int_type)


        # Define other important info
        self.halo = int_type(1)
        local_size = self.sim.ctx_info['local_size']
        self.buf_nx = int_type(local_size[0] + 2*self.halo)
        self.buf_ny = int_type(local_size[1] + 2*self.halo)
        self.buf_nz = int_type(local_size[2] + 2*self.halo)

        self.nx_bc = int_type(self.sim.ctx_info['nx'] + 2*self.halo)
        self.ny_bc = int_type(self.sim.ctx_info['ny'] + 2*self.halo)
        self.nz_bc = int_type(self.sim.ctx_info['nz'] + 2*self.halo)

        self.bc_size = (self.nx_bc, self.ny_bc, self.nz_bc)

        # Now that everything is defined...set the corresponding kernel definitions
        self.set_kernel_args()