<%def name='print_kernel_args()' filter='trim'>
<%
    k = kernel_arguments['current_kernel_list']
    num_args = len(k)
    for i in range(num_args):
        context.write('     ')
        context.write(k[i][1])
        if i < num_args - 1:
            context.write(',\n')
%>
</%def>

<%def name='set_current_kernel(name)' filter='trim'>
<%
    kernel_arguments[name] = []
    kernel_arguments['current_kernel'] = name
    kernel_arguments['current_kernel_list'] = kernel_arguments[name]
%>
</%def>


###### Functions to determine what arguments are needed by kernels ######

<%def name='needs_bc_map(modifier="")' filter='trim'>
<%
    k = kernel_arguments['current_kernel_list']
    k.append(['bc_map', '__global '+modifier+' int *bc_map_global'])
    k.append(['nx_bc', 'const int nx_bc'])
    k.append(['ny_bc', 'const int ny_bc'])
    if dimension == 3:
        k.append(['nz_bc', 'const int nz_bc'])
%>
</%def>

<%def name='needs_bc_map_streamed(modifier="")' filter='trim'>
<%
    k = kernel_arguments['current_kernel_list']
    k.append(['bc_map', '__global '+modifier+' int *bc_map_streamed_global'])
%>
</%def>


<%def name='needs_f(modifier="")' filter='trim'>
<%
    k = kernel_arguments['current_kernel_list']
    k.append(['f', '__global '+modifier+' '+num_type+' *f_global'])
%>
</%def>


<%def name='needs_f_streamed(modifier="")' filter='trim'>
<%
    k = kernel_arguments['current_kernel_list']
    k.append(['f_streamed', '__global '+modifier+' '+num_type+' *f_streamed_global'])
%>
</%def>


<%def name='needs_feq(modifier="")' filter='trim'>
<%
    k = kernel_arguments['current_kernel_list']
    k.append(['feq', '__global '+modifier+' '+num_type+' *feq_global'])
%>
</%def>


<%def name='needs_rho(modifier="")' filter='trim'>
<%
    k = kernel_arguments['current_kernel_list']
    k.append(['rho', '__global '+modifier+' '+num_type+' *rho_global'])
%>
</%def>


<%def name='needs_absorbed_mass(modifier="")' filter='trim'>
<%
    k = kernel_arguments['current_kernel_list']
    k.append(['absorbed_mass', '__global '+modifier+' '+num_type+' *absorbed_mass_global'])
%>
</%def>


<%def name='needs_local_mem_num(name, modifier="")' filter='trim'>
<%
    k = kernel_arguments['current_kernel_list']
    k.append(['local_mem_num', '__local '+modifier+' '+num_type+' *'+name])
%>
</%def>


<%def name='needs_local_mem_int(name, modifier="")' filter='trim'>
<%
    k = kernel_arguments['current_kernel_list']
    k.append(['local_mem_int', '__local '+modifier+' '+num_type+' *'+name])
%>
</%def>


<%def name='needs_local_buf_size()' filter='trim'>
<%
    k = kernel_arguments['current_kernel_list']
    k.append(['buf_nx', 'const int buf_nx'])
    k.append(['buf_ny', 'const int buf_ny'])
    if dimension == 3:
        k.append(['buf_nz', 'const int buf_nz'])
%>
</%def>


<%def name='needs_k_list()' filter='trim'>
<%
    k = kernel_arguments['current_kernel_list']
    k.append(['k_list', '__constant '+num_type+' *k'])
%>
</%def>

<%def name='needs_m_reproduce_list()' filter='trim'>
<%
    k = kernel_arguments['current_kernel_list']
    k.append(['m_reproduce_list', '__constant '+num_type+' *m_reproduce'])
%>
</%def>

<%def name='needs_D()' filter='trim'>
<%
    k = kernel_arguments['current_kernel_list']
    k.append(['D', 'const '+num_type+' D'])
%>
</%def>


<%def name='needs_num_jumpers()' filter='trim'>
<%
    k = kernel_arguments['current_kernel_list']
    k.append(['num_jumpers', 'const int num_jumpers'])
%>
</%def>


<%def name='needs_omega()' filter='trim'>
<%
    k = kernel_arguments['current_kernel_list']

    k.append(['omega', 'const '+num_type+' omega'])
%>
</%def>


<%def name='needs_c_vec()' filter='trim'>
<%
    k = kernel_arguments['current_kernel_list']
    k.append(['c_vec', '__constant int *c_vec'])
%>
</%def>


<%def name='needs_c_mag()' filter='trim'>
<%
    k = kernel_arguments['current_kernel_list']
    k.append(['c_mag', '__constant '+num_type+' *c_mag'])
%>
</%def>


<%def name='needs_w()' filter='trim'>
<%
    k = kernel_arguments['current_kernel_list']
    k.append(['w', '__constant '+num_type+' *w'])
%>
</%def>

<%def name='needs_reflect_list()' filter='trim'>
<%
    k = kernel_arguments['current_kernel_list']
    k.append(['reflect_list', '__constant int *reflect_list'])
%>
</%def>


<%def name='needs_rand(modifier="")' filter='trim'>
<%
    k = kernel_arguments['current_kernel_list']
    k.append(['rand', '__global '+modifier+' '+num_type+' *rand_global'])
%>
</%def>