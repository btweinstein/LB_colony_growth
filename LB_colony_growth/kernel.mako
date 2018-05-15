<%def name='print_kernel_args(cur_kernel_list)' filter='trim'>
<%
    num_args = len(cur_kernel_list)
    for i in range(num_args):
        context.write('     ')
        context.write(cur_kernel_list[i][1])
        if i < num_args - 1:
            context.write(',\n')
%>
</%def>

<%def name='set_current_kernel(name)' filter='trim'>
<%
    kernel_arguments[name] = []
    cur_kernel_list = kernel_arguments[name]
%>
</%def>


###### Functions to determine what arguments are needed by kernels ######

<%def name='needs_bc_map(modifier="")' filter='trim'>
<%
    cur_kernel_list.append(['bc_map', '__global '+modifier+' int *bc_map_global'])
    cur_kernel_list.append(['nx_bc', 'const int nx_bc'])
    cur_kernel_list.append(['ny_bc', 'const int ny_bc'])
    if dimension == 3:
        cur_kernel_list.append(['nz_bc', 'const int nz_bc'])
%>
</%def>


<%def name='needs_f(modifier="")' filter='trim'>
<%
    cur_kernel_list.append(['f', '__global '+modifier+' '+num_type+' *f_global'])
%>
</%def>


<%def name='needs_f_streamed(modifier="")' filter='trim'>
<%
    cur_kernel_list.append(['f_streamed', '__global '+modifier+' '+num_type+' *f_streamed_global'])
%>
</%def>


<%def name='needs_feq(modifier="")' filter='trim'>
<%
    cur_kernel_list.append(['feq', '__global '+modifier+' '+num_type+' *feq_global'])
%>
</%def>


<%def name='needs_rho(modifier="")' filter='trim'>
<%
    cur_kernel_list.append(['rho', '__global '+modifier+' '+num_type+' *rho_global'])
%>
</%def>


<%def name='needs_absorbed_mass(modifier="")' filter='trim'>
<%
    cur_kernel_list.append(['absorbed_mass', '__global '+modifier+' '+num_type+' *absorbed_mass_global'])
%>
</%def>


<%def name='needs_local_mem_num(name, modifier="")' filter='trim'>
<%
    cur_kernel_list.append(['local_mem_num', '__local '+modifier+' '+num_type+' *'+name])
%>
</%def>


<%def name='needs_local_mem_int(name, modifier="")' filter='trim'>
<%
    cur_kernel_list.append(['local_mem_int', '__local '+modifier+' '+num_type+' *'+name])
%>
</%def>


<%def name='needs_local_buf_size()' filter='trim'>
<%
    cur_kernel_list.append(['buf_nx', 'const int buf_nx'])
    cur_kernel_list.append(['buf_ny', 'const int buf_ny'])
    if dimension == 3:
        cur_kernel_list.append(['buf_nz', 'const int buf_nz'])
%>
</%def>


<%def name='needs_k_list()' filter='trim'>
<%
    cur_kernel_list.append(['k_list', '__constant '+num_type+' *k'])
%>
</%def>


<%def name='needs_D()' filter='trim'>
<%
    cur_kernel_list.append(['D', 'const '+num_type+' D'])
%>
</%def>


<%def name='needs_num_jumpers()' filter='trim'>
<%
    cur_kernel_list.append(['num_jumpers', 'const int num_jumpers'])
%>
</%def>


<%def name='needs_omega()' filter='trim'>
<%
    cur_kernel_list.append(['omega', 'const '+num_type+' omega'])
%>
</%def>


<%def name='needs_c_vec()' filter='trim'>
<%
    cur_kernel_list.append(['c_vec', '__constant int *c_vec'])
%>
</%def>


<%def name='needs_c_mag()' filter='trim'>
<%
    cur_kernel_list.append(['c_mag', '__constant '+num_type+' *c_mag'])
%>
</%def>


<%def name='needs_w()' filter='trim'>
<%
    cur_kernel_list.append(['w', '__constant '+num_type+' *w'])
%>
</%def>

<%def name='needs_reflect_list()' filter='trim'>
<%
    cur_kernel_list.append(['reflect_list', '__constant int *reflect_list'])
%>
</%def>
