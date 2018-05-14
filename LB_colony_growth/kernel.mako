
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