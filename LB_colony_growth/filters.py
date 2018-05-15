space4 = '    '

def wrap0(t):
    return t.replace('\n'+space4, '\n')

def wrap1(t):
    return t.replace('\n', '\n' + space4)


def wrap2(t):
    return t.replace('\n', '\n' + space4 + space4)


def wrap3(t):
    return t.replace('\n', '\n' + space4 + space4 + space4)


def wrap4(t):
    return t.replace('\n', '\n' + space4 + space4 + space4 + space4)