def hg(pre_id, next_id, node_id, param_id):
    # 添加前驱节点指向第一次下采样前、最外层的skip的residual的第一个节点和add节点
    print('// add %d next -> %d,%d,%d'%(pre_id, node_id, node_id+131, node_id+140))

    print('%d pre(%d) next(%d,%d) name(maxpool) options(2)'%(node_id, pre_id, node_id+1, node_id+10))

    ret = residual(node_id, [node_id+11, node_id+109, node_id+118], node_id+1, param_id, 256, 256)
    for item in ret:
        print(item)

    print('%d pre(%d) next(%d,%d) name(maxpool) options(2)'%(node_id+11, node_id+10, node_id+12, node_id+21))

    ret = residual(node_id+11, [node_id+22, node_id+87, node_id+96], node_id+12, param_id+18, 256, 256)
    for item in ret:
        print(item)

    print('%d pre(%d) next(%d,%d) name(maxpool) options(2)'%(node_id+22, node_id+21, node_id+23, node_id+32))

    ret = residual(node_id+22, [node_id+33, node_id+65, node_id+74], node_id+23, param_id+36, 256, 256)
    for item in ret:
        print(item)

    print('%d pre(%d) next(%d,%d) name(maxpool) options(2)'%(node_id+33, node_id+32, node_id+34, node_id+43))

    #　最底层的３个residual
    ret = residual(node_id+33, [node_id+44, node_id+53], node_id+34, param_id+54, 256, 256)
    for item in ret:
        print(item)

    ret = residual(node_id+43, [node_id+54, node_id+63], node_id+44, param_id+72, 256, 256)
    for item in ret:
        print(item)

    ret = residual(node_id+53, node_id+64, node_id+54, param_id+90, 256, 256)
    for item in ret:
        print(item)

    print('%d pre(%d) next(%d) name(upsample) options(2)'%(node_id+64, node_id+63, node_id+75))

    ret = residual(node_id+32, node_id+75, node_id+65, param_id+108, 256, 256)
    for item in ret:
        print(item)

    print('%d pre(%d,%d) next(%d,%d) name(add)'%(node_id+75, node_id+64, node_id+74, node_id+76, node_id+85))

    ret = residual(node_id+75, node_id+86, node_id+76, param_id+126, 256, 256)
    for item in ret:
        print(item)

    print('%d pre(%d) next(%d) name(upsample) options(2)'%(node_id+86, node_id+85, node_id+97))

    ret = residual(node_id+21, node_id+97, node_id+87, param_id+144, 256, 256)
    for item in ret:
        print(item)

    print('%d pre(%d,%d) next(%d,%d) name(add)'%(node_id+97, node_id+86, node_id+96, node_id+98, node_id+107))

    ret = residual(node_id+97, node_id+108, node_id+98, param_id+162, 256, 256)
    for item in ret:
        print(item)

    print('%d pre(%d) next(%d) name(upsample) options(2)'%(node_id+108, node_id+107, node_id+119))

    ret = residual(node_id+10, node_id+119, node_id+109, param_id+180, 256, 256)
    for item in ret:
        print(item)

    print('%d pre(%d,%d) next(%d,%d) name(add)'%(node_id+119, node_id+108, node_id+118, node_id+120, node_id+129))

    ret = residual(node_id+119, node_id+130, node_id+120, param_id+198, 256, 256)
    for item in ret:
        print(item)

    print('%d pre(%d) next(%d) name(upsample) options(2)'%(node_id+130, node_id+129, node_id+141))

    ret = residual(pre_id, node_id+141, node_id+131, param_id+216, 256, 256)
    for item in ret:
        print(item)

    print('%d pre(%d,%d) next(%d,%d) name(add)'%(node_id+141, node_id+130, node_id+140, node_id+142, node_id+151))

    ret = residual(node_id+141, node_id+152, node_id+142, param_id+234, 256, 256)
    for item in ret:
        print(item)

    print('%d pre(%d) next(%d) name(conv) options(256,256,1,1,0,1) parameters(%d,%d)'%(
        node_id+152, node_id+151, node_id+153, param_id+252, param_id+253))

    print('%d pre(%d) next(%d) name(bn) options(256,1e-5) parameters(%d,%d,%d,%d)'%(
        node_id+153, node_id+152, node_id+154, param_id+254, param_id+255, param_id+256, param_id+257))

    print('%d pre(%d) next(%d) name(relu)'%(node_id+154, node_id+153, node_id+155))




def residual(pre_id, next_id, node_id, param_id, inp, out):

    skip = (inp != out)

    s1 = '%d pre(%d) next(%d) name(bn) options(%d,1e-5) parameters(%d,%d,%d,%d)'%(
        node_id, pre_id, node_id+1, inp, param_id, param_id+1, param_id+2, param_id+3)
    s2 = '%d pre(%d) next(%d) name(relu)'%(node_id+1, node_id, node_id+2)
    s3 = '%d pre(%d) next(%d) name(conv) options(%d,%d,1,1,0,1) parameters(%d,%d)'%(
        node_id+2, node_id+1, node_id+3, inp, out//2, param_id+4, param_id+5)
    s4 = '%d pre(%d) next(%d) name(bn) options(%d,1e-5) parameters(%d,%d,%d,%d)'%(
        node_id+3, node_id+2, node_id+4, out//2, param_id+6, param_id+7, param_id+8, param_id+9)
    s5 = '%d pre(%d) next(%d) name(relu)'%(node_id+4, node_id+3, node_id+5)
    s6 = '%d pre(%d) next(%d) name(conv) options(%d,%d,3,1,1,1) parameters(%d,%d)'%(
        node_id+5, node_id+4, node_id+6, out//2, out//2, param_id+10, param_id+11)
    s7 = '%d pre(%d) next(%d) name(bn) options(%d,1e-5) parameters(%d,%d,%d,%d)'%(
        node_id+6, node_id+5, node_id+7, out//2, param_id+12, param_id+13, param_id+14, param_id+15)
    s8 = '%d pre(%d) next(%d) name(relu)'%(node_id+7, node_id+6, node_id+8)

    if skip:
        skip = 1
    '''
    if next_id is None:
        next_id = node_id + 10 + skip
    '''
    if type(next_id) == int:
        next_id = [next_id]
    next_id_s = '%d'%next_id[0]
    for i in range(1, len(next_id)):
        next_id_s += ',%d'%next_id[i]

    s9 = '%d pre(%d) next(%d) name(conv) options(%d,%d,1,1,0,1) parameters(%d,%d)'%(
        node_id+8, node_id+7, node_id+9+skip, out//2, out, param_id+16, param_id+17)
    ret = [s1, s2, s3, s4, s5, s6, s7, s8, s9]
    if skip:
        tmp = '%d pre(%d) next(%d) name(conv) options(%d,%d,1,1,0,1) parameters(%d,%d)'%(
            node_id+9, pre_id, node_id+10, inp, out, param_id+18, param_id+19)
        ret.append(tmp)
        s10 = '%d pre(%d,%d) next(%s) name(add)'%(node_id+10, node_id+8, node_id+9, next_id_s)
        ret.append(s10)
    else:
        s10 = '%d pre(%d,%d) next(%s) name(add)'%(node_id+9, pre_id, node_id+8, next_id_s)
        ret.append(s10)
    print('// add %d next -> %d, %d'%(pre_id, node_id, node_id+9))
    return ret



hg(513, None, 514, 856)
