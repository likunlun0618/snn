def block(pre, cur, para, inplanes, planes, stride=1, downsample=None):
    if downsample is not None:
        print('// add %d next -> %d,%d'%(pre, cur, cur+8))
    else:
        print('// add %d next -> %d,%d'%(pre, cur, cur+8))

    print(f'{cur} pre({pre}) next({cur+1}) name(conv) options({inplanes},{planes},1,1,0,0) parameters({para})')
    print(f'{cur+1} pre({cur}) next({cur+2}) name(bn) options({planes},1e-5) parameters({para+1},{para+2},{para+3},{para+4})')
    print(f'{cur+2} pre({cur+1}) next({cur+3}) name(relu)')

    print(f'{cur+3} pre({cur+2}) next({cur+4}) name(conv) options({planes},{planes},3,{stride},1,0) parameters({para+5})')
    print(f'{cur+4} pre({cur+3}) next({cur+5}) name(bn) options({planes},1e-5) parameters({para+6},{para+7},{para+8},{para+9})')
    print(f'{cur+5} pre({cur+4}) next({cur+6}) name(relu)')

    print(f'{cur+6} pre({cur+5}) next({cur+7}) name(conv) options({planes},{planes*4},1,1,0,0) parameters({para+10})')

    if downsample is not None:
        print(f'{cur+7} pre({cur+6}) next({cur+10}) name(bn) options({planes*4},1e-5) parameters({para+11},{para+12},{para+13},{para+14})')

        print(f'{cur+8} pre({pre}) next({cur+9}) name(conv) options({inplanes},{planes*4},1,{stride},0,0) parameters({para+15})')
        print(f'{cur+9} pre({cur+8}) next({cur+10}) name(bn) options({planes*4},1e-5) parameters({para+16},{para+17},{para+18},{para+19})')
        print(f'{cur+10} pre({cur+7},{cur+9}) next({cur+11}) name(add)')
        print(f'{cur+11} pre({cur+10}) next({cur+12}) name(relu)')
        return cur+12, para+20
    else:
        print(f'{cur+7} pre({cur+6}) next({cur+8}) name(bn) options({planes*4},1e-5) parameters({para+11},{para+12},{para+13},{para+14})')

        print(f'{cur+8} pre({cur+7},{pre}) next({cur+9}) name(add)')
        print(f'{cur+9} pre({cur+8}) next({cur+10}) name(relu)')
        return cur+10, para+15


cur = 5
para = 5

cur, para = block(4, cur, para, 64, 64, downsample=True)
cur, para = block(cur-1, cur, para, 256, 64)
cur, para = block(cur-1, cur, para, 256, 64)

cur, para = block(cur-1, cur, para, 256, 128, stride=2, downsample=True)
cur, para = block(cur-1, cur, para, 512, 128)
cur, para = block(cur-1, cur, para, 512, 128)
cur, para = block(cur-1, cur, para, 512, 128)

cur, para = block(cur-1, cur, para, 512, 256, stride=2, downsample=True)
cur, para = block(cur-1, cur, para, 1024, 256)
cur, para = block(cur-1, cur, para, 1024, 256)
cur, para = block(cur-1, cur, para, 1024, 256)
cur, para = block(cur-1, cur, para, 1024, 256)
cur, para = block(cur-1, cur, para, 1024, 256)

cur, para = block(cur-1, cur, para, 1024, 512, stride=2, downsample=True)
cur, para = block(cur-1, cur, para, 2048, 512)
cur, para = block(cur-1, cur, para, 2048, 512)
