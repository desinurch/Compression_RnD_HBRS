
import torch
import os

import models.WideResNet as WRN
import models.PyramidNet as PYN
import models.ResNet as RN
import models.PreResNet as PRN
import models.MobileNetv2 as MN

def load_paper_settings(args):

    WRN_path = os.path.join('./logs/teacher', 'WRN28-4_21.09.pt')
    Pyramid_path = os.path.join(args.data_path, 'pyramid200_mixup_15.6.tar')
    PreResNet_path = os.path.join('./logs/teacher','prn_best.pth.tar')
    ResNet_path = os.path.join('./logs/teacher','rn_best.pth.tar')

    if args.paper_setting == 'a':
        teacher = WRN.WideResNet(depth=28, widen_factor=4, num_classes=10)
        state = torch.load(WRN_path, map_location={'cuda:0': 'cpu'})['model']
        teacher.load_state_dict(state)
        student = WRN.WideResNet(depth=16, widen_factor=4, num_classes=10)

    elif args.paper_setting == 'b':
        teacher = WRN.WideResNet(depth=28, widen_factor=4, num_classes=10)
        state = torch.load(WRN_path, map_location={'cuda:0': 'cpu'})['model']
        teacher.load_state_dict(state)
        student = WRN.WideResNet(depth=28, widen_factor=2, num_classes=10)

    elif args.paper_setting == 'c':
        teacher = WRN.WideResNet(depth=28, widen_factor=4, num_classes=10)
        state = torch.load(WRN_path, map_location={'cuda:0': 'cpu'})['model']
        teacher.load_state_dict(state)
        student = WRN.WideResNet(depth=16, widen_factor=2, num_classes=10)

    elif args.paper_setting == 'd':
        teacher = WRN.WideResNet(depth=28, widen_factor=4, num_classes=100)
        state = torch.load(WRN_path, map_location={'cuda:0': 'cpu'})['model']
        teacher.load_state_dict(state)
        student = RN.ResNet(depth=56, num_classes=10)

    elif args.paper_setting == 'e':
        teacher = PYN.PyramidNet(depth=200, alpha=240, num_classes=10, bottleneck=True)
        state = torch.load(Pyramid_path, map_location={'cuda:0': 'cpu'})['state_dict']
        from collections import OrderedDict
        new_state = OrderedDict()
        for k, v in state.items():
            name = k[7:]  # remove 'module.' of dataparallel
            new_state[name] = v
        teacher.load_state_dict(new_state)
        student = WRN.WideResNet(depth=28, widen_factor=4, num_classes=10)

    elif args.paper_setting == 'f':
        teacher = PYN.PyramidNet(depth=200, alpha=240, num_classes=10, bottleneck=True)
        state = torch.load(Pyramid_path, map_location={'cuda:0': 'cpu'})['state_dict']
        from collections import OrderedDict
        new_state = OrderedDict()
        for k, v in state.items():
            name = k[7:]  # remove 'module.' of dataparallel
            new_state[name] = v
        teacher.load_state_dict(new_state)
        student = PYN.PyramidNet(depth=110, alpha=84, num_classes=10, bottleneck=False)

    elif args.paper_setting == 'g':
        teacher = PRN.preresnet(depth=110, num_classes=10)
        state = torch.load(PreResNet_path, map_location={'cuda:0': 'cpu'})['state_dict']
        from collections import OrderedDict
        new_state = OrderedDict()
        for k, v in state.items():
            name = k[7:]  # remove 'module.' of dataparallel
            new_state[name] = v
        teacher.load_state_dict(new_state)  
        # teacher.load_state_dict(new_state)
        student = PRN.preresnet(depth=56, num_classes=10)

    elif args.paper_setting == 'h':
        teacher = RN.ResNet(depth=56, num_classes=10, bottleneck=True)
        state = torch.load(ResNet_path, map_location={'cuda:0': 'cpu'})['state_dict']
        from collections import OrderedDict
        new_state = OrderedDict()
        for k, v in state.items():
            name = k[7:]  # remove 'module.' of dataparallel
            new_state[name] = v
        teacher.load_state_dict(new_state)
        student = RN.ResNet(depth=20, num_classes=10)

    else:
        print('Undefined setting name !!!')

    return teacher, student, args