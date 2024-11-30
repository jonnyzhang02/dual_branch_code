"""
@ Author       : jonnyzhang 71881972+jonnyzhang02@users.noreply.github.com
@ LastEditTime : 2024-11-29 21:19
@ FilePath     : /mmpretrain/my_config/Final_vig_ours_dual_module_mstar_bs64_lr01.py
@ 
@ coded by ZhangYang@BUPT, my email is zhangynag0207@bupt.edu.cn
"""
_base_ = [
    './datasets/mstar.py',
    './models/vig_ours.py',
    './schedules/bs64.py',
    './runtimes/runtime.py',
]

model = dict(
    head = dict(
        num_classes=10
    )
)

custom_hooks = [
    dict(type='SavenpyHook', out_dir='work_dirs/vig_ours_mstar/')
]