"""
@ Author       : jonnyzhang 71881972+jonnyzhang02@users.noreply.github.com
@ LastEditTime : 2024-09-12 10:15
@ FilePath     : /mmpretrain/my_config/vig_ours_dual_module_saracd_bs32_test_lr02_09121003.py
@ 
@ coded by ZhangYang@BUPT, my email is zhangynag0207@bupt.edu.cn
"""
_base_ = [
    './datasets/SAR-ACD-sbs32.py',
    './models/vig_ours.py',
    './schedules/bs64.py',
    './runtimes/runtime.py',
]