from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class TCL_Options:
    def __init__(self):
        """
        Here are the parameters related to the GC and AiC  modules.
        Please add any additional necessary parameters based on your specific requirements.
        Please refer to the suggestions in the README file for guidance and experimentation when adjusting the parameter values.
        """

        self.parser = argparse.ArgumentParser(description="TCL options without exact value")


        self.parser.add_argument('-dw', '--depth_wl_weight', type=float, default=1e-3, help='depth consistency weight')
        self.parser.add_argument('-pw', '--pose_wl_weight', type=float, default=5e-1, help='pose consistency weight')
        self.parser.add_argument('--w_trans', type=float, default=1.0, help='pose consistency trans part weight')

        self.parser.add_argument('-dcwarm', '--dc_warm', type=int, default=5,
                                 help='depth consistency warm up epoch nums')
        self.parser.add_argument('-pcwarm', '--pc_warm', type=int, default=5,
                                 help='pose consistency warm up epoch nums')

        self.parser.add_argument('-pb', '--bound', type=float, default=0.2, help='synthesis triplet:pose perturbance bonud')

        self.parser.add_argument('-illunum', '--illu_listnum', type=int, default=0, help='feature list index for AiC')

        self.parser.add_argument('-tmthre', '--tm_thre', type=float, default=0.5, help='triplet mask lower bound ')
        self.parser.add_argument('-illuw', '--illu_ssimweight', type=float, default=0.8, help='triplet mask ssim weight')
