# -*- coding: utf-8 -*-
# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2014 and later, Alexander J G Pitchford
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################

# @author: Alexander Pitchford
# @email1: agp1@aber.ac.uk
# @email2: alex.pitchford@gmail.com
# @organization: Aberystwyth University
# @supervisor: Daniel Burgarth

"""
Control solver
"""
import os
import numpy as np
# QuTiP
from qutip import Qobj
# QuTiP logging
import qutip.logging_utils as logging
logger = logging.get_logger()

def _is_string(var):
    try:
        if isinstance(var, basestring):
            return True
    except NameError:
        try:
            if isinstance(var, str):
                return True
        except:
            return False
    except:
        return False

    return False


class ControlSolver(object):
    """

    Attributes
    ----------

    """
    #ToDo: Make this an abstract class

    def reset(self):
        self.evo_solver = None
        # self.drift_dyn_gen = None
        self.ctrl_dyn_gen = None
        self.initial = None
        self.target = None
        self.clear()

        self._num_ctrls = 0

    @property
    def num_ctrls(self):
        try:
            self._num_tslots = len(self.tlist)
        except:
            self._num_tslots = 0
        return self._num_tslots

    def apply_params(self, params=None):
        """
        Set object attributes based on the dictionary (if any) passed in the
        instantiation, or passed as a parameter
        This is called during the instantiation automatically.
        The key value pairs are the attribute name and value
        Note: attributes are created if they do not exist already,
        and are overwritten if they do.
        """
        if not params:
            params = self.params

        if isinstance(params, dict):
            self.params = params
            for key in params:
                setattr(self, key, params[key])

    @property
    def log_level(self):
        return logger.level

    @log_level.setter
    def log_level(self, lvl):
        """
        Set the log_level attribute and set the level of the logger
        that is call logger.setLevel(lvl)
        """
        logger.setLevel(lvl)

    def clear(self):
        pass

class ControlSolverPWC(ControlSolver):

    def __init__(self, evo_solver, fidelity_meter, tlist,
                 ctrl_dyn_gen=None, initial_amps=None):
        self.reset()
        self.evo_solver = evo_solver
        self.tlist = tlist
        self.ctrl_dyn_gen = ctrl_dyn_gen
        self.ctrl_amps = initial_amps

    def reset(self):
        ControlSolver.reset(self)
        self.tlist = None
        self.ctrl_amps = None

        self._num_tslots = 0
        self._total_time = 0.0

    @property
    def num_tslots(self):
        try:
            self._num_tslots = len(self.tlist)
        except:
            self._num_tslots = 0
        return self._num_tslots

    @property
    def total_time(self):
        try:
            self._total_time = sum(self.tlist)
        except:
            self._total_time = 0.0
        return self._num_tslots

    def init_solve(self):

        # Check attribute types and values
        pass

    def _get_optim_params(self):
        """Return the params to be optimised"""
        return self.ctrl_amps.ravel()

    def _set_ctrl_amp_params(self, optim_params):
        """Set the control amps based on the optimisation parameters"""
        try:
            self.ctrl_amps = optim_params.reshape([self._num_tslots,
                                                   self._num_ctrls])
        except ValueError as e:
            raise ValueError("Unable to set ctrl amplitude values: "
                            "{}".format(e))
