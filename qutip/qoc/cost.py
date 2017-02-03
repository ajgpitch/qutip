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
Fidelity Computer

These classes calculate the fidelity error - function to be minimised
and fidelity error gradient, which is used to direct the optimisation

They may calculate the fidelity as an intermediary step, as in some case
e.g. unitary dynamics, this is more efficient

The idea is that different methods for computing the fidelity can be tried
and compared using simple configuration switches.

Note the methods in these classes were inspired by:
DYNAMO - Dynamic Framework for Quantum Optimal Control
See Machnes et.al., arXiv.1011.4874
The unitary dynamics fidelity is taken directly frm DYNAMO
The other fidelity measures are extensions, and the sources are given
in the class descriptions.
"""

import os
import warnings
import numpy as np
import scipy.sparse as sp
# import scipy.linalg as la
import timeit
# QuTiP
from qutip import Qobj
# QuTiP logging
import qutip.logging_utils as logging
logger = logging.get_logger()
# QuTiP control modules


class CostMeter(object):
    """
    Measures fidelity of evolved compared with the target

    Attributes
    ----------
    """
    #ToDo: Make abstract

    def __init__(self):
        self.reset()

    def reset(self):
        """
        reset any configuration data and
        clear any temporarily held status data
        """
        #FIXME: self.log_level = self.parent.log_level
        self.dimensional_norm = 1.0
        self.fid_norm_func = None
        self.grad_norm_func = None
        self.uses_onwd_evo = False
        self.uses_onto_evo = False
        #self.apply_params()
        self.clear()

    def clear(self):
        """
        clear any temporarily held status data
        """
        self.fid_err = None
        self.fidelity = None
        self.fid_err_grad = None
        self.grad_norm = np.inf
        self.fidelity_current = False
        self.fid_err_grad_current = False
        self.grad_norm = 0.0

#    def apply_params(self, params=None):
#        """
#        Set object attributes based on the dictionary (if any) passed in the
#        instantiation, or passed as a parameter
#        This is called during the instantiation automatically.
#        The key value pairs are the attribute name and value
#        Note: attributes are created if they do not exist already,
#        and are overwritten if they do.
#        """
#        if not params:
#            params = self.params
#
#        if isinstance(params, dict):
#            self.params = params
#            for key in params:
#                setattr(self, key, params[key])

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


class CostMeterUnitary(CostMeter):
    """
    Computes fidelity error and gradient assuming unitary dynamics, e.g.
    closed systems
    Note fidelity and gradient calculations were taken from DYNAMO
    (see file header)

    Attributes
    ----------
    ignore_global_phase : bool
        Differences between target and evolved state / gate due to global
        phase only are ignored when this is `True` (default)

    """

    def reset(self):
        CostMeter.reset(self)
        self.uses_onto_evo = True
        self.ignore_global_phase = True

    def init_meter(self, ctrl_solver):
        """
        Check configuration and initialise the normalisation
        """
        self.init_normalization(ctrl_solver)

    def init_normalization(self, ctrl_solver):
        """
        Calc norm of <Ufinal | Ufinal> to scale subsequent norms
        When considering unitary time evolution operators, this basically
        results in calculating the trace of the identity matrix
        and is hence equal to the size of the target matrix
        There may be situations where this is not the case, and hence it
        is not assumed to be so.
        """

        self.scale_factor = 1.0
        self.scale_factor = 1.0 / self._normalize(
                                ctrl_solver.target.dag()*ctrl_solver.target)

    def _normalize(self, A):
        """

        """
        if hasattr(A, 'shape'):
            norm = A.tr()

        if self.ignore_global_phase:
            return self.scale_factor * np.abs(norm)
        else:
            return self.scale_factor * np.real(norm)

    def compute_cost(self, final, target):
        """

        """
        f = (final.dag()*target).tr()
        self._fidelity_prenorm = f
        self.fidelity = self._normalize(f)
        self.cost = 1.0 - self.fidelity
        return self.cost

class CostMeterSqFrobDiff(CostMeter):
    """
    Computes fidelity error and gradient

    Attributes
    ----------


    """

    def reset(self):
        CostMeter.reset(self)
        self.uses_onto_evo = True

    def init_meter(self, ctrl_solver):
        """
        Check configuration and initialise the normalisation
        """
        self.init_normalization(ctrl_solver)

    def init_normalization(self, ctrl_solver):
        """
        Calc norm of <Ufinal | Ufinal> to scale subsequent norms
        When considering unitary time evolution operators, this basically
        results in calculating the trace of the identity matrix
        and is hence equal to the size of the target matrix
        There may be situations where this is not the case, and hence it
        is not assumed to be so.
        """

        self.scale_factor = 1.0
        self.scale_factor = 1.0 / self._normalize(
                                2.0*ctrl_solver.target.dag()*ctrl_solver.target)

    def _normalize(self, A):
        """

        """
        if hasattr(A, 'shape'):
            norm = A.tr()

        return self.scale_factor * np.real(norm)

    def compute_cost(self, final, target):
        """

        """
        diff = target - final
        self.cost = self._normalize(diff.dag()*diff)
        return self.cost
