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
        self.fidelity_meter = None
        self.drift_dyn_gen = None
        self.ctrl_dyn_gen = None
        self.initial = None
        self.target = None
        self.clear()

        self._initialized = False
        self._ctrls_initialized = False
        self._num_ctrls = 0

    def clear(self):
        self.cost = np.inf

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

    @property
    def initialized(self):
        return self._initialized

    @property
    def ctrls_initialized(self):
        return self._ctrls_initialized

    @property
    def drift_dyn_gen(self):
        """Drift or 'system' dynamics generator, e.g Hamiltonian"""
        return self._get_drift_dyn_gen()

    def _get_drift_dyn_gen(self):
        if self.evo_solver is not None:
            return self.evo_solver.dyn_gen
        else:
            return None

    @property
    def num_ctrls(self):
        """Number of control operators"""
        return self._get_num_ctrls()

    def _get_num_ctrls(self):
        try:
            self._num_ctrls = len(self.ctrl_dyn_gen)
        except:
            self._num_ctrls = 0
        return self._num_ctrls

    def _get_ctrl_dyn_gen(self, t, j):
        # t parameter may be used in some overriding methods for some
        # time dependent ctrls
        return self.ctrl_dyn_gen[j]

    def _check_drift(self):
        # In separate function, as may be overridden
        # Assumes initial has already been checked
        if not isinstance(self.drift_dyn_gen, Qobj):
            raise TypeError("'drift_dyn_gen' must be a Qobj, check evo_solver")
        if not self.drift_dyn_gen.isoper:
            raise TypeError("'drift_dyn_gen' must be an operator, "
                            "check evo_solver")

        # This check is not valid for all evo_solvers
        # Should be moved to the evo_solver
#        if self.drift_dyn_gen.dims[1] != self.initial.dims[0]:
#                        raise TypeError("Incompatible quantum object dimensions "
#                                        "for 'drift_dyn_gen' and 'initial'")

    def _check_ctrls(self):
        # In separate function, as may be overridden
        # Assumes that _check_drift has already been called
        if self._num_ctrls == 0:
            logger.warning("No controls")
        else:
            for j, ctrl in enumerate(self.ctrl_dyn_gen):
                if not isinstance(ctrl, Qobj):
                    raise TypeError("'ctrl_dyn_gen[{}]' must be a "
                                    "Qobj".format(j))
                else:
                    if ctrl.dims != self.drift_dyn_gen.dims:
                        raise TypeError("Incompatible quantum object dimensions "
                                        "for 'ctrl_dyn_gen[{}]' and "
                                        "'drift_dyn_gen'".format(j))

    def init_solve(self):
        """
        Initialise the control solver
        Check all the attribute types and dimensional compatibility
        """
        self._get_num_ctrls()
        if not isinstance(self.initial, Qobj):
            raise TypeError("Attribute 'initial' must be a Qobj")

        if not isinstance(self.target, Qobj):
            raise TypeError("Attribute 'target' must be a Qobj")
        else:
            if self.target.dims != self.initial.dims:
                raise TypeError("Incompatible quantum object dimensions "
                                "for 'initial' and 'target'")

        self._check_drift()
        self._check_ctrls()

        # self._initialized not set here, as only considered initialised
        # when subclass init_solve has been called

class ControlSolverPWC(ControlSolver):

    def __init__(self, evo_solver, fidelity_meter, tlist,
                 ctrl_dyn_gen=None, initial_amps=None):
        self.reset()
        self.evo_solver = evo_solver
        self.fidelity_meter = fidelity_meter
        self.tlist = tlist
        self.ctrl_dyn_gen = ctrl_dyn_gen
        if initial_amps:
            self.init_ctrl_amps(initial_amps)

    def reset(self):
        ControlSolver.reset(self)
        self.tlist = None
        self.ctrl_amps = None
        self.fidelity_meter = None
        self._num_tslots = 0
        self._total_time = 0.0


    @property
    def num_tslots(self):
        """Number of timeslots"""
        return self._get_num_tslots()

    def _get_num_tslots(self):
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

        ControlSolver.init_solve(self)
        # Todo: Check attribute types and values

        # Initialise the containers
        self._get_num_tslots()
        #self._dyn_gen = [object for x in range(self._num_tslots)]
        #self._init_dyn_gen()

        self._initialized = True

    def _get_combined_dyn_gen(self, k):
        """Combine the drift and control dynamics generators for the timeslot"""
        dg = self.evo_solver.dyn_gen.copy()
        for j in self._num_ctrls:
            dg.data += self.ctrl_amps[k, j]*self._get_ctrl_dyn_gen(k, j).data
        return dg

    def _init_dyn_gen(self):

        self._dyn_gen = [self._get_combined_dyn_gen(k)
                            for k in range(self._num_tslots)]

    def _get_optim_params(self):
        """Return the params to be optimised"""
        return self.ctrl_amps.ravel()

    def init_ctrl_amps(self, ctrl_amps):
        """
        Set the control amps based on the optimisation parameters

        Parameters
        ---------
        ctrl_amps : array_like
            float valued array of inital control amplitudes
            Must be of shape (num_tslots, num_ctrls)
        """

        if not self._initialized:
            self.init_solver()

        if self._num_ctrls == 0:
            logger.warning("No controls")
            return

        try:
            ctrl_amps = np.array(ctrl_amps)
        except Exception as e:
            raise TypeError("Unable to set ctrl amplitude values: "
                            "{}".format(e))

        if (ctrl_amps.shape[0] != self._num_tslots or
            ctrl_amps.shape[1] != self._num_ctrls):
            raise ValueError("'ctrl_amps' must be of shape "
                            "(num_tslots, num_ctrls)")

        self.ctrl_amps = ctrl_amps
        self._init_dyn_gen()

        self._ctrls_initialized = True

    def _set_ctrl_amp_params(self, optim_params, changed_param_mask):
        """Set the control amps based on the optimisation parameters"""
        # Assumes that the shapes are compatible, as this will have been
        # tested in init_ctrl_amp_params
        self.ctrl_amps = optim_params.reshape([self._num_tslots,
                                               self._num_ctrls])

        changed_amp_mask = changed_param_mask.reshape([self._num_tslots,
                                               self._num_ctrls])

        for k in self._num_tslots:
            if np.any(changed_amp_mask[k, :]):
                self._dyn_gen[k] = self._get_combined_dyn_gen(k)


    def solve(self):
        """
        Solve the evolution with the PWC dynamics generators
        """
        # For now we will assume that this is the HEOM solver
        solres = self.evo_solver.run(self.initial, self.tlist, self._dyn_gen)
        self.cost = self.fidelity_meter(solres.states[-1], self.target)
