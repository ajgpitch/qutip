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
# QuTiP Control
import qutip.qoc.cost as qoccost

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
    #TODO: Make this an abstract class

    def __init__(self, evo_solver, cost_meter, initial, target, ctrl_dyn_gen):
        #TODO: Check type of solver
        self.evo_solver = evo_solver
        if not isinstance(cost_meter, qoccost.CostMeter):
            raise TypeError("Invalid type {} for 'cost_meter'. Must be of type "
                            "{}".format(type(cost_meter), qoccost.CostMeter))
        self.cost_meter = cost_meter
        self._check_initial(initial)
        self.initial = initial
        self._check_target(target)
        self.target = target
        self._check_ctrls(ctrl_dyn_gen)
        self.ctrl_dyn_gen = ctrl_dyn_gen

    def reset(self):
        self.evo_solver = None
        self.cost_meter = None
        self._drift_dyn_gen = None
        self.ctrl_dyn_gen = None
        self.initial = None
        self.target = None
        self.clear()

        self._solve_initialized = False
        self._num_ctrls = 0

    def clear(self):
        self.cost = None

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
    def solve_initialized(self):
        return self._initialized

    @property
    def ctrls_initialized(self):
        return self._ctrls_initialized

    @property
    def drift_dyn_gen(self):
        """Drift or 'system' dynamics generator, e.g Hamiltonian"""
        return self._get_drift_dyn_gen()

    def _get_drift_dyn_gen(self):
        self._drift_dyn_gen = self.evo_solver.dyn_gen
        return self._drift_dyn_gen

    @property
    def num_ctrls(self):
        """Number of control operators"""
        return self._get_num_ctrls()

    def _get_num_ctrls(self, ctrl_dyn_gen=None):
        if ctrl_dyn_gen is None:
            ctrl_dyn_gen = self.ctrl_dyn_gen
        try:
            self._num_ctrls = len(ctrl_dyn_gen)
        except:
            self._num_ctrls = 0
        return self._num_ctrls

    def _get_ctrl_dyn_gen(self, t, j):
        # t parameter may be used in some overriding methods for some
        # time dependent ctrls
        return self.ctrl_dyn_gen[j]

    def _check_initial(self, initial=None):
        # In separate function, as may be overridden
        desc = 'parameter'
        if initial is None:
            initial = self.initial
            desc = 'attribute'

        if not isinstance(initial, Qobj):
            raise TypeError("Invalid type {} for {} 'initial'. "
                            "Must be of type {}.".format(type(initial),
                                                        desc, Qobj))

    def _check_target(self, target=None):
        # In separate function, as may be overridden
        # Assumes initial has already been checked (and set)
        desc = 'parameter'
        if target is None:
            target = self.target
            desc = 'attribute'

        if not isinstance(target, Qobj):
            raise TypeError("Invalid type {} for {} 'target'. "
                            "Must be of type {}.".format(type(target),
                                                        desc, Qobj))

        if target.dims != self.initial.dims:
            raise TypeError("Incompatible quantum object dimensions "
                            "for 'initial' and 'target'")

    def _check_drift(self, drift_dyn_gen=None):
        # In separate function, as may be overridden
        # Assumes initial has already been checked
        desc = 'parameter'
        if drift_dyn_gen is None:
            drift_dyn_gen = self.drift_dyn_gen
            desc = 'attribute'

        if not isinstance(drift_dyn_gen, Qobj):
            raise TypeError("Invalid type {} for {} 'drift_dyn_gen'. "
                            "Must be of type {}. Check "
                            "evo_solver.dyn_gen".format(type(drift_dyn_gen),
                                                        desc, Qobj))
        if not drift_dyn_gen.isoper:
            raise TypeError("'drift_dyn_gen' must be an operator, "
                            "check evo_solver")

        # This check is not valid for all evo_solvers
        # Should be moved to the evo_solver
#        if self.drift_dyn_gen.dims[1] != self.initial.dims[0]:
#                        raise TypeError("Incompatible quantum object dimensions "
#                                        "for 'drift_dyn_gen' and 'initial'")

    def _check_ctrls(self, ctrl_dyn_gen=None):
        # In separate function, as may be overridden
        # Assumes that _check_drift has already been called
        desc = 'parameter'
        if ctrl_dyn_gen is None:
            ctrl_dyn_gen = self.ctrl_dyn_gen
            desc = 'attribute'

        if self._get_num_ctrls(ctrl_dyn_gen) == 0:
            raise TypeError("Invalid type {} for {} 'ctrl_dyn_gen'. "
                            "Must be iterable.".format(type(ctrl_dyn_gen), desc))
        else:
            for j, ctrl in enumerate(ctrl_dyn_gen):
                if not isinstance(ctrl, Qobj):
                    raise TypeError("Invalid type {} for 'ctrl_dyn_gen[{}]'. "
                                    "Must be a {}.".format(type(ctrl), j, Qobj))
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

        self._check_initial()
        self._check_drift()
        self.cost_meter.init_normalization(self)
        self._check_drift()
        self._check_ctrls()

        # self._initialized not set here, as only considered initialised
        # when subclass init_solve has been called

    @property
    def is_solution_current(self):
        # Abstract
        return False

class ControlSolverPWC(ControlSolver):

    def __init__(self, evo_solver, cost_meter, initial, target, ctrl_dyn_gen,
                 tlist, initial_amps=None):
        self.reset()
        ControlSolver.__init__(self, evo_solver, cost_meter, initial, target,
                               ctrl_dyn_gen)
        self._check_tlist(tlist)
        self.tlist = tlist
        #TODO: Check ctrl amps
        self.ctrl_amps = initial_amps

    def reset(self):
        ControlSolver.reset(self)
        self.tlist = None
        self.ctrl_amps = None
        self.cost_meter = None
        self._num_tslots = 0
        self._total_time = 0.0
        self._changed_amp_mask = None


    @property
    def num_tslots(self):
        """Number of timeslots"""
        return self._get_num_tslots()

    def _check_tlist(self, tlist=None):
        # In separate function, as may be overridden
        # Assumes that _check_drift has already been called
        desc = 'parameter'
        if tlist is None:
            tlist = self.tlist
            desc = 'attribute'

        if not hasattr(tlist, '__iter__'):
            raise TypeError("Invalid type {} for {} 'tlist'. "
                            "Must be iterable.".format(type(tlist), desc))

        if self._get_num_tslots(tlist) == 0:
            raise ValueError("Invalid value {} for {} 'tlist'. Must define "
                            "at least one timeslot.".format(type(tlist), desc))

        if self._get_total_time(tlist) == 0.0:
            raise TypeError("total time cannot be zero")

    def _get_num_tslots(self, tlist=None):
        if tlist is None:
            tlist = self.tlist

        try:
            self._num_tslots = len(tlist) - 1
        except:
            self._num_tslots = 0
        return self._num_tslots

    @property
    def total_time(self):
        return self._get_total_time()

    def _get_total_time(self, tlist=None):
        if tlist is None:
            tlist = self.tlist
        try:
            self._total_time = tlist[-1]
        except:
            self._total_time = 0.0
        return self._total_time

    def _get_combined_dyn_gen(self, k):
        """Combine the drift and control dynamics generators for the timeslot"""
        dg = self._drift_dyn_gen.copy()
        for j in range(self._num_ctrls):
            dg.data += self.ctrl_amps[k, j]*self._get_ctrl_dyn_gen(k, j).data
        return dg

    def _init_dyn_gen(self):

        self._dyn_gen = [self._get_combined_dyn_gen(k)
                            for k in range(self._num_tslots)]

    def _get_optim_params(self):
        """Return the params to be optimised"""
        return self.ctrl_amps.ravel()

    def init_solve(self, ctrl_amps=None):
        """
        Set the control amps based on the optimisation parameters

        Parameters
        ---------
        ctrl_amps : array_like
            float valued array of inital control amplitudes
            Must be of shape (num_tslots, num_ctrls)
        """

        #TODO: Add skip checks

        ControlSolver.init_solve(self)
        self._check_tlist()

        if ctrl_amps is None:
            ctrl_amps = self.ctrl_amps

        try:
            ctrl_amps = np.array(ctrl_amps)
        except Exception as e:
            raise TypeError("Unable to set ctrl amplitude values: "
                            "{}".format(e))

        if (len(ctrl_amps.shape) != 2 or
            ctrl_amps.shape[0] != self.num_tslots or
            ctrl_amps.shape[1] != self.num_ctrls):
            try:
                ctrl_amps = ctrl_amps.reshape([self._num_tslots,
                                               self._num_ctrls])
            except Exception as e:
                raise ValueError("Incorrect shape {} for 'ctrl_amps'. "
                                "Must be of shape, or reshapeable to, "
                                "(num_tslots, num_ctrls)="
                                "({}, {})".format(ctrl_amps.shape,
                                                  self._num_tslots,
                                                  self._num_ctrls))

        self.ctrl_amps = ctrl_amps
        self._init_dyn_gen()

        self._solve_initialized = True

    def _set_ctrl_amp_params(self, optim_params, chg_mask=None):
        """Set the control amps based on the optimisation parameters"""
        # Assumes that the shapes are compatible, as this will have been
        # tested in init_ctrl_amp_params
        self.ctrl_amps = optim_params.reshape([self._num_tslots,
                                               self._num_ctrls])

        if chg_mask is None:
            self._changed_amp_mask = None
        else:
            self._changed_amp_mask = chg_mask.reshape([self._num_tslots,
                                                       self._num_ctrls])

    def _update_dyn_gen(self):
        for k in range(self._num_tslots):
            if (self._changed_amp_mask is None
                    or np.any(self._changed_amp_mask[k, :])):
                self._dyn_gen[k] = self._get_combined_dyn_gen(k)


    @property
    def is_solution_current(self):
        if self.cost is None:
            return False
        if self._changed_amp_mask is None:
            return True
        if np.any(self._changed_amp_mask):
            return False
        else:
            return True

    def solve(self, skip_init=False):
        """
        Solve the evolution with the PWC dynamics generators
        """
        if not self._solve_initialized and not skip_init:
            self.init_solve()

        self._update_dyn_gen()

        #FIXME: For now we will assume that this is the HEOM solver
        solres = self.evo_solver.run(self.initial, self.tlist, self._dyn_gen)
        self.cost = self.cost_meter.compute_cost(solres.states[-1], self.target)
        return self.cost
