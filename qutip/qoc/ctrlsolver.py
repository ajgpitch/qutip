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
import qutip.solver
from qutip.rhs_generate import rhs_clear
from qutip.cy.utilities import _cython_build_cleanup
# QuTiP logging
import qutip.settings as qset
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

    # Define these as a class attribute, as it is used in the __del__
    _integ_tdname = None
    integ_rhs_tidyup = True

    def __init__(self, evo_solver, cost_meter, initial, target,
                 drift_dyn_gen, ctrl_dyn_gen):
        #TODO: Check type of solver
        self.evo_solver = evo_solver
        if not isinstance(cost_meter, qoccost.CostMeter):
            raise TypeError("Invalid type {} for 'cost_meter'. Must be of type "
                            "{}".format(type(cost_meter), qoccost.CostMeter))
        self.cost_meter = cost_meter
        self.initial = self._check_initial(initial)
        self.target = self._check_target(target)
        self.drift_dyn_gen = self._check_drift(drift_dyn_gen)
        self.ctrl_dyn_gen = self._check_ctrls(ctrl_dyn_gen)

    def __del__(self):
        if self.integ_rhs_tidyup:
            self.tidyup_integ_td()

    def tidyup_integ_td(self):
        if self._integ_tdname is not None:
            print("cleaning up: {}".format(self._integ_tdname))
            _cython_build_cleanup(self._integ_tdname)
            self._integ_tdname = None
        rhs_clear()

    def reset(self):
        self.evo_solver = None
        self.cost_meter = None
        self.drift_dyn_gen = None
        self.ctrl_dyn_gen = None
        self.initial = None
        self.target = None
        self.clear()

        self._initialized = False
        self._num_ctrls = 0
        self._dyn_gen_dims = None

        self.integ_rhs_reuse = True
        self.integ_rhs_tidyup = True
        self._integ_tdname = None

    def clear(self):
        self.evo_solver_result = None
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

#    @property
#    def drift_dyn_gen(self):
#        """Drift or 'system' dynamics generator, e.g Hamiltonian"""
#        return self._get_drift_dyn_gen()

    def _get_drift_dyn_gen(self):
        return self.drift_dyn_gen

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
        self._dyn_gen_dims = [initial.dims[0], initial.dims[0]]
        return initial

    def _check_dg(self, dg, name, dims=None):
        self._check_dg_oper(dg, name, dims)

    def _check_dg_oper(self, dg, name, dims=None):
        """Check dynamics generator operator"""

        if not isinstance(dg, Qobj):
            raise TypeError("Invalid type {} for {}. "
                            "Must be of type {}.".format(type(dg),
                                                        name, Qobj))
        if not dg.isoper:
            raise TypeError("Invalid Qobj type {} for {}. "
                            "Must be 'oper' or 'super'.".format(dg.type, name))

        if dims is not None:
            if dg.dims != dims:
                raise ValueError("Invalid dims {} for {}. "
                                "Must be compatible with initial oper / state "
                                "dims.".format(dg.dims, dims))

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
            raise TypeError("Incompatible Qobj dimensions "
                            "for 'initial' and 'target'")
        return target

    def _check_drift(self, drift_dyn_gen=None):
        # In separate function, as may be overridden
        # Assumes initial has already been checked
        desc = 'parameter'
        if drift_dyn_gen is None:
            drift_dyn_gen = self.drift_dyn_gen
            desc = 'attribute'

        self._check_dg(drift_dyn_gen,
                            "{} '{}'".format(desc, 'drift_dyn_gen'),
                            dims=self._dyn_gen_dims)

        # This check is not valid for all evo_solvers
        # Should be moved to the evo_solver
#        if self.drift_dyn_gen.dims[1] != self.initial.dims[0]:
#                        raise TypeError("Incompatible quantum object dimensions "
#                                        "for 'drift_dyn_gen' and 'initial'")
        return drift_dyn_gen

    def _check_ctrls(self, ctrl_dyn_gen=None):
        # In separate function, as may be overridden
        # Assumes that _check_drift has already been called
        desc = 'parameter'
        if ctrl_dyn_gen is None:
            ctrl_dyn_gen = self.ctrl_dyn_gen
            desc = 'attribute'

        if self._get_num_ctrls(ctrl_dyn_gen) == 0:
            raise TypeError("Invalid type {} for {} 'ctrl_dyn_gen'. Must "
                            "be iterable.".format(type(ctrl_dyn_gen), desc))
        else:
            for j, ctrl in enumerate(ctrl_dyn_gen):
                self._check_dg(ctrl,
                               "{} '{}[{}]'".format(desc, 'ctrl_dyn_gen', j),
                               dims=self._dyn_gen_dims)
        return ctrl_dyn_gen

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

    def __init__(self, evo_solver, cost_meter, initial, target,
                 drift_dyn_gen, ctrl_dyn_gen,
                 tslot_duration, tlist=None, initial_amps=None,
                 solver_combines_dyn_gen=True):
        self.reset()
        ControlSolver.__init__(self, evo_solver, cost_meter, initial, target,
                               drift_dyn_gen, ctrl_dyn_gen)
        self.tslot_duration = self._check_tslot_duration(tslot_duration)
        self.tlist = self._check_tlist(tlist)
        #TODO: Check ctrl amps
        self.ctrl_amps = self._check_ctrl_amps(initial_amps)
        # The plan is to use the solver internal combining of
        # dynamics generators
        self.solver_combines_dyn_gen = solver_combines_dyn_gen

    def reset(self):
        ControlSolver.reset(self)
        #TODO: Switch to property setters?
        self.tslot_duration = None
        self.tlist = None
        self.ctrl_amps = None
        self.cost_meter = None
        self._num_tslots = 0
        self._total_time = 0.0
        self._tslot_time = None
        self._changed_amp_mask = None

    @property
    def num_tslots(self):
        """Number of tslot_duration"""
        return self._get_num_tslots()

    def _check_tslot_duration(self, tslot_duration=None):
        desc = 'parameter'
        if tslot_duration is None:
            tslot_duration = self.tslot_duration
            desc = 'attribute'

        try:
            tslot_duration = np.array(tslot_duration)
        except Exception as e:
            raise TypeError("Invalid type {} for {} 'tslot_duration'. "
                            "Must be array_like. Attempt at array raised: "
                            "{}".format(type(tslot_duration), desc, e))

        if len(tslot_duration.shape) != 1:
            raise ValueError("Invalid shape {} for {} 'tslot_duration'. "
                            "Must be 1 dim.".format(tslot_duration.shape,
                                                    desc))

        if self._get_num_tslots(tslot_duration) == 0:
            raise ValueError("Invalid  {} 'tslot_duration'. Must define at "
                            "least one timeslot.".format(
                                                    type(tslot_duration), desc))

        if self._get_total_time(tslot_duration) == 0.0:
            raise TypeError("total time cannot be zero")

        self._tslot_time = np.insert(np.cumsum(tslot_duration), 0, 0.0)

        return tslot_duration

    @property
    def tslot_time(self):
        return self._tslot_time


    def _check_tlist(self, tlist=None):
        # Assumes that _check_tslot_duration has already been called
        desc = 'parameter'
        if tlist is None:
            tlist = self.tlist
            desc = 'attribute'

        if tlist is None:
            return np.insert(np.cumsum(self.tslot_duration), 0, 0.0)

        try:
            tlist = np.array(tlist, dtype='f')
        except Exception as e:
            raise TypeError("Invalid type {} for {} 'tlist'. "
                            "Must be array_like. Attempt at array(tlist) "
                            "raised: {}".format(type(tlist), desc, e))

        end_time = tlist[-1]
        if abs(end_time - self._get_total_time()) > qset.atol:
            raise ValueError("Invalid end time {} for {} 'tlist'. "
                            "Must be equal to the timeslot total time "
                            "{}".format(end_time, desc, self._get_total_time()))

        # The number of timeslots in the tlist must be a multiple of the
        # number of timeslots for the controls
        nts = len(tlist) - 1
        if nts % self._num_tslots != 0:
            nts = (nts//self._num_tslots + 1)*self._num_tslots
            tlist = np.linspace(0.0, end_time, nts+1)

        return tlist

    def _get_num_tslots(self, tslot_duration=None):
        if tslot_duration is None:
            tslot_duration = self.tslot_duration

        try:
            self._num_tslots = len(tslot_duration)
        except:
            self._num_tslots = 0
        return self._num_tslots

    def _check_ctrl_amps(self, ctrl_amps=None):
        # In separate function, as may be overridden
        # Assumes that _check_tslot_duration has already been called
        desc = 'parameter'
        if ctrl_amps is None:
            ctrl_amps = self.ctrl_amps
            desc = 'attribute'

        try:
            ctrl_amps = np.array(ctrl_amps)
        except Exception as e:
            raise TypeError("Invalid type {} for {} 'ctrl_amps'. "
                            "Must be array_like. Attempt at array raised: "
                            "{}".format(type(ctrl_amps), desc, e))

        if (len(ctrl_amps.shape) != 2 or
            ctrl_amps.shape[0] != self.num_tslots or
            ctrl_amps.shape[1] != self.num_ctrls):
            try:
                ctrl_amps = ctrl_amps.reshape([self._num_tslots,
                                               self._num_ctrls])
            except Exception as e:
                raise ValueError("Incorrect shape {} for {} 'ctrl_amps'. "
                                "Must be of shape, or reshapeable to, "
                                "(num_tslots, num_ctrls)="
                                "({}, {})".format(ctrl_amps.shape, desc,
                                                  self._num_tslots,
                                                  self._num_ctrls))

        return ctrl_amps

    @property
    def total_time(self):
        return self._get_total_time()

    def _get_total_time(self, tslot_duration=None):
        if tslot_duration is None:
            tslot_duration = self.tslot_duration
        try:
            self._total_time = np.sum(tslot_duration)
        except:
            self._total_time = 0.0
        return self._total_time

    def _get_combined_dyn_gen(self, k):
        """Combine the drift and control dynamics generators for the timeslot"""
        dg = self._get_drift_dyn_gen.copy()
        for j in range(self._num_ctrls):
            dg.data += self.ctrl_amps[k, j]*self._get_ctrl_dyn_gen(k, j).data
        return dg

    def _init_dyn_gen(self):

        if self.solver_combines_dyn_gen:
            self.evo_solver.dyn_gen = self._construct_td_dyn_gen()
        else:
            self._dyn_gen = [self._get_combined_dyn_gen(k)
                                for k in range(self._num_tslots)]

    def _check_dg(self, dg, name, dims=None):
        #NOTE: This overrides ControlSolver method

        if isinstance(dg, Qobj):
            # No time dependance
            self._check_dg_oper(dg, name, dims)
        elif isinstance(dg, list):
            if len(dg) != 2:
                raise TypeError("Invalid td format for {}".format(name))
                self._check_dg_oper(dg[0], name, dims)
            if not _is_string(dg[1]):
                raise TypeError("Invalid td format for {}. Only string type "
                                 "td is supported by the "
                                 "ctrl solver.".format(name))
        else:
            raise TypeError("Invalid type for {}".format(name))

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
        self.tslot_duration = self._check_tslot_duration()
        self.ctrl_amps = self._check_ctrl_amps(ctrl_amps)
        self._init_dyn_gen()

        if self.evo_solver is not None:
            if self.evo_solver.options is None:
                self.evo_solver.options = qutip.solver.Options(
                                            rhs_reuse=self.integ_rhs_reuse)
            else:
                self.evo_solver.options.rhs_reuse = self.integ_rhs_reuse

        self._initialized = True

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

    def _get_td_dyn_gen(self, t, args):
        """Time dependent Hamiltonian function for solver"""

        # get time slot
        k = np.where(self._tslot_time <= t)[0][-1]
        #print("time: {}".format(t))
        return self._dyn_gen[k]

    def _build_td_dg(self, dg, j=-1):
        """Specific dynamics generator in str type td format"""

        # assumes dyn gen opers have been checked for format
        dg_coeff = None
        if isinstance(dg, Qobj):
            # No other time dependance
            dg_op = dg
        elif isinstance(dg, list):
            dg_op = dg[0]
            dg_coeff = dg[1]
        else:
            # this should never happen, as should have been checked
            raise TypeError("Invalid type for ctrl "
                             "dynamics generator {}".format(j))

        #TODO: support td drift here
        if j >= 0:
            # Ctrl not drift
            T = self.tslot_time[-1]
            amp_str = "0 if (t >= {}) else {}[int({} * (t/{}))]".format(
                        T, 'ctrlamps', self._num_tslots, T)
#            amp_str = "0 if (t >= {}) else {}[int({} * (t/{})), {}]".format(
#                        T, 'ctrlamps', self._num_tslots, T, j)
            if dg_coeff is not None:
                dg_coeff = "({}*{})".format(amp_str, dg_coeff)
            else:
                dg_coeff = "({})".format(amp_str)

        if dg_coeff is None:
            return dg_op
        else:
            return [dg_op, dg_coeff]

    def _construct_td_dyn_gen(self):
        """
        Make the string type td dynamics generator
        """

        self.tidyup_integ_td()
        dg_comb = [self._build_td_dg(self.drift_dyn_gen)]
        for j, cdg in enumerate(self.ctrl_dyn_gen):
            dg_comb.append(self._build_td_dg(cdg, j))

        return dg_comb

    def solve(self, skip_init=False):
        """
        Solve the evolution with the PWC dynamics generators
        """
        if not self._initialized and not skip_init:
            self.init_solve()

        if self.solver_combines_dyn_gen:
            self.evo_solver.args['ctrlamps'] = self.ctrl_amps.flatten()
            print("Amps: {}".format(self.ctrl_amps.flatten()))
        else:
            self._update_dyn_gen()

        #FIXME: Need to make work with the HEOM solver

        self.evo_solver_result = self.evo_solver.run(initial=self.initial,
                                                     tlist=self.tlist)
        if self.solver_combines_dyn_gen:
            self._integ_tdname = qutip.solver.config.tdname
        self.cost = self.cost_meter.compute_cost(
                                self.evo_solver_result.states[-1], self.target)
        return self.cost





