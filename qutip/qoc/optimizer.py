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
Classes here are expected to implement a run_optimization function
that will use some method for optimising the control pulse, as defined
by the control amplitudes. The system that the pulse acts upon are defined
by the Dynamics object that must be passed in the instantiation.

The methods are typically N dimensional function optimisers that
find the minima of a fidelity error function. Note the number of variables
for the fidelity function is the number of control timeslots,
i.e. n_ctrls x Ntimeslots
The methods will call functions on the Dynamics.fid_computer object,
one or many times per interation,
to get the fidelity error and gradient wrt to the amplitudes.
The optimisation will stop when one of the termination conditions are met,
for example: the fidelity aim has be reached, a local minima has been found,
the maximum time allowed has been exceeded

These function optimisation methods are so far from SciPy.optimize
The two methods implemented are:

    BFGS - Broyden–Fletcher–Goldfarb–Shanno algorithm

        This a quasi second order Newton method. It uses successive calls to
        the gradient function to make an estimation of the curvature (Hessian)
        and hence direct its search for the function minima
        The SciPy implementation is pure Python and hance is execution speed is
        not high
        use subclass: OptimizerBFGS

    L-BFGS-B - Bounded, limited memory BFGS

        This a version of the BFGS method where the Hessian approximation is
        only based on a set of the most recent gradient calls. It generally
        performs better where the are a large number of variables
        The SciPy implementation of L-BFGS-B is wrapper around a well
        established and actively maintained implementation in Fortran
        Its is therefore very fast.
        # See SciPy documentation for credit and details on the
        # scipy.optimize.fmin_l_bfgs_b function
        use subclass: OptimizerLBFGSB

The baseclass Optimizer implements the function wrappers to the
fidelity error, gradient, and iteration callback functions.
These are called from the within the SciPy optimisation functions.
The subclasses implement the algorithm specific pulse optimisation function.
"""

import os
import numpy as np
import timeit
import scipy.optimize as spopt
import copy
import collections
# QuTiP
import qutip.settings as qset
from qutip import Qobj
import qutip.logging_utils as logging
logger = logging.get_logger()
# QuTiP control modules
import qutip.qoc.optimresult as optimresult
import qutip.qoc.terminator as terminator
import qutip.qoc.ctrlsolver as ctrlsolver
#import qutip.qoc.pulsegen as pulsegen

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

class Optimizer(object):
    """
    Base class for all control pulse optimisers. This class should not be
    instantiated, use its subclasses
    This class implements the fidelity, gradient and interation callback
    functions.
    All subclass objects must be initialised with a

        OptimConfig instance - various configuration options
        Dynamics instance - describes the dynamics of the (quantum) system
                            to be control optimised

    Parameters
    ----------

    Attributes
    ----------

    """

    def __init__(self, ctrl_solver):
        self.reset()
        self.ctrl_solver = ctrl_solver

    def reset(self):
        self.log_level = 20
        self.disp_conv_msg = False
        self._method = 'L-BFGS-B'
        self.alg = 'GRAPE'
        self.param_atol = qset.atol
        #FIXME: make pvt
        self.optim_params = None
        self.approx_grad = False
        self.amp_lbound = None
        self.amp_ubound = None
        self.bounds = None
        self.num_iter = 0
        self.num_cost_evals = 0
        self.num_grad_evals = 0
        self.wall_time_optim_start = 0.0
        self.wall_time_optim_end = 0.0
        self.method_options = None

        # Default termination conditions
        self.cost_target = 1.0e-6
        # maxiter
        self.max_iter = 1000
        # maxfun
        self.max_cost_evals = 10000
        self.max_wall_time = 600.0
        # ftol (note scipy default is 2.220446049250313e-09
        self.cost_eval_tol = None
        # gtol
        self.grad_eval_tol = None
        # eps
        self.approx_grad_step_size = None


    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, value):
        try:
            self._method = value.upper()
        except Exception as e:
            raise e.__class__("Invalid value '{}' for Optimizer method"
                              ". {}".format(value, e))

    def _get_method_options_from_attribs(self):
        """Create method_options dictionary for the scipy.optimize.minimize
        function based on the attributes of this object.
        """

        # The only option that is common to all methods is maxiter
        mo = {'maxiter': self.max_iter}

        def add_if_not_none(key, value):
            if value is not None:
                mo[key] = value

        if self.method == 'L-BFGS-B':
            add_if_not_none('maxfun', self.max_cost_evals)
            add_if_not_none('ftol', self.cost_eval_tol)
            add_if_not_none('gtol', self.grad_eval_tol)
            add_if_not_none('eps', self.approx_grad_step_size)

        #TODO: Add other methods that we wish to support

        return mo

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

    def _create_result(self):
        """
        create the result object
        and set the initial_amps attribute as the current amplitudes
        """
        result = optimresult.OptimResult()
        return result

    def init_optim(self, method_options=None):
        """
        Check optimiser attribute status and passed parameters before
        running the optimisation.
        This is called by run_optimization, but could called independently
        to check the configuration.
        """

        if not isinstance(self.ctrl_solver, ctrlsolver.ControlSolver):
            raise TypeError("ctrl_solver not set")
        self.ctrl_solver.init_solve()

        # self.apply_method_params()
        self._build_bounds_list()

        self.approx_grad = True
        self.num_iter = 0
        self.num_cost_evals = 0
        self.num_grad_evals = 0

        self.method_options = self._get_method_options_from_attribs()
        if not method_options is None:
            for key in method_options:
                self.method_options[key] = method_options[key]

    def optim_end(self):
        if self.ctrl_solver.integ_rhs_tidyup:
            self.ctrl_solver.tidyup_integ_td()

    def _build_bounds_list(self):
        # This assumes that the ctrl_solver amp_lbound and amp_ubound
        # have been checked to be either None, float or array_like
        # and equal to the number of controls
        csol = self.ctrl_solver
        bounds = []
        all_none = True
        #TODO this is gonna have to be shifted to ctrlsolver
        if hasattr(csol, 'num_tslots'):
            for k in range(csol.num_tslots):
                for j in range(csol.num_ctrls):
                    if csol.amp_lbound is None:
                        lb = None
                    else:
                        if hasattr(csol.amp_lbound, 'len'):
                            lb = csol.amp_lbound[j]
                        else:
                            lb = csol.amp_lbound

                    if csol.amp_ubound is None:
                        ub = None
                    else:
                        if hasattr(csol.amp_ubound, 'len'):
                            ub = csol.amp_ubound[j]
                        else:
                            ub = csol.amp_ubound

                    if (lb is not None) or (ub is not None):
                        all_none = False

                    bounds.append((lb, ub))

        if all_none:
            self.bounds = None
        else:
            self.bounds = bounds

    def optimize_ctrls(self, method_options=None):
        """
        This default function optimisation method is a wrapper to the
        scipy.optimize.minimize function.

        It will attempt to minimise the fidelity error with respect to some
        parameters, which are determined by _get_optim_var_vals (see below)

        The optimisation end when one of the passed termination conditions
        has been met, e.g. target achieved, wall time, or
        function call or iteration count exceeded. Note these
        conditions include gradient minimum met (local minima) for
        methods that use a gradient.

        The function minimisation method is taken from the optim_method
        attribute. Note that not all of these methods have been tested.
        Note that some of these use a gradient and some do not.
        See the scipy documentation for details. Options specific to the
        method can be passed setting the method_params attribute.

        If the parameter term_conds=None, then the termination_conditions
        attribute must already be set. It will be overwritten if the
        parameter is not None

        The result is returned in an OptimResult object, which includes
        the final fidelity, time evolution, reason for termination etc

        """
        print("Start optim")
        self.wall_time_optim_start = timeit.default_timer()
        self.init_optim()
        # Init these to None so that all are seen to have changed on first
        # cost call.
        self.optim_params = self.ctrl_solver.get_optim_params()
#        init_params =
#        self.num_params = len(init_params)
        print("initial params:\n{}".format(self.optim_params))

        self.result = self._create_result()

        if self.approx_grad:
            jac=None
        else:
            jac=self.fid_err_grad_wrapper

        if self.log_level <= logging.INFO:
            msg = ("Optimising pulse(s) using {} with "
                        "minimise '{}' method").format(self.alg, self.method)
            if self.approx_grad:
                msg += " (approx grad)"
            logger.info(msg)

        try:
            opt_res = spopt.minimize(
                self._get_cost, self.optim_params,
                method=self.method,
                jac=jac,
                bounds=self.bounds,
                options=self.method_options,
                callback=self._iter_step)

            self._update_ctrl_params(opt_res.x)
            self.result.termination_reason = opt_res.message
            # Note the iterations are counted in this object as well
            # so there are compared here for interest sake only
            if self.num_iter != opt_res.nit:
                logger.info("The number of iterations counted {} "
                            " does not match the number reported {} "
                            "by {}".format(self.num_iter, opt_res.nit,
                                            self.method))

        except terminator.OptimizationTerminate as except_term:
            self._interpret_term_exception(except_term, self.result)

        self.wall_time_optim_end = timeit.default_timer()
        self.result._end_optim_update(self)

        self.optim_end()
        return self.result
#
#    def _get_optim_var_vals(self):
#        """
#        Generate the 1d array that holds the current variable values
#        of the function to be optimised
#        By default (as used in GRAPE) these are the control amplitudes
#        in each timeslot
#        """
#        return self.dynamics.ctrl_amps.reshape([-1])
#
#    def _get_ctrl_amps(self, optim_var_vals):
#        """
#        Get the control amplitudes from the current variable values
#        of the function to be optimised.
#        that is the 1d array that is passed from the optimisation method
#        Note for GRAPE these are the function optimiser parameters
#        (and this is the default)
#
#        Returns
#        -------
#        float array[dynamics.num_tslots, dynamics.num_ctrls]
#        """
#        amps = optim_var_vals.reshape(self.dynamics.ctrl_amps.shape)
#
#        return amps

    def _get_cost(self, *args):
        """
        Get the fidelity error achieved using the ctrl amplitudes passed
        in as the first argument.

        This is called by generic optimisation algorithm as the
        func to the minimised. The argument is the current
        variable values, i.e. control amplitudes, passed as
        a flat array. Hence these are reshaped as [nTimeslots, n_ctrls]
        and then used to update the stored ctrl values (if they have changed)

        The error is checked against the target, and the optimisation is
        terminated if the target has been achieved.
        """
        self.num_cost_evals += 1
        # *** update stats ***
        # TODO: Update stats

        #print(args[0])
        self._update_ctrl_params(args[0])

        if not self.ctrl_solver.is_solution_current:
            self.ctrl_solver.solve()

        #print("Cost {}".format(self.ctrl_solver.cost))

        if self.result.initial_cost is None:
            # Assume this is the first solve
            self.result.initial_cost = self.ctrl_solver.cost
            self.result.initial_optim_params = args[0].copy()

        if self.ctrl_solver.cost <= self.cost_target:
            raise terminator.GoalAchievedTerminate(self.ctrl_solver.cost)

        if self.num_cost_evals > self.max_cost_evals:
            raise terminator.MaxCostEvalTerminate()


        return self.ctrl_solver.cost

    def _iter_step(self, *args):
        """
        Check the elapsed wall time for the optimisation run so far.
        Terminate if this has exceeded the maximum allowed time
        """
        self.num_iter += 1

        print("Iter {}".format(self.num_iter))
        if self.log_level <= logging.DEBUG:
            logger.debug("Iteration callback {}".format(self.num_iter))

        wall_time = timeit.default_timer() - self.wall_time_optim_start

        # This not reallyunnessary, as handled in scipy, but here anyway
        if self.num_iter > self.max_iter:
            raise terminator.MaxIterTerminate()

        if wall_time > self.max_wall_time:
            raise terminator.MaxWallTimeTerminate()

    def _interpret_term_exception(self, except_term, result):
        """
        Update the result object based on the exception that occurred
        during the optimisation
        """
        result.termination_reason = except_term.reason
        if isinstance(except_term, terminator.GoalAchievedTerminate):
            result.goal_achieved = True
        elif isinstance(except_term, terminator.MaxWallTimeTerminate):
            result.wall_time_limit_exceeded = True
        elif isinstance(except_term, terminator.GradMinReachedTerminate):
            result.grad_norm_min_reached = True
        elif isinstance(except_term, terminator.MaxIterTerminate):
            result.max_iter_exceeded = True
        elif isinstance(except_term, terminator.MaxCostEvalTerminate):
            result.max_cost_eval_exceeded = True

    def _compare_optim_params(self, new_params):
        """
        Determine if any parameters have changed.

        Parameters
        ----------
        new_params : ndarray
            Optimisation parameters to compare

        Returns
        -------
#        num_changed : int
#            Number of params changed

        changed_index : ndarray
            bool index of changed parameters
        """

        if self.optim_params is None:
            #num_changed = num_params
            changed_index = np.ones((self.num_params), dtype=bool)
        else:
            changed_index = (np.abs(self.optim_params - new_params) >
                                                            self.param_atol)

        if self.log_level <= logging.DEBUG:
            if np.any(changed_index):
                logger.debug("{} optim params changed".format(
                    np.count_nonzero(changed_index)))
            else:
                logger.debug("No optim params changed")

        return changed_index

    def _update_ctrl_params(self, new_params):
        changed_param_index = self._compare_optim_params(new_params)
        if np.any(changed_param_index):
            #print("Set amps from params: {}".format(args[0]))
            #FIXME: Try removing this copy()
            self.ctrl_solver.set_ctrl_amp_params(new_params.copy(),
                                                  changed_param_index)
        else:
            print("Nothing changed")
