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
from qutip import Qobj
import qutip.logging_utils as logging
logger = logging.get_logger()
# QuTiP control modules
import qutip.control.optimresult as optimresult
import qutip.control.termcond as termcond
import qutip.control.errors as errors
import qutip.control.dynamics as dynamics
import qutip.control.pulsegen as pulsegen
import qutip.control.dump as qtrldump

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

    Attributes
    ----------
    log_level : integer
        level of messaging output from the logger.
        Options are attributes of qutip.logging_utils,
        in decreasing levels of messaging, are:
        DEBUG_INTENSE, DEBUG_VERBOSE, DEBUG, INFO, WARN, ERROR, CRITICAL
        Anything WARN or above is effectively 'quiet' execution,
        assuming everything runs as expected.
        The default NOTSET implies that the level will be taken from
        the QuTiP settings file, which by default is WARN

    params:  Dictionary
        The key value pairs are the attribute name and value
        Note: attributes are created if they do not exist already,
        and are overwritten if they do.

    alg : string
        Algorithm to use in pulse optimisation.
        Options are:
            'GRAPE' (default) - GRadient Ascent Pulse Engineering
            'CRAB' - Chopped RAndom Basis

    alg_params : Dictionary
        options that are specific to the pulse optim algorithm
        that is GRAPE or CRAB

    disp_conv_msg : bool
        Set true to display a convergence message
        (for scipy.optimize.minimize methods anyway)

    optim_method : string
        a scipy.optimize.minimize method that will be used to optimise
        the pulse for minimum fidelity error

    method_params : Dictionary
        Options for the optim_method.
        Note that where there is an equivalent attribute of this instance
        or the termination_conditions (for example maxiter)
        it will override an value in these options

    approx_grad : bool
        If set True then the method will approximate the gradient itself
        (if it has requirement and facility for this)
        This will mean that the fid_err_grad_wrapper will not get called
        Note it should be left False when using the Dynamics
        to calculate approximate gradients
        Note it is set True automatically when the alg is CRAB

    amp_lbound : float or list of floats
        lower boundaries for the control amplitudes
        Can be a scalar value applied to all controls
        or a list of bounds for each control

    amp_ubound : float or list of floats
        upper boundaries for the control amplitudes
        Can be a scalar value applied to all controls
        or a list of bounds for each control

    bounds : List of floats
        Bounds for the parameters.
        If not set before the run_optimization call then the list
        is built automatically based on the amp_lbound and amp_ubound
        attributes.
        Setting this attribute directly allows specific bounds to be set
        for individual parameters.
        Note: Only some methods use bounds

    dynamics : Dynamics (subclass instance)
        describes the dynamics of the (quantum) system to be control optimised
        (see Dynamics classes for details)

    config : OptimConfig instance
        various configuration options
        (see OptimConfig for details)

    termination_conditions : TerminationCondition instance
        attributes determine when the optimisation will end

    pulse_generator : PulseGen (subclass instance)
        (can be) used to create initial pulses
        not used by the class, but set by pulseoptim.create_pulse_optimizer

    stats : Stats
        attributes of which give performance stats for the optimisation
        set to None to reduce overhead of calculating stats.
        Note it is (usually) shared with the Dynamics instance

    dump : :class:`dump.OptimDump`
        Container for data dumped during the optimisation.
        Can be set by specifying the dumping level or set directly.
        Note this is mainly intended for user and a development debugging
        but could be used for status information during a long optimisation.

    dumping : string
        level of data dumping: NONE, SUMMARY, FULL or CUSTOM
        See property docstring for details

    dump_to_file : bool
        If set True then data will be dumped to file during the optimisation
        dumping will be set to SUMMARY during init_optim
        if dump_to_file is True and dumping not set.
        Default is False

    dump_dir : string
        Basically a link to dump.dump_dir. Exists so that it can be set through
        optim_params.
        If dump is None then will return None or will set dumping to SUMMARY
        when setting a path

    iter_summary : :class:`OptimIterSummary`
        Summary of the most recent iteration.
        Note this is only set if dummping is on
    
    """

    def __init__(self, ctrl_solver):
        self.reset()
        self.ctrl_solver = ctrl_solver
        

    def reset(self):
        self.log_level = self.config.log_level
        self.id_text = 'OPTIM'
        self.termination_conditions = None
        self.disp_conv_msg = False
        self.iteration_steps = None
        self.record_iteration_steps=False
        self.alg = 'GRAPE'
        self.alg_params = None
        self.method = 'l_bfgs_b'
        self.method_params = None
        self.method_options = None
        self.approx_grad = False
        self.amp_lbound = None
        self.amp_ubound = None
        self.bounds = None
        self.num_iter = 0
        self.num_infidelity_func_calls = 0
        self.num_grad_func_calls = 0
        self.stats = None
        self.wall_time_optim_start = 0.0

        self.dump_to_file = False
        self.dump = None
        self.iter_summary = None

        # AJGP 2015-04-21:
        # These (copying from config) are here for backward compatibility
        if hasattr(self.config, 'amp_lbound'):
            if self.config.amp_lbound:
                self.amp_lbound = self.config.amp_lbound
        if hasattr(self.config, 'amp_ubound'):
            if self.config.amp_ubound:
                self.amp_ubound = self.config.amp_ubound

        self.apply_params()

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
    def dumping(self):
        """
        The level of data dumping that will occur during the optimisation
         - NONE : No processing data dumped (Default)
         - SUMMARY : A summary at each iteration will be recorded
         - FULL : All logs will be generated and dumped
         - CUSTOM : Some customised level of dumping
        When first set to CUSTOM this is equivalent to SUMMARY. It is then up
        to the user to specify which logs are dumped
        """
        if self.dump is None:
            lvl = 'NONE'
        else:
            lvl = self.dump.level

        return lvl

    @dumping.setter
    def dumping(self, value):
        if value is None:
            self.dump = None
        else:
            if not _is_string(value):
                raise TypeError("Value must be string value")
            lvl = value.upper()
            if lvl == 'NONE':
                self.dump = None
            else:
                if not isinstance(self.dump, qtrldump.OptimDump):
                    self.dump = qtrldump.OptimDump(self, level=lvl)
                else:
                    self.dump.level = lvl
    @property
    def dump_dir(self):
        if self.dump:
            return self.dump.dump_dir
        else:
            return None

    @dump_dir.setter
    def dump_dir(self, value):
        if not self.dump:
            self.dumping = 'SUMMARY'
        self.dump.dump_dir = value

    def _create_result(self):
        """
        create the result object
        and set the initial_amps attribute as the current amplitudes
        """
        result = optimresult.OptimResult()
        result.initial_fid_err = self.dynamics.fid_computer.get_fid_err()
        result.initial_amps = self.dynamics.ctrl_amps.copy()
        result.time = self.dynamics.time
        result.optimizer = self
        return result

    def init_optim(self):
        """
        Check optimiser attribute status and passed parameters before
        running the optimisation.
        This is called by run_optimization, but could called independently
        to check the configuration.
        """

        if not isinstance(self.dynamics, dynamics.Dynamics):
            raise TypeError("ctrl_solver not set")
        self.ctrl_solver.init()

        # self.apply_method_params()


        if self.alg == 'CRAB':
            self.approx_grad = True

        if self.stats is not None:
            self.stats.clear()

        self.num_iter = 0
        self.num_infidelity_func_calls = 0
        self.num_grad_func_calls = 0

    def apply_method_params(self, params=None):
        """
        Loops through all the method_params
        (either passed here or the method_params attribute)
        If the name matches an attribute of this object or the
        termination conditions object, then the value of this attribute
        is set. Otherwise it is assumed to a method_option for the
        scipy.optimize.minimize function
        """
        if not params:
            params = self.method_params

        if isinstance(params, dict):
            self.method_params = params
            unused_params = {}
            for key in params:
                val = params[key]
                if hasattr(self, key):
                    setattr(self, key, val)
                if hasattr(self.termination_conditions, key):
                    setattr(self.termination_conditions, key, val)
                else:
                    unused_params[key] = val

            if len(unused_params) > 0:
                if not isinstance(self.method_options, dict):
                    self.method_options = unused_params
                else:
                    self.method_options.update(unused_params)

#    def _build_bounds_list(self):
#        cfg = self.config
#        dyn = self.dynamics
#        n_ctrls = dyn.num_ctrls
#        self.bounds = []
#        for t in range(dyn.num_tslots):
#            for c in range(n_ctrls):
#                if isinstance(self.amp_lbound, list):
#                    lb = self.amp_lbound[c]
#                else:
#                    lb = self.amp_lbound
#                if isinstance(self.amp_ubound, list):
#                    ub = self.amp_ubound[c]
#                else:
#                    ub = self.amp_ubound
#
#                if not lb is None and np.isinf(lb):
#                    lb = None
#                if not ub is None and np.isinf(ub):
#                    ub = None
#
#                self.bounds.append((lb, ub))

    def run_optimization(self, term_conds=None):
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
        self.init()
        term_conds = self.termination_conditions
        dyn = self.dynamics
        cfg = self.config
        self.optim_var_vals = self._get_optim_var_vals()
        st_time = timeit.default_timer()
        self.wall_time_optimize_start = st_time

        if self.stats is not None:
            self.stats.wall_time_optim_start = st_time
            self.stats.wall_time_optim_end = 0.0
            self.stats.num_iter = 0

        if self.bounds is None:
            self._build_bounds_list()

        self._build_method_options()

        result = self._create_result()

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
                self.fid_err_func_wrapper, self.optim_var_vals,
                method=self.method,
                jac=jac,
                bounds=self.bounds,
                options=self.method_options,
                callback=self.iter_step_callback_func)

            amps = self._get_ctrl_amps(opt_res.x)
            dyn.update_ctrl_amps(amps)
            result.termination_reason = opt_res.message
            # Note the iterations are counted in this object as well
            # so there are compared here for interest sake only
            if self.num_iter != opt_res.nit:
                logger.info("The number of iterations counted {} "
                            " does not match the number reported {} "
                            "by {}".format(self.num_iter, opt_res.nit,
                                            self.method))
            result.num_iter = opt_res.nit

        except errors.OptimizationTerminate as except_term:
            self._interpret_term_exception(except_term, result)

        end_time = timeit.default_timer()
        self._add_common_result_attribs(result, st_time, end_time)

        return result
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

    def get_infidelity(self, *args):
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
        self.num_fidelity_func_calls += 1
        # *** update stats ***
        if self.stats is not None:
            self.stats.num_fidelity_func_calls = self.num_fid_func_calls
            if self.log_level <= logging.DEBUG:
                logger.debug("fidelity error call {}".format(
                    self.stats.num_fidelity_func_calls))

        if self.optim_params_changed(args[0]):
            self.ctrl_solver._set_ctrl_amp_params(args[0].copy())
            self.ctrl_solver.solve()
            
        if self.ctrl_solver.infidelity <= self.infidelity_targ:
            raise errors.GoalAchievedTerminate(self.ctrl_solver.infidelity)

        if self.num_fidelity_func_calls > self.max_fidelity_func_calls:
            raise errors.MaxInfidelityCallTerminate()

        return self.ctrl_solver.infidelity

#    def fid_err_grad_wrapper(self, *args):
#        """
#        Get the gradient of the fidelity error with respect to all of the
#        variables, i.e. the ctrl amplidutes in each timeslot
#
#        This is called by generic optimisation algorithm as the gradients of
#        func to the minimised wrt the variables. The argument is the current
#        variable values, i.e. control amplitudes, passed as
#        a flat array. Hence these are reshaped as [nTimeslots, n_ctrls]
#        and then used to update the stored ctrl values (if they have changed)
#
#        Although the optimisation algorithms have a check within them for
#        function convergence, i.e. local minima, the sum of the squares
#        of the normalised gradient is checked explicitly, and the
#        optimisation is terminated if this is below the min_gradient_norm
#        condition
#        """
#        # *** update stats ***
#        self.num_grad_func_calls += 1
#        if self.stats is not None:
#            self.stats.num_grad_func_calls = self.num_grad_func_calls
#            if self.log_level <= logging.DEBUG:
#                logger.debug("gradient call {}".format(
#                    self.stats.num_grad_func_calls))
#        amps = self._get_ctrl_amps(args[0].copy())
#        self.dynamics.update_ctrl_amps(amps)
#        fid_comp = self.dynamics.fid_computer
#        # gradient_norm_func is a pointer to the function set in the config
#        # that returns the normalised gradients
#        grad = fid_comp.get_fid_err_gradient()
#
#        if self.iter_summary:
#            self.iter_summary.grad_func_call_num = self.num_grad_func_calls
#            self.iter_summary.grad_norm = fid_comp.grad_norm
#
#        if self.dump:
#            if self.dump.dump_grad_norm:
#                self.dump.update_grad_norm_log(fid_comp.grad_norm)
#
#            if self.dump.dump_grad:
#                self.dump.update_grad_log(grad)
#
#        tc = self.termination_conditions
#        if fid_comp.grad_norm < tc.min_gradient_norm:
#            raise errors.GradMinReachedTerminate(fid_comp.grad_norm)
#        return grad.flatten()

    def iter_step_callback_func(self, *args):
        """
        Check the elapsed wall time for the optimisation run so far.
        Terminate if this has exceeded the maximum allowed time
        """
        self.num_iter += 1

        if self.log_level <= logging.DEBUG:
            logger.debug("Iteration callback {}".format(self.num_iter))

        wall_time = timeit.default_timer() - self.wall_time_optimize_start

        if self.iter_summary:
            self.iter_summary.iter_num = self.num_iter
            self.iter_summary.wall_time = wall_time

        if self.dump and self.dump.dump_summary:
            self.dump.add_iter_summary()

        tc = self.termination_conditions

        if wall_time > tc.max_wall_time:
            raise errors.MaxWallTimeTerminate()

        # *** update stats ***
        if self.stats is not None:
            self.stats.num_iter = self.num_iter

    def _interpret_term_exception(self, except_term, result):
        """
        Update the result object based on the exception that occurred
        during the optimisation
        """
        result.termination_reason = except_term.reason
        if isinstance(except_term, errors.GoalAchievedTerminate):
            result.goal_achieved = True
        elif isinstance(except_term, errors.MaxWallTimeTerminate):
            result.wall_time_limit_exceeded = True
        elif isinstance(except_term, errors.GradMinReachedTerminate):
            result.grad_norm_min_reached = True
        elif isinstance(except_term, errors.MaxFidFuncCallTerminate):
            result.max_fid_func_exceeded = True

    def _add_common_result_attribs(self, result, st_time, end_time):
        """
        Update the result object attributes which are common to all
        optimisers and outcomes
        """
        dyn = self.dynamics
        result.num_iter = self.num_iter
        result.num_fid_func_calls = self.num_fid_func_calls
        result.wall_time = end_time - st_time
        result.fid_err = dyn.fid_computer.get_fid_err()
        result.grad_norm_final = dyn.fid_computer.grad_norm
        result.final_amps = dyn.ctrl_amps
        final_evo = dyn.full_evo
        if isinstance(final_evo, Qobj):
            result.evo_full_final = final_evo
        else:
            result.evo_full_final = Qobj(final_evo, dims=dyn.sys_dims)
        # *** update stats ***
        if self.stats is not None:
            self.stats.wall_time_optim_end = end_time
            self.stats.calculate()
            result.stats = copy.copy(self.stats)
            
    def compare_optim_params(self, new_amps):
        """
        Determine if any amplitudes have changed. If so, then mark the
        timeslots as needing recalculation
        Returns: True if amplitudes are the same, False if they have changed
        """
        changed = False
        dyn = self.parent
                
        if dyn.ctrl_amps is None:
            # Flag fidelity and gradients as needing recalculation
            changed = True
        else:
            # create boolean array with same shape as ctrl_amps
            # True where value in new_amps differs, otherwise false
            changed_amps = dyn.ctrl_amps != new_amps
            if np.any(changed_amps):
                # Flag fidelity and gradients as needing recalculation
                changed = True
                if self.log_level <= logging.DEBUG:
                    logger.debug("{} amplitudes changed".format(
                        changed_amps.sum()))
                
                if ecs:
                    ecs.num_amps_changed = changed_amps.sum()
                    ecs.num_timeslots_changed = np.any(changed_amps, 1).sum()

            else:
                if self.log_level <= logging.DEBUG:
                    logger.debug("No amplitudes changed")

        # *** update stats ***
        if dyn.stats:
            dyn.stats.num_ctrl_amp_updates += bool(ecs.num_amps_changed)
            dyn.stats.num_ctrl_amp_changes += ecs.num_amps_changed
            dyn.stats.num_timeslot_changes += ecs.num_timeslots_changed
            
        if changed:
            dyn.ctrl_amps = new_amps
            dyn.flag_system_changed()
            return False
        else:
            return True


class OptimIterSummary(qtrldump.DumpSummaryItem):
    """A summary of the most recent iteration of the pulse optimisation

    Attributes
    ----------
    iter_num : int
        Iteration number of the pulse optimisation

    fid_func_call_num : int
        Fidelity function call number of the pulse optimisation

    grad_func_call_num : int
        Gradient function call number of the pulse optimisation

    fid_err : float
        Fidelity error

    grad_norm : float
        fidelity gradient (wrt the control parameters) vector norm
        that is the magnitude of the gradient

    wall_time : float
        Time spent computing the pulse optimisation so far
        (in seconds of elapsed time)
    """
    # Note there is some duplication here with Optimizer attributes
    # this exists solely to be copied into the summary dump
    min_col_width = 11
    summary_property_names = (
        "idx", "iter_num", "fid_func_call_num", "grad_func_call_num",
        "fid_err", "grad_norm", "wall_time"
        )

    summary_property_fmt_type = (
        'd', 'd', 'd', 'd',
        'g', 'g', 'g'
        )

    summary_property_fmt_prec = (
        0, 0, 0, 0,
        4, 4, 2
        )

    def __init__(self):
        self.reset()

    def reset(self):
        qtrldump.DumpSummaryItem.reset(self)
        self.iter_num = None
        self.fid_func_call_num = None
        self.grad_func_call_num = None
        self.fid_err = None
        self.grad_norm = None
        self.wall_time = 0.0
