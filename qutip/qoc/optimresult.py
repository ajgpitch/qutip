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
Class containing the results of the pulse optimisation
"""

import numpy as np


class OptimResult(object):
    """
    Attributes give the result of the pulse optimisation attempt

    Attributes
    ----------
    termination_reason : string
        Description of the reason for terminating the optimisation

    initial_cost : float
        Cost before optimisation starting

    final_cost : float
        final cost value that was achieved

    goal_achieved : boolean
        True is the fidely error achieved was below the target

    grad_norm_final : float
        Final value of the sum of the squares of the (normalised) fidelity
        error gradients

    grad_norm_min_reached : float
        True if the optimisation terminated due to the minimum value
        of the gradient being reached

    num_iter : integer
        Number of iterations of the optimisation algorithm completed

    max_iter_exceeded : boolean
        True if the iteration limit was reached

    max_fid_func_exceeded : boolean
        True if the fidelity function call limit was reached

    wall_time : float
        time elapsed during the optimisation

    wall_time_limit_exceeded  : boolean
        True if the wall time limit was reached

    time : array[num_tslots+1] of float
        Time are the start of each timeslot
        with the final value being the total evolution time

    initial_optim_params : array[num_tslots, n_ctrls]
        The amplitudes at the start of the optimisation

    final_optim_params : array[num_tslots, n_ctrls]
        The amplitudes at the end of the optimisation

    evo_full_final : Qobj
        The evolution operator from t=0 to t=T based on the final amps

    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.initial_cost = None
        self.final_cost = None
        self.goal_achieved = False
        self.grad_norm_final = 0.0
        self.grad_norm_min_reached = False
        self.num_iter = 0
        self.max_iter_exceeded = False
        self.num_infidelity_calls = 0
        self.max_infidelity_calls_exceeded = False
        self.wall_time = 0.0
        self.wall_time_limit_exceeded = False
        self.termination_reason = "not started yet"
        self.initial_optim_params = None
        self.final_optim_params = None
        self.final_evo = None

    def __str__(self):
        if self.goal_achieved:
            s = "Optimisation goal achieved."
        else:
            s = ("Failed to achieve optimisation goal, reason: "
                "{}.".format(self.termination_reason))
        s = "{} {} iterations, final cost {}".format(s, self.num_iter,
                                                     self.final_cost)
        return s

    def _end_optim_update(self, optim):
        """
        Update the result object attributes which are common to all
        optimisers and outcomes
        """
        self.num_iter = optim.num_iter
        self.num_cost_calls = optim.num_cost_calls
        self.wall_time = optim.wall_time_optim_end - optim.wall_time_optim_start
        self.final_cost = optim.ctrl_solver.cost
        # FIXME:
        #self.grad_norm_final = dyn.fid_computer.grad_norm
        self.final_optim_params = optim.ctrl_solver._get_optim_params()
#        final_evo = dyn.full_evo
#        if isinstance(final_evo, Qobj):
#            result.evo_full_final = final_evo
#        else:
#            result.evo_full_final = Qobj(final_evo, dims=dyn.sys_dims)