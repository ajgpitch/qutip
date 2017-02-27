#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 12:50:21 2017

@author: alex
"""
import os
import types
from functools import partial
import numpy as np
import scipy.integrate
from scipy.linalg import norm

from qutip.qobj import Qobj, isket
#from qutip.rhs_generate import rhs_generate
from qutip.solver import Result, Options, config

class SESolver(object):
    """Solves quantum dynamics described by Schrodingers Eq."""
    
    def __init__(self, H, args={}, options=None):
        self.reset()
        self.H = H
        self.args = args
        
    def reset(self):
        self.H = None
        self.args = {}
        self.options = None
        
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
        return initial
        
    def _check_options(self, options=None):
        desc = 'parameter'
        if options is None:
            if self.options is None:
                return Options()
            options = self.options
            desc = 'attribute'
            
        if not isinstance(options, Options):
            raise TypeError("Invalid type {} for {} 'options'. "
                            "Must be of type {}.".format(type(options),
                                                        desc, Options))
        return options

    def _check_tlist(self, tlist=None):
        # Assumes that _check_tslot_duration has already been called
        desc = 'parameter'
        if tlist is None:
            tlist = self.tlist
            desc = 'attribute'

        try:
            tlist = np.array(tlist, dtype='f')
        except Exception as e:
            raise TypeError("Invalid type {} for {} 'tlist'. "
                            "Must be array_like. Attempt at array(tlist) "
                            "raised: {}".format(type(tlist), desc, e))

        return tlist
        
        
    def solve(self, initial, tlist, e_ops=[], options=None):

        options = self._check_options(options)
        initial = self._check_initial(initial)
        tlist = self._check_tlist(tlist)
            
        if not isket(psi0):
            raise TypeError("psi0 must be a ket")

        #
        # setup integrator.
        #
        initial_vector = initial.full().ravel()
        r = scipy.integrate.ode(cy_ode_rhs)
        L = -1.0j * H
        r.set_f_params(L.data.data, L.data.indices, L.data.indptr)  # cython RHS
        r.set_integrator('zvode', method=opt.method, order=opt.order,
                         atol=opt.atol, rtol=opt.rtol, nsteps=opt.nsteps,
                         first_step=opt.first_step, min_step=opt.min_step,
                         max_step=opt.max_step)
    
        r.set_initial_value(initial_vector, tlist[0])          
        #
        # prepare output array
        #
        n_tsteps = len(tlist)
        output = Result()
        output.solver = "sesolve"
        output.times = tlist
    
        if options.store_states:
            output.states = []
    
        if isinstance(e_ops, types.FunctionType):
            n_expt_op = 0
            expt_callback = True
    
        elif isinstance(e_ops, list):
    
            n_expt_op = len(e_ops)
            expt_callback = False
    
            if n_expt_op == 0:
                # fallback on storing states
                output.states = []
                opt.store_states = True
            else:
                output.expect = []
                output.num_expect = n_expt_op
                for op in e_ops:
                    if op.isherm:
                        output.expect.append(np.zeros(n_tsteps))
                    else:
                        output.expect.append(np.zeros(n_tsteps, dtype=complex))
        else:
            raise TypeError("Expectation parameter must be a list or a function")
    
        #
        # start evolution
        #
        progress_bar.start(n_tsteps)
    
        dt = np.diff(tlist)
        for t_idx, t in enumerate(tlist):
            progress_bar.update(t_idx)
    
            if not r.successful():
                raise Exception("ODE integration error: Try to increase "
                                "the allowed number of substeps by increasing "
                                "the nsteps parameter in the Options class.")
    
            if state_norm_func:
                data = r.y / state_norm_func(r.y)
                r.set_initial_value(data, r.t)
    
            if opt.store_states:
                output.states.append(Qobj(r.y, dims=dims))
    
            if expt_callback:
                # use callback method
                e_ops(t, Qobj(r.y, dims=psi0.dims))
    
            for m in range(n_expt_op):
                output.expect[m][t_idx] = cy_expect_psi(e_ops[m].data,
                                                        r.y, e_ops[m].isherm)
    
            if t_idx < n_tsteps - 1:
                r.integrate(r.t + dt[t_idx])
    
        progress_bar.finished()
    
        if not opt.rhs_reuse and config.tdname is not None:
            try:
                os.remove(config.tdname + ".pyx")
            except:
                pass
    
        if opt.store_final_state:
            output.final_state = Qobj(r.y, dims=dims)
    
        return output
    